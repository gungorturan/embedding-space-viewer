import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { ServerBackend, StaticBackend } from './backend.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const MOD_COLORS = { i: '#4c9aff', t: '#ff9d42' };
const PALETTE = ['#4c9aff', '#ff9d42', '#41c98a', '#ff5c7a', '#b18cff', '#e8d44d', '#3fd0d4', '#ff7ad9', '#9bd55a', '#c9a27c'];
const PAIR_COLOR = '#41c98a';
const K = 12;
const METHOD_LABEL = { pca: 'PCA', umap: 'UMAP', tsne: 't-SNE', rtdae: 'RTD-AE' };

// Renkler sRGB olarak doğrudan shader'a gider (ShaderMaterial renk dönüşümü yapmaz)
const rgb = (hex) => [1, 3, 5].map((o) => parseInt(hex.slice(o, o + 2), 16) / 255);

// docs/ altındaki statik sürümde <html data-mode="static"> bulunur
const STATIC = document.documentElement.dataset.mode === 'static';
const backend = STATIC ? new StaticBackend((msg) => busy(msg)) : new ServerBackend();

// ------------------------------------------------------------ tercihler
const PREF_KEY = 'esv-prefs';
const prefs = (() => { try { return JSON.parse(localStorage.getItem(PREF_KEY)) || {}; } catch { return {}; } })();
const savePrefs = () => { try { localStorage.setItem(PREF_KEY, JSON.stringify(prefs)); } catch { /* yoksay */ } };

// ------------------------------------------------------------ durum
const S = {
  cfg: null, space: null,
  method: prefs.method || 'pca',
  pointMode: prefs.pointMode || 'dot',
  colorBy: prefs.colorBy || 'modality',
  size: prefs.size || 10,
  show: { i: true, t: true },
  showGap: true,
  selected: -1, detail: null, neighborScope: 'all',
  query: null, queryScope: 'image',
  gap: null, info: null, busy: false,
};

// ------------------------------------------------------------ sahne
const wrap = $('#canvas-wrap');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0e1116);
wrap.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
Object.assign(labelRenderer.domElement.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
wrap.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
const HOME = new THREE.Vector3(0.6, 0.4, 3.2);
camera.position.copy(HOME);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotateSpeed = 0.8;

const axes = new THREE.AxesHelper(0.2);
axes.material.transparent = true;
axes.material.opacity = 0.5;
scene.add(axes);

new ResizeObserver(() => {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  labelRenderer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}).observe(wrap);

const VERT = /* glsl */`
  attribute vec3 aColor; attribute float aKind; attribute float aCell; attribute float aState;
  uniform float uSize, uPR, uSprite, uOverlay;
  varying vec3 vColor; varying float vKind, vCell, vState;
  void main() {
    vColor = aColor; vKind = aKind; vCell = aCell; vState = aState;
    if (aState < -0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float s = uSize;
    if (aKind < 0.5 && uSprite > 0.5) s *= 2.4;
    if (uOverlay > 0.5) s *= 1.5;
    gl_PointSize = s * uPR * (2.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const FRAG = /* glsl */`
  uniform sampler2D uAtlas; uniform vec2 uGrid; uniform float uSprite, uDim, uOverlay;
  varying vec3 vColor; varying float vKind, vCell, vState;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float a = (vState > 0.5) ? uDim : 1.0;
    vec3 col = vColor;
    float edge;
    if (vKind < 0.5 && uSprite > 0.5 && vCell >= 0.0) {
      float c = mod(vCell, uGrid.x), r = floor(vCell / uGrid.x);
      vec2 uv = vec2((c + gl_PointCoord.x) / uGrid.x, 1.0 - (r + gl_PointCoord.y) / uGrid.y);
      edge = max(abs(p.x), abs(p.y));
      col = edge > 0.86 ? vColor : texture2D(uAtlas, uv).rgb;
    } else if (vKind < 0.5) {
      edge = length(p);
      if (edge > 1.0) discard;
      a *= smoothstep(1.0, 0.8, edge);
    } else {
      edge = abs(p.x) + abs(p.y);
      if (edge > 1.0) discard;
      a *= smoothstep(1.0, 0.85, edge);
    }
    if (uOverlay > 0.5 && edge > 0.74) col = vec3(1.0);
    gl_FragColor = vec4(col, a);
  }`;

const makeMat = (overlay) => new THREE.ShaderMaterial({
  uniforms: {
    uAtlas: { value: null }, uGrid: { value: new THREE.Vector2(1, 1) },
    uSprite: { value: 0 }, uSize: { value: S.size }, uPR: { value: renderer.getPixelRatio() },
    uDim: { value: 0.1 }, uOverlay: { value: overlay ? 1 : 0 },
  },
  vertexShader: VERT, fragmentShader: FRAG, transparent: true,
  depthTest: !overlay, depthWrite: false,
});
const cloudMat = makeMat(false);
const overlayMat = makeMat(true);

let N = 0;
let cloud = null, overlay = null;
let pos = new Float32Array(0);           // anlık (animasyonlu) konumlar
let anim = null;                         // {from, to, t0}
let hl = [];                             // vurgulanan indeksler

const linkGeo = new THREE.BufferGeometry();
const links = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.75, depthTest: false,
}));
links.renderOrder = 5;
links.frustumCulled = false;
scene.add(links);
let linkList = [];                       // [a, b, hex]; a === -1 → sorgu işaretçisi

const queryMarker = new THREE.Mesh(new THREE.OctahedronGeometry(0.035),
  new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
queryMarker.renderOrder = 10;
queryMarker.visible = false;
const queryLabelEl = Object.assign(document.createElement('div'), { className: 'label3d' });
const queryLabel = new CSS2DObject(queryLabelEl);
queryLabel.position.set(0, 0.07, 0);
queryMarker.add(queryLabel);
scene.add(queryMarker);

// Modality gap: iki modalite merkezinin 3B izdüşümü ve aralarındaki ok
const gapGroup = new THREE.Group();
gapGroup.renderOrder = 8;
const mkCentroid = (hex, text) => {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.03, 20, 14),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(hex), depthTest: false }));
  m.renderOrder = 8;
  const el = Object.assign(document.createElement('div'), { className: 'label3d', textContent: text });
  const l = new CSS2DObject(el);
  l.position.set(0, 0.06, 0);
  m.add(l);
  gapGroup.add(m);
  return m;
};
const cImg = mkCentroid(MOD_COLORS.i, 'μ görüntü');
const cTxt = mkCentroid(MOD_COLORS.t, 'μ metin');
const gapArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xffffff, 0.06, 0.035);
gapArrow.line.material.depthTest = false;
gapArrow.cone.material.depthTest = false;
gapArrow.renderOrder = 8;
gapGroup.add(gapArrow);
gapGroup.visible = false;
scene.add(gapGroup);

// ------------------------------------------------------------ bulut
function buildCloud() {
  if (cloud) { scene.remove(cloud); cloud.geometry.dispose(); }
  if (overlay) { scene.remove(overlay); overlay.geometry.dispose(); }
  const items = S.space.items;
  N = items.length;
  pos = new Float32Array(N * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  geo.setAttribute('aKind', new THREE.BufferAttribute(Float32Array.from(items, (it) => (it.m === 'i' ? 0 : 1)), 1));
  geo.setAttribute('aCell', new THREE.BufferAttribute(Float32Array.from(items, (it, i) => (it.m === 'i' ? i : -1)), 1));
  geo.setAttribute('aState', new THREE.BufferAttribute(new Float32Array(N), 1));
  cloud = new THREE.Points(geo, cloudMat);
  cloud.frustumCulled = false;
  scene.add(cloud);

  const og = new THREE.BufferGeometry();
  for (const [name, size] of [['position', 3], ['aColor', 3], ['aKind', 1], ['aCell', 1], ['aState', 1]]) {
    og.setAttribute(name, new THREE.BufferAttribute(new Float32Array(64 * size), size));
  }
  og.setDrawRange(0, 0);
  overlay = new THREE.Points(og, overlayMat);
  overlay.frustumCulled = false;
  overlay.renderOrder = 6;
  scene.add(overlay);

  const g = S.space.atlas;
  new THREE.TextureLoader().load(backend.atlasUrl(), (tex) => {
    tex.minFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    for (const m of [cloudMat, overlayMat]) {
      m.uniforms.uAtlas.value = tex;
      m.uniforms.uGrid.value.set(g.cols, g.rows);
    }
  });
  applyColors();
  applyStates();
}

function colorOf(i) {
  const it = S.space.items[i];
  if (S.colorBy === 'label' && it.label != null) {
    return PALETTE[S.space.classes.indexOf(it.label) % PALETTE.length];
  }
  return MOD_COLORS[it.m];
}

function applyColors() {
  if (!cloud) return;
  const a = cloud.geometry.attributes.aColor;
  for (let i = 0; i < N; i++) a.array.set(rgb(colorOf(i)), i * 3);
  a.needsUpdate = true;
  syncOverlay();
  renderLegend();
}

function applyStates() {
  if (!cloud) return;
  const st = cloud.geometry.attributes.aState;
  const set = new Set(hl);
  const items = S.space.items;
  for (let i = 0; i < N; i++) {
    st.array[i] = !S.show[items[i].m] ? -1 : (set.size && !set.has(i) ? 1 : 0);
  }
  st.needsUpdate = true;
  // Görüntü sprite'ları opak olduğunda derinlik yazmak üst üste binmeyi düzgün gösterir
  cloudMat.depthWrite = S.pointMode === 'sprite' && !set.size;
  syncOverlay();
}

function syncOverlay() {
  if (!overlay) return;
  const idx = hl.filter((i) => S.show[S.space.items[i].m]).slice(0, 64);
  const g = overlay.geometry.attributes;
  const src = cloud.geometry.attributes;
  idx.forEach((i, k) => {
    g.position.array.set(pos.subarray(i * 3, i * 3 + 3), k * 3);
    g.aColor.array.set(src.aColor.array.subarray(i * 3, i * 3 + 3), k * 3);
    g.aKind.array[k] = src.aKind.array[i];
    g.aCell.array[k] = src.aCell.array[i];
    g.aState.array[k] = 0;
  });
  for (const a of Object.values(g)) a.needsUpdate = true;
  overlay.geometry.setDrawRange(0, idx.length);
}

function setHighlight(indices, linkSpec) {
  hl = [...new Set(indices)];
  linkList = linkSpec;
  applyStates();
  updateLinks();
}

function pointAt(i, out) {
  if (i === -1) return out.copy(queryMarker.position);
  return out.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
}

const _a = new THREE.Vector3(), _b = new THREE.Vector3();
function updateLinks() {
  const p = new Float32Array(linkList.length * 6);
  const c = new Float32Array(linkList.length * 6);
  linkList.forEach(([a, b, hex], k) => {
    pointAt(a, _a).toArray(p, k * 6);
    pointAt(b, _b).toArray(p, k * 6 + 3);
    const col = rgb(hex);
    c.set(col, k * 6);
    c.set(col, k * 6 + 3);
  });
  linkGeo.setAttribute('position', new THREE.BufferAttribute(p, 3));
  linkGeo.setAttribute('color', new THREE.BufferAttribute(c, 3));
}

function animateTo(coords) {
  anim = { from: pos.slice(), to: coords, t0: performance.now() };
}

// Kamerayı bakış yönünü koruyarak hedef noktaya yaklaştırır
let fly = null;
function flyTo(target, dist) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  fly = {
    t0: performance.now(),
    fromT: controls.target.clone(), toT: target.clone(),
    fromP: camera.position.clone(), toP: target.clone().add(dir.multiplyScalar(dist)),
  };
}

// ------------------------------------------------------------ render döngüsü
function frame(now) {
  requestAnimationFrame(frame);
  if (anim && cloud) {
    const t = Math.min(1, (now - anim.t0) / 800);
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    for (let i = 0; i < pos.length; i++) pos[i] = anim.from[i] + (anim.to[i] - anim.from[i]) * e;
    cloud.geometry.attributes.position.needsUpdate = true;
    syncOverlay();
    updateLinks();
    if (t === 1) anim = null;
  }
  if (fly) {
    const t = Math.min(1, (now - fly.t0) / 600);
    const e = 1 - (1 - t) ** 3;
    controls.target.lerpVectors(fly.fromT, fly.toT, e);
    camera.position.lerpVectors(fly.fromP, fly.toP, e);
    if (t === 1) fly = null;
  }
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}
requestAnimationFrame(frame);

// ------------------------------------------------------------ seçme (ekran uzayında)
const _m = new THREE.Matrix4();
function pick(mx, my) {
  if (!cloud) return -1;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  _m.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = _m.elements;
  const items = S.space.items;
  const sprite = S.pointMode === 'sprite';
  let best = -1, bestW = Infinity, bestD = Infinity;
  for (let i = 0; i < N; i++) {
    const it = items[i];
    if (!S.show[it.m]) continue;
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    const cw = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (cw <= 0) continue;
    const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / cw * 0.5 + 0.5) * w;
    const sy = (0.5 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / cw * 0.5) * h;
    const d2 = (sx - mx) ** 2 + (sy - my) ** 2;
    const r = Math.max(5, (S.size * (sprite && it.m === 'i' ? 2.4 : 1)) / cw);
    if (d2 > r * r) continue;
    // Sprite modunda öndeki, nokta modunda imlece en yakın olan kazanır
    if (sprite ? cw < bestW : d2 < bestD) { best = i; bestW = cw; bestD = d2; }
  }
  return best;
}

const tooltip = $('#tooltip');
let hover = -1, down = null;
wrap.addEventListener('pointermove', (ev) => {
  const r = wrap.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const i = down ? -1 : pick(mx, my);
  if (i !== hover) {
    hover = i;
    if (i >= 0) tooltip.innerHTML = tooltipHtml(i);
  }
  tooltip.hidden = i < 0;
  if (i >= 0) {
    const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
    tooltip.style.left = `${Math.min(mx + 14, r.width - tw - 6)}px`;
    tooltip.style.top = `${Math.min(my + 14, r.height - th - 6)}px`;
  }
  wrap.style.cursor = i >= 0 ? 'pointer' : 'crosshair';
});
wrap.addEventListener('pointerleave', () => { tooltip.hidden = true; hover = -1; });
wrap.addEventListener('pointerdown', (ev) => { down = { x: ev.clientX, y: ev.clientY }; tooltip.hidden = true; });
wrap.addEventListener('pointerup', (ev) => {
  if (!down) return;
  const moved = Math.hypot(ev.clientX - down.x, ev.clientY - down.y);
  down = null;
  if (moved > 4) return;
  const r = wrap.getBoundingClientRect();
  const i = pick(ev.clientX - r.left, ev.clientY - r.top);
  if (i >= 0) select(i); else clearSelection();
});
wrap.addEventListener('dblclick', (ev) => {
  const r = wrap.getBoundingClientRect();
  const i = pick(ev.clientX - r.left, ev.clientY - r.top);
  if (i >= 0) flyTo(pointAt(i, new THREE.Vector3()), 0.45);
});

function tooltipHtml(i) {
  const it = S.space.items[i];
  const meta = `${it.m === 'i' ? 'Görüntü' : 'Metin'} #${i}${it.label ? ` · ${esc(it.label)}` : ''}`;
  if (it.m === 'i') return `<img src="${thumbUrl(i)}"><div class="meta">${meta}</div>`;
  return `<div class="t">${esc(it.text)}</div><div class="meta">${meta}</div>`;
}

const thumbUrl = (i) => backend.thumbUrl(i);

// ------------------------------------------------------------ seçim & sorgu
async function select(i) {
  S.selected = i;
  S.query = null;
  queryMarker.visible = false;
  try {
    S.detail = await backend.item(i, K, S.neighborScope);
  } catch (e) { return showError(e); }
  if (S.selected !== i) return;
  const nb = S.detail.neighbors.map((n) => n.id);
  const pr = S.detail.pairs.map((p) => p.id);
  setHighlight([i, ...pr, ...nb], [
    ...nb.map((j) => [i, j, '#c8d0dc']),
    ...pr.map((j) => [i, j, PAIR_COLOR]),
  ]);
  renderDetail();
}

function clearSelection() {
  S.selected = -1;
  S.detail = null;
  S.query = null;
  queryMarker.visible = false;
  setHighlight([], []);
  renderDetail();
}

async function runQuery(text) {
  if (!S.space || !text.trim()) return;
  S.selected = -1;
  S.detail = null;
  let r;
  try {
    r = await backend.query({ ...projBody(), text, k: K, scope: S.queryScope });
  } catch (e) { return showError(e); }
  S.query = { text, ...r };
  queryMarker.position.fromArray(r.pos);
  queryMarker.visible = true;
  queryLabelEl.textContent = text.length > 40 ? `${text.slice(0, 40)}…` : text;
  const nb = r.neighbors.map((n) => n.id);
  setHighlight(nb, nb.map((j) => [-1, j, '#ffffff']));
  renderDetail();
}

// ------------------------------------------------------------ paneller
function rowHtml(n, opts = {}) {
  const it = S.space.items[n.id];
  const vis = it.m === 'i' ? `<img src="${thumbUrl(n.id)}" loading="lazy">` : '<div class="tx">T</div>';
  const body = it.m === 'i' ? esc(it.label ?? `Görüntü #${n.id}`) : esc(it.text);
  const w = Math.max(0, Math.min(1, opts.max ? n.score / opts.max : n.score)) * 100;
  const rank = n.rank ? `<div class="rank">sıra #${n.rank}</div>` : '';
  const same = opts.label && it.label === opts.label ? ' same' : '';
  return `<div class="li${same}" data-id="${n.id}">${vis}<div class="body">${body}</div>
    <div class="score">${n.score.toFixed(3)}<div class="bar"><div style="width:${w}%"></div></div>${rank}</div></div>`;
}

function scopeSeg() {
  return `<div class="seg" id="nb-scope">${[['all', 'Tümü'], ['image', 'Görüntü'], ['text', 'Metin']]
    .map(([v, l]) => `<button data-v="${v}" class="${S.neighborScope === v ? 'on' : ''}">${l}</button>`).join('')}</div>`;
}

function renderDetail() {
  const el = $('#detail');
  if (S.query) {
    const q = S.query;
    const max = Math.max(...q.neighbors.map((n) => n.score));
    el.innerHTML = `<h2>Metin sorgusu</h2>
      <div class="sel-head"><div><span class="badge q">Sorgu</span>
      <div class="caption">${esc(q.text)}</div></div></div>
      ${S.info?.approx_query ? `<p class="note">Bu projeksiyonda (${METHOD_LABEL[S.method]}) sorgunun 3B konumu yaklaşıktır: yüksek boyutlu en yakın komşuların ağırlıklı ortalaması. Komşu listesi ise kesindir.</p>` : ''}
      <h3>En yakın ${q.neighbors.length} (cosine, ${S.space.dim}B uzayda)</h3>
      <div class="list">${q.neighbors.map((n) => rowHtml(n, { max })).join('')}</div>`;
  } else if (S.selected >= 0 && S.detail) {
    const i = S.selected;
    const it = S.space.items[i];
    const head = it.m === 'i'
      ? `<img src="${thumbUrl(i)}"><div><span class="badge img">Görüntü</span>${it.label ? `<span class="badge">${esc(it.label)}</span>` : ''}<span class="badge">#${i}</span></div>`
      : `<div><span class="badge txt">Metin</span>${it.label ? `<span class="badge">${esc(it.label)}</span>` : ''}<span class="badge">#${i}</span><div class="caption">${esc(it.text)}</div></div>`;
    const nbs = S.detail.neighbors;
    const max = Math.max(...nbs.map((n) => n.score));
    let pairs = '';
    if (S.detail.pairs.length) {
      const other = it.m === 'i' ? S.space.n_txt : S.space.n_img;
      pairs = `<h3>${it.m === 'i' ? "Gerçek caption'ları" : 'Ait olduğu görüntü'} <span class="muted">sıra / ${other}</span></h3>
        <div class="list">${S.detail.pairs.map((n) => rowHtml(n, { max })).join('')}</div>`;
    }
    let sameNote = '';
    if (it.label) {
      const same = nbs.filter((n) => S.space.items[n.id].label === it.label).length;
      sameNote = `<p class="note">Top-${nbs.length} içinde aynı sınıf: <b>${same}/${nbs.length}</b></p>`;
    }
    el.innerHTML = `<h2>Seçim</h2><div class="sel-head">${head}</div>${pairs}
      <h3>En yakın komşular ${scopeSeg()}</h3>${sameNote}
      <div class="list">${nbs.map((n) => rowHtml(n, { max, label: it.label })).join('')}</div>`;
  } else {
    el.innerHTML = '<h2>Seçim</h2><p class="muted">Bir noktaya tıklayın veya metin sorgusu yapın. Komşular modelin gerçek (yüksek boyutlu) uzayındaki cosine benzerliğine göre listelenir; 3B projeksiyona göre değil.</p>';
  }
  renderLegend();
}

$('#detail').addEventListener('click', (ev) => {
  const sc = ev.target.closest('#nb-scope button');
  if (sc) {
    S.neighborScope = sc.dataset.v;
    if (S.selected >= 0) select(S.selected);
    return;
  }
  const li = ev.target.closest('.li');
  if (li) select(+li.dataset.id);
});

function renderGap() {
  const g = S.gap, el = $('#gap-body');
  if (!g) { el.innerHTML = '—'; return; }
  const cls = S.space.kind === 'classes';
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  el.className = '';
  el.innerHTML = `<table class="metrics">
    <tr><td>Gap normu ‖μ<sub>img</sub> − μ<sub>txt</sub>‖</td><td>${g.gap_norm.toFixed(3)}</td></tr>
    <tr><td>Merkezler arası cosine</td><td>${g.centroid_cos.toFixed(3)}</td></tr>
    <tr><td>Eşleşen çift ort. cosine</td><td>${g.matched_cos.toFixed(3)}</td></tr>
    <tr><td>Rastgele çift ort. cosine</td><td>${g.random_cos.toFixed(3)}</td></tr>
    <tr><td>Görüntü içi ort. cosine</td><td>${g.intra_img_cos.toFixed(3)}</td></tr>
    <tr><td>Metin içi ort. cosine</td><td>${g.intra_txt_cos.toFixed(3)}</td></tr>
    <tr><td>${cls ? 'Zero-shot doğruluk (görüntü→en yakın prompt)' : 'Görüntü→metin top-1'}</td><td>${pct(g.i2t_top1)}</td></tr>
    <tr><td>${cls ? 'Prompt→en yakın görüntü doğru sınıf' : 'Metin→görüntü top-1'}</td><td>${pct(g.t2i_top1)}</td></tr>
  </table>
  <p class="note">Metrikler mevcut ön işleme (normalize/centering) sonrasındaki uzayda hesaplanır.
  Eşleşen çiftlerin cosine'ı rastgele çiftlerden yalnızca biraz yüksekken modalite içi benzerliklerin
  yüksek olması gap'in işaretidir. Centering'i “Modalite başına” yapıp tekrar uygulayın: gap kapanır,
  retrieval ise neredeyse değişmez.</p>`;

  const a = new THREE.Vector3().fromArray(g.centroid3d_img);
  const b = new THREE.Vector3().fromArray(g.centroid3d_txt);
  cImg.position.copy(a);
  cTxt.position.copy(b);
  const d = b.clone().sub(a);
  const len = d.length();
  gapArrow.position.copy(a);
  if (len > 1e-4) {
    gapArrow.setDirection(d.normalize());
    gapArrow.setLength(len, Math.min(0.06, len * 0.3), Math.min(0.035, len * 0.2));
  }
  gapArrow.visible = len > 0.02;
  gapGroup.visible = S.showGap;
}

function renderLegend() {
  const el = $('#legend');
  if (!S.space) { el.innerHTML = ''; return; }
  const parts = [];
  if (S.colorBy === 'label' && S.space.classes.length) {
    S.space.classes.forEach((c, k) => parts.push(`<span><i style="background:${PALETTE[k % PALETTE.length]}"></i>${esc(c)}</span>`));
    parts.push('<span class="muted">● görüntü ◆ metin</span>');
  } else {
    parts.push(`<span><i style="background:${MOD_COLORS.i}"></i>Görüntü (${S.space.n_img})</span>`);
    parts.push(`<span><i class="diamond" style="background:${MOD_COLORS.t}"></i>Metin (${S.space.n_txt})</span>`);
  }
  if (S.query) parts.push('<span><i class="diamond" style="background:#fff"></i>Sorgu</span>');
  el.innerHTML = parts.join('');
}

function renderInfo() {
  const sp = S.space, inf = S.info;
  if (!sp) return;
  const bits = [`${sp.n_img + sp.n_txt} nokta`, `${sp.dim} boyut`, S.cfg.models[sp.model].title];
  if (inf) {
    // Statik sürümde süre export anındaki hesaplamaya aittir, ziyaretçi için anlamsız
    bits.push(STATIC ? METHOD_LABEL[S.method] : `${METHOD_LABEL[S.method]} ${inf.seconds}s`);
    if (inf.explained_variance) {
      const ev = inf.explained_variance;
      bits.push(`açıklanan varyans: ${ev.map((v) => `${(v * 100).toFixed(1)}%`).join(' / ')} (Σ ${(ev.reduce((a, b) => a + b) * 100).toFixed(1)}%)`);
    }
  }
  $('#infobar').textContent = bits.join(' · ');
}

// ------------------------------------------------------------ yükleme & projeksiyon
function busy(msg) {
  S.busy = !!msg;
  $('#busy').hidden = !msg;
  $('#busy-msg').textContent = msg || '';
  for (const b of document.querySelectorAll('#load, #apply')) b.disabled = !!msg;
}

function showError(e) {
  busy(null);
  $('#detail').innerHTML = `<h2>Hata</h2><p class="err">${esc(e.message || e)}</p>`;
}

function projBody() {
  const params = S.method === 'umap'
    ? { n_neighbors: +$('#p-nn').value, min_dist: +$('#p-md').value }
    : S.method === 'tsne' ? { perplexity: +$('#p-perp').value } : {};
  return { method: S.method, normalize: $('#normalize').checked, center: $('#center').value, params };
}

async function project() {
  if (!S.space) return;
  busy(STATIC || S.method === 'pca' ? `${METHOD_LABEL[S.method]} yükleniyor…`
    : S.method === 'rtdae' ? 'RTD-AE eğitiliyor… (ilk seferde ~1–2 dk)'
    : `${METHOD_LABEL[S.method]} hesaplanıyor… (ilk seferde biraz sürebilir)`);
  const body = projBody();
  try {
    const [r, g] = await Promise.all([backend.project(body), backend.gap(body)]);
    S.info = r.info;
    animateTo(Float32Array.from(r.coords));
    S.gap = g;
    renderGap();
    renderInfo();
    busy(null);
    if (S.query) runQuery(S.query.text);
  } catch (e) { showError(e); }
}

async function waitReady() {
  for (;;) {
    const st = await backend.status();
    if (st.status === 'ready') return;
    if (st.status === 'error') throw new Error(st.error);
    busy(st.message || 'Yükleniyor…');
    await new Promise((r) => setTimeout(r, 500));
  }
}

async function loadSpace() {
  const body = { dataset: $('#dataset').value, model: $('#model').value, n: +$('#n').value };
  Object.assign(prefs, body);
  savePrefs();
  busy('Başlatılıyor…');
  try {
    await backend.load(body);
    await new Promise((r) => setTimeout(r, 300));
    await waitReady();
    await onSpaceReady();
  } catch (e) { showError(e); }
}

async function onSpaceReady() {
  S.space = await backend.space();
  $('#dataset').value = S.space.dataset;
  $('#model').value = S.space.model;
  syncN(S.space.n);
  const hasClasses = S.space.classes.length > 0;
  $('#color-by').querySelector('[value=label]').disabled = !hasClasses;
  if (!hasClasses && S.colorBy === 'label') S.colorBy = 'modality';
  if (hasClasses && !prefs.colorBy) S.colorBy = 'label';
  $('#color-by').value = S.colorBy;
  // Eski indeksler yeni bulutta geçersiz: bulutu kurmadan önce vurguları sıfırla
  hl = [];
  linkList = [];
  buildCloud();
  clearSelection();
  await project();
}

function syncN(value) {
  const d = S.cfg.datasets[$('#dataset').value];
  const n = $('#n');
  n.max = d.max_n;
  n.value = value ?? d.default_n;
  $('#n-out').value = n.value;
  $('#n-label').textContent = d.kind === 'captions' ? 'Görüntü sayısı (×5 caption)' : 'Sınıf başına görüntü';
}

// ------------------------------------------------------------ kontroller
function seg(id, value, onChange) {
  const el = $(id);
  const set = (v) => el.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
  set(value);
  el.addEventListener('click', (ev) => {
    const b = ev.target.closest('button');
    if (!b) return;
    set(b.dataset.v);
    onChange(b.dataset.v);
  });
}

function bindRange(id, onInput) {
  const el = $(id), out = el.parentElement.querySelector('output');
  const sync = () => { if (out) out.value = el.value; };
  sync();
  el.addEventListener('input', () => { sync(); onInput?.(+el.value); });
}

function showParams() {
  $('#params-umap').classList.toggle('on', S.method === 'umap');
  $('#params-tsne').classList.toggle('on', S.method === 'tsne');
  document.querySelectorAll('#method-notes [data-m]').forEach((el) => { el.hidden = el.dataset.m !== S.method; });
}

function setupControls() {
  const { datasets, models, device } = S.cfg;
  $('#device').textContent = device.toUpperCase();
  $('#dataset').innerHTML = Object.entries(datasets).map(([k, v]) => `<option value="${k}">${esc(v.title)}</option>`).join('');
  $('#model').innerHTML = Object.entries(models).map(([k, v]) => `<option value="${k}">${esc(v.title)}</option>`).join('');
  if (prefs.dataset in datasets) $('#dataset').value = prefs.dataset;
  if (prefs.model in models) $('#model').value = prefs.model;
  syncN(prefs.dataset === $('#dataset').value ? prefs.n : undefined);
  $('#dataset').addEventListener('change', () => syncN());
  bindRange('#n');
  $('#load').addEventListener('click', loadSpace);
  if (STATIC) {
    // Statik sürümde projeksiyonlar varsayılan parametrelerle önceden hesaplanmıştır
    document.body.classList.add('static');
    $('#model').disabled = Object.keys(models).length < 2;
    $('#normalize').checked = true;
  }

  seg('#method', S.method, (v) => { S.method = v; prefs.method = v; savePrefs(); showParams(); project(); });
  showParams();
  bindRange('#p-nn');
  bindRange('#p-md');
  bindRange('#p-perp');
  if (prefs.center) $('#center').value = prefs.center;
  if (prefs.normalize === false) $('#normalize').checked = false;
  $('#apply').addEventListener('click', () => {
    prefs.center = $('#center').value;
    prefs.normalize = $('#normalize').checked;
    savePrefs();
    project();
  });

  seg('#query-scope', S.queryScope, (v) => { S.queryScope = v; if (S.query) runQuery(S.query.text); });
  $('#query').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') runQuery(ev.target.value); });

  seg('#point-mode', S.pointMode, (v) => {
    S.pointMode = v; prefs.pointMode = v; savePrefs();
    cloudMat.uniforms.uSprite.value = overlayMat.uniforms.uSprite.value = v === 'sprite' ? 1 : 0;
    applyStates();
  });
  cloudMat.uniforms.uSprite.value = overlayMat.uniforms.uSprite.value = S.pointMode === 'sprite' ? 1 : 0;
  $('#color-by').addEventListener('change', (ev) => { S.colorBy = ev.target.value; prefs.colorBy = S.colorBy; savePrefs(); applyColors(); });
  $('#size').value = S.size;
  bindRange('#size', (v) => {
    S.size = v; prefs.size = v; savePrefs();
    cloudMat.uniforms.uSize.value = overlayMat.uniforms.uSize.value = v;
  });
  $('#show-img').addEventListener('change', (ev) => { S.show.i = ev.target.checked; applyStates(); });
  $('#show-txt').addEventListener('change', (ev) => { S.show.t = ev.target.checked; applyStates(); });
  $('#show-gap').addEventListener('change', (ev) => { S.showGap = ev.target.checked; gapGroup.visible = S.showGap && !!S.gap; });
  $('#rotate').addEventListener('change', (ev) => { controls.autoRotate = ev.target.checked; });
  $('#reset-cam').addEventListener('click', () => flyTo(new THREE.Vector3(), HOME.length()));

  addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, select')) {
      if (ev.key === 'Escape') ev.target.blur();
      return;
    }
    if (ev.key === '/') { ev.preventDefault(); $('#query').focus(); }
    if (ev.key === 'Escape') clearSelection();
  });
}

// ------------------------------------------------------------ başlat
(async function init() {
  try {
    S.cfg = await backend.config();
    setupControls();
    renderDetail();
    const st = await backend.status();
    if (st.status === 'ready') await onSpaceReady();
    else if (st.status === 'loading') { await waitReady(); await onSpaceReady(); }
    else await loadSpace();
  } catch (e) { showError(e); }
})();
