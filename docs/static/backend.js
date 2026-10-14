// Veri katmanı: aynı arayüz iki kaynakla çalışır.
//  - ServerBackend: yerel FastAPI (canlı encode + projeksiyon)
//  - StaticBackend: docs/ altındaki önceden hesaplanmış dosyalar + tarayıcıda CLIP text encoder

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

async function request(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* yoksay */ }
    throw new Error(msg);
  }
  return r;
}
const json = async (path, body) => (await request(path, body)).json();

export class ServerBackend {
  isStatic = false;
  config() { return json('api/config'); }
  status() { return json('api/status'); }
  load(body) { return json('api/load', body); }
  async space() {
    this.sp = await json('api/space');
    return this.sp;
  }
  project(body) { return json('api/project', body); }
  gap(body) { return json('api/gap', body); }
  query(body) { return json('api/query', body); }
  item(i, k, scope) { return json(`api/item/${i}?k=${k}&scope=${scope}`); }
  thumbUrl(i) { return `api/thumb/${i}?v=${this.sp.version}`; }
  atlasUrl() { return `api/atlas?v=${this.sp.version}`; }
}

// ---------------------------------------------------------------- statik
const l2n = (v) => {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  return v.map((x) => x / s);
};

export class StaticBackend {
  isStatic = true;
  sp = null;
  #emb = null;           // Float32Array (N × dim), L2 normalize
  #projCache = new Map();
  #prepCache = new Map();
  #encoder = null;
  #status = { status: 'idle', message: '', error: null, version: 0 };

  constructor(onProgress) { this.onProgress = onProgress; }

  async config() {
    this.manifest = await json('data/manifest.json');
    const datasets = {};
    for (const c of this.manifest.combos) {
      datasets[c.key] = { title: `${c.title} · ${this.manifest.models[c.model].title}`, kind: c.kind, default_n: c.n, max_n: c.n };
    }
    const models = Object.fromEntries(Object.entries(this.manifest.models).map(([k, v]) => [k, { title: v.title }]));
    return { datasets, models, device: 'Statik' };
  }

  status() { return this.#status; }

  async load({ dataset }) {
    const combo = this.manifest.combos.find((c) => c.key === dataset) || this.manifest.combos[0];
    this.base = `data/${combo.key}/`;
    this.#status = { ...this.#status, status: 'loading', message: 'Veri indiriliyor…' };
    const [sp, buf] = await Promise.all([
      json(`${this.base}space.json`),
      request(`${this.base}emb.bin`).then((r) => r.arrayBuffer()),
    ]);
    this.#emb = new Float32Array(buf);
    this.#projCache.clear();
    this.#prepCache.clear();
    this.sp = { ...sp, dataset: combo.key, version: this.#status.version + 1 };
    this.#status = { status: 'ready', message: 'Hazır', error: null, version: this.sp.version };
    return { ok: true };
  }

  async space() { return this.sp; }

  thumbUrl(i) { return `${this.base}thumbs/${i}.jpg`; }
  atlasUrl() { return `${this.base}atlas.jpg`; }

  #center(body) { return this.sp.centers[body.center] || this.sp.centers.none; }

  async project(body) {
    const p = this.#center(body).proj[body.method];
    if (!this.#projCache.has(p.file)) {
      const buf = await (await request(this.base + p.file)).arrayBuffer();
      this.#projCache.set(p.file, new Float32Array(buf));
    }
    return { coords: this.#projCache.get(p.file), info: p.info };
  }

  async gap(body) {
    const { coords } = await this.project(body);
    const ci = [0, 0, 0], ct = [0, 0, 0];
    const { n_img: ni, n_txt: nt } = this.sp;
    for (let i = 0; i < ni + nt; i++) {
      const c = i < ni ? ci : ct;
      for (let a = 0; a < 3; a++) c[a] += coords[i * 3 + a];
    }
    return {
      ...this.#center(body).gap,
      centroid3d_img: ci.map((v) => v / ni),
      centroid3d_txt: ct.map((v) => v / nt),
    };
  }

  // Ön işlenmiş matris: x - μ(modalite), centering varsa tekrar normalize (backend/space.py ile aynı)
  #prep(center) {
    if (this.#prepCache.has(center)) return this.#prepCache.get(center);
    const c = this.sp.centers[center];
    const { dim, n_img: ni } = this.sp;
    const N = this.#emb.length / dim;
    const X = new Float32Array(this.#emb.length);
    for (let i = 0; i < N; i++) {
      const mu = i < ni ? c.mu_img : c.mu_txt;
      let s = 0;
      for (let d = 0; d < dim; d++) { const v = this.#emb[i * dim + d] - mu[d]; X[i * dim + d] = v; s += v * v; }
      s = Math.sqrt(s) || 1;
      for (let d = 0; d < dim; d++) X[i * dim + d] /= s;  // kNN için her durumda birim vektör
    }
    this.#prepCache.set(center, X);
    return X;
  }

  #sims(M, v) {
    const dim = v.length, N = M.length / dim, out = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let d = 0; d < dim; d++) s += M[i * dim + d] * v[d];
      out[i] = s;
    }
    return out;
  }

  #topk(sims, k, scope, exclude = -1) {
    const ni = this.sp.n_img, idx = [];
    for (let i = 0; i < sims.length; i++) {
      if (i === exclude || (scope === 'image' && i >= ni) || (scope === 'text' && i < ni)) continue;
      idx.push(i);
    }
    idx.sort((a, b) => sims[b] - sims[a]);
    return idx.slice(0, k).map((id) => ({ id, score: Math.round(sims[id] * 1e4) / 1e4 }));
  }

  async item(i, k, scope) {
    const dim = this.sp.dim;
    const sims = this.#sims(this.#emb, this.#emb.subarray(i * dim, (i + 1) * dim));
    const res = { neighbors: this.#topk(sims, k, scope, i), pairs: [] };
    if (this.sp.pairs) {
      const ni = this.sp.n_img, isImg = i < ni;
      const other = [];
      for (let j = 0; j < sims.length; j++) if ((j < ni) !== isImg) other.push(j);
      other.sort((a, b) => sims[b] - sims[a]);
      const rank = new Map(other.map((j, r) => [j, r + 1]));
      res.pairs = this.sp.pairs[i].map((j) => ({ id: j, score: Math.round(sims[j] * 1e4) / 1e4, rank: rank.get(j) }));
    }
    return res;
  }

  async #encode(text) {
    if (!this.#encoder) {
      const model = this.manifest.models[this.sp.model].browser;
      this.onProgress?.('Metin encoder\'ı indiriliyor (yalnızca ilk sorguda, ~127 MB)…');
      const tf = await import(TRANSFORMERS_URL);
      const files = new Map();
      const progress = (p) => {
        if (p.status !== 'progress' || !p.total) return;
        files.set(p.file, p);
        let done = 0, total = 0;
        for (const f of files.values()) { done += f.loaded; total += f.total; }
        this.onProgress?.(`Metin encoder'ı indiriliyor… ${Math.round(done / 1e6)} / ${Math.round(total / 1e6)} MB`);
      };
      const [tok, enc] = await Promise.all([
        tf.AutoTokenizer.from_pretrained(model),
        // fp16: PyTorch embedding'leriyle cos ≈ 1.000. 8-bit varyantlar (q8/int8/uint8) ortalama
        // 0.79'a, q4 0.98'e düşüyor ve komşu sıralamasını bozuyor (ölçüm: 5 Flickr caption'ı)
        tf.CLIPTextModelWithProjection.from_pretrained(model, { dtype: 'fp16', progress_callback: progress }),
      ]);
      this.#encoder = { tok, enc };
      this.onProgress?.(null);
    }
    const { tok, enc } = this.#encoder;
    const inputs = tok([text], { padding: true, truncation: true });
    const { text_embeds } = await enc(inputs);
    return l2n(Float32Array.from(text_embeds.data));
  }

  async query({ text, k, scope, method, center }) {
    const v = await this.#encode(text);
    const sims = this.#sims(this.#emb, v);
    const c = this.#center({ center });
    const p = c.proj[method];
    const { coords } = await this.project({ method, center });

    // Sorguyu aynı ön işlemden geçir (metin modalitesi)
    let q = v.map((x, d) => x - c.mu_txt[d]);
    if (center !== 'none') q = l2n(q);
    let pos;
    if (p.pca) {
      const { mean, components, shift, scale } = p.pca;
      pos = components.map((row, a) => {
        let s = 0;
        for (let d = 0; d < row.length; d++) s += (q[d] - mean[d]) * row[d];
        return (s - shift[a]) / scale;
      });
    } else {
      // UMAP/t-SNE: yüksek boyutlu en yakın 10 komşunun 3B konumlarının ağırlıklı ortalaması
      const s2 = this.#sims(this.#prep(center), l2n(q));
      const nn = this.#topk(s2, 10, 'all');
      const mx = s2[nn[0].id];
      pos = [0, 0, 0];
      let wsum = 0;
      for (const { id } of nn) {
        const w = Math.exp((s2[id] - mx) / 0.02);
        wsum += w;
        for (let a = 0; a < 3; a++) pos[a] += w * coords[id * 3 + a];
      }
      pos = pos.map((x) => x / wsum);
    }
    return { pos, neighbors: this.#topk(sims, k, scope) };
  }
}
