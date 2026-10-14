"""Embedding uzayı: hesaplama, ön işleme, 3B projeksiyon, komşuluk ve modality gap analizi."""

from __future__ import annotations

import io
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import data, models

CACHE = Path(__file__).resolve().parent.parent / "cache"
ATLAS_MAX = 4096


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


class Space:
    def __init__(self, dataset: str, model: str, n: int, progress=lambda msg: None):
        self.dataset, self.model, self.n = dataset, model, n
        progress("Dataset yükleniyor…")
        self.ds = data.load(dataset, n)
        n_img, n_txt = len(self.ds.images), len(self.ds.texts)

        emb_path = CACHE / "emb" / f"{dataset}_{model}_{n}.npz"
        if emb_path.exists():
            progress("Önbellekten embedding'ler okunuyor…")
            z = np.load(emb_path)
            img, txt = z["img"], z["txt"]
        else:
            progress("Model yükleniyor (ilk seferde indirilir)…")
            enc = models.get_encoder(model)
            progress(f"{n_img} görüntü encode ediliyor…")
            img = enc.encode_images(self.ds.images)
            progress(f"{n_txt} metin encode ediliyor…")
            txt = enc.encode_texts(self.ds.texts)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(emb_path, img=img, txt=txt)

        self.raw = np.concatenate([img, txt]).astype(np.float32)
        self.is_img = np.r_[np.ones(n_img, bool), np.zeros(n_txt, bool)]
        self.n_img, self.n_txt = n_img, n_txt
        self.labels = list(self.ds.image_labels) + list(self.ds.text_labels)
        # pairs[i]: i'nin "doğru eşleri" sayılan karşı modalite indeksleri
        self.pairs = self._build_pairs()
        self.unit = l2n(self.raw)  # modelin gerçek (cosine) uzayı

        progress("Thumbnail atlası hazırlanıyor…")
        self._build_thumbs()
        self._proj_cache: dict[tuple, dict] = {}
        # Numba'nın varsayılan threading katmanı eşzamanlı çağrıda süreci çökertiyor;
        # ayrıca aynı projeksiyonun paralel isteklerle iki kez hesaplanmasını önler
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ yapı
    def _build_pairs(self) -> list[list[int]]:
        n_img = self.n_img
        pairs: list[list[int]] = [[] for _ in range(n_img + self.n_txt)]
        if self.ds.classes:
            by_label: dict[str, list[int]] = {}
            for i, lab in enumerate(self.labels):
                by_label.setdefault(lab, []).append(i)
            for i, lab in enumerate(self.labels):
                pairs[i] = [j for j in by_label[lab] if self.is_img[j] != self.is_img[i]]
        else:
            for t, im in enumerate(self.ds.text_pairs):
                pairs[n_img + t].append(im)
                pairs[im].append(n_img + t)
        return pairs

    def _build_thumbs(self):
        n = self.n_img
        cols = math.ceil(math.sqrt(n))
        cell = max(16, min(64, ATLAS_MAX // cols))
        rows = math.ceil(n / cols)
        atlas = Image.new("RGB", (cols * cell, rows * cell), (40, 40, 40))
        self.thumbs: list[bytes] = []
        for i, im in enumerate(self.ds.images):
            atlas.paste(_square(im, cell), ((i % cols) * cell, (i // cols) * cell))
            t = im.copy()
            t.thumbnail((192, 192))
            buf = io.BytesIO()
            t.save(buf, "JPEG", quality=85)
            self.thumbs.append(buf.getvalue())
        buf = io.BytesIO()
        atlas.save(buf, "JPEG", quality=88)
        self.atlas = buf.getvalue()
        self.atlas_grid = {"cols": cols, "rows": rows, "cell": cell}

    def items(self) -> list[dict]:
        out = []
        for i in range(len(self.raw)):
            if self.is_img[i]:
                out.append({"m": "i", "label": self.labels[i]})
            else:
                out.append({"m": "t", "label": self.labels[i],
                            "text": self.ds.texts[i - self.n_img]})
        return out

    # ----------------------------------------------------------- ön işleme
    def _prep_stats(self, normalize: bool, center: str):
        X = l2n(self.raw) if normalize else self.raw.copy()
        mu_img = mu_txt = np.zeros(X.shape[1], np.float32)
        if center == "global":
            mu_img = mu_txt = X.mean(0)
        elif center == "modality":
            mu_img, mu_txt = X[self.is_img].mean(0), X[~self.is_img].mean(0)
        return mu_img, mu_txt

    def prep(self, normalize: bool, center: str):
        """Ön işlenmiş matris ve yeni metin vektörlerini aynı dönüşümden geçiren fonksiyon."""
        mu_img, mu_txt = self._prep_stats(normalize, center)

        def tf(v: np.ndarray, is_img: np.ndarray | bool) -> np.ndarray:
            v = l2n(v) if normalize else v
            mu = np.where(np.asarray(is_img)[..., None], mu_img, mu_txt)
            v = v - mu
            # Merkezledikten sonra tekrar küreye yansıtılır ki cosine geometrisi korunsun
            return l2n(v) if (normalize and center != "none") else v

        return tf(self.raw, self.is_img).astype(np.float32), lambda v: tf(v, False)

    # --------------------------------------------------------- projeksiyon
    def project(self, method: str, normalize: bool, center: str, params: dict) -> dict:
        key = (method, normalize, center, tuple(sorted(params.items())))
        with self._lock:
            if key not in self._proj_cache:
                self._proj_cache[key] = self._project(method, normalize, center, params)
            return self._proj_cache[key]

    def _project(self, method: str, normalize: bool, center: str, params: dict) -> dict:
        X, tf = self.prep(normalize, center)
        t0 = time.time()
        info: dict = {}
        fitted = None
        if method == "pca":
            fitted = PCA(3, random_state=0).fit(X)
            Y = fitted.transform(X)
            info["explained_variance"] = fitted.explained_variance_ratio_.tolist()
        elif method == "umap":
            import umap  # numba JIT ilk import'ta yavaş, sadece gerektiğinde yükle
            fitted = umap.UMAP(n_components=3, n_neighbors=int(params.get("n_neighbors", 15)),
                               min_dist=float(params.get("min_dist", 0.1)),
                               metric="cosine", random_state=42)
            Y = fitted.fit_transform(X)
        elif method == "tsne":
            perp = min(float(params.get("perplexity", 30)), (len(X) - 1) / 3)
            Y = TSNE(3, perplexity=perp, init="pca", metric="cosine",
                     random_state=42).fit_transform(X)
        else:
            raise ValueError(method)

        # 3B koordinatları birim küreye sığdır (aykırı noktalar ölçeği bozmasın diye %99)
        shift = Y.mean(0)
        scale = float(np.percentile(np.linalg.norm(Y - shift, axis=1), 99)) or 1.0
        Yn = ((Y - shift) / scale).astype(np.float32)

        def place(v: np.ndarray) -> np.ndarray:
            """Yeni bir vektörü (ön işlenmiş) 3B'ye yerleştir."""
            if method == "pca" or method == "umap":
                with self._lock:
                    return (fitted.transform(v[None])[0] - shift) / scale
            # t-SNE'nin transform'u yok: yüksek boyutlu en yakın komşuların
            # 3B konumlarının benzerlikle ağırlıklı ortalaması
            sims = l2n(X) @ l2n(v)
            nn = np.argsort(-sims)[:10]
            w = np.exp((sims[nn] - sims[nn].max()) / 0.02)
            return (w[:, None] * Yn[nn]).sum(0) / w.sum()

        return {
            "coords": Yn.ravel().round(5).tolist(),
            "info": {**info, "seconds": round(time.time() - t0, 2),
                     "approx_query": method == "tsne"},
            "_place": place, "_tf": tf, "_Y": Yn,
            # Statik export'un sorguları tarayıcıda yerleştirebilmesi için
            "_pca": fitted if method == "pca" else None, "_shift": shift, "_scale": scale,
        }

    # ----------------------------------------------------------- sorgular
    def item(self, idx: int, k: int, scope: str) -> dict:
        sims = self.unit @ self.unit[idx]
        res = {"neighbors": self._topk(sims, k, scope, exclude=idx), "pairs": []}
        # Caption dataset'lerinde gerçek eşlerin skoru ve karşı modalitedeki sırası
        if not self.ds.classes:
            other = np.where(self.is_img != self.is_img[idx])[0]
            order = other[np.argsort(-sims[other])]
            rank = {int(j): r + 1 for r, j in enumerate(order)}
            res["pairs"] = [{"id": j, "score": round(float(sims[j]), 4), "rank": rank[j]}
                            for j in self.pairs[idx]]
        return res

    def _topk(self, sims, k, scope, exclude=None):
        mask = np.ones(len(sims), bool)
        if scope == "image":
            mask &= self.is_img
        elif scope == "text":
            mask &= ~self.is_img
        if exclude is not None:
            mask[exclude] = False
        cand = np.where(mask)[0]
        top = cand[np.argsort(-sims[cand])[:k]]
        return [{"id": int(j), "score": round(float(sims[j]), 4)} for j in top]

    def query(self, text: str, k: int, scope: str, proj: dict) -> dict:
        enc = models.get_encoder(self.model)
        v = enc.encode_texts([text])[0]
        sims = self.unit @ l2n(v)
        pos = proj["_place"](proj["_tf"](v))
        return {"pos": np.asarray(pos, float).round(5).tolist(),
                "neighbors": self._topk(sims, k, scope)}

    # ------------------------------------------------------- modality gap
    def gap(self, normalize: bool, center: str, proj: dict) -> dict:
        X, _ = self.prep(normalize, center)
        U = l2n(X)
        I, T = U[self.is_img], U[~self.is_img]
        Xi, Xt = X[self.is_img], X[~self.is_img]
        ci, ct = Xi.mean(0), Xt.mean(0)

        # Eşleşen çiftler ve rastgele çiftler arası ortalama cosine
        rows, cols = [], []
        for t in range(self.n_txt):
            for im in self.pairs[self.n_img + t]:
                rows.append(t), cols.append(im)
        rows, cols = np.array(rows), np.array(cols)
        rng = np.random.default_rng(0)
        if len(rows) > 20000:
            sel = rng.choice(len(rows), 20000, replace=False)
            rows, cols = rows[sel], cols[sel]
        matched = float((T[rows] * I[cols]).sum(1).mean())
        ri = rng.integers(0, self.n_img, 5000)
        rt = rng.integers(0, self.n_txt, 5000)
        rand = float((I[ri] * T[rt]).sum(1).mean())

        def intra(A):
            s = rng.choice(len(A), min(len(A), 1500), replace=False)
            G = A[s] @ A[s].T
            return float((G.sum() - np.trace(G)) / (len(s) * (len(s) - 1)))

        # Görüntü → metin top-1: en yakın metin doğru eş mi?
        S = I @ T.T
        best_t = S.argmax(1) + self.n_img
        i2t = float(np.mean([bt in self.pairs[i] for i, bt in enumerate(best_t)]))
        best_i = S.argmax(0)
        t2i = float(np.mean([bi in self.pairs[self.n_img + t] for t, bi in enumerate(best_i)]))

        Y = proj["_Y"]
        return {
            "gap_norm": round(float(np.linalg.norm(ci - ct)), 4),
            "centroid_cos": round(float(l2n(ci) @ l2n(ct)), 4),
            "matched_cos": round(matched, 4),
            "random_cos": round(rand, 4),
            "intra_img_cos": round(intra(I), 4),
            "intra_txt_cos": round(intra(T), 4),
            "i2t_top1": round(i2t, 4),
            "t2i_top1": round(t2i, 4),
            "centroid3d_img": Y[self.is_img].mean(0).round(5).tolist(),
            "centroid3d_txt": Y[~self.is_img].mean(0).round(5).tolist(),
        }


def _square(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize(
        (size, size), Image.BILINEAR)
