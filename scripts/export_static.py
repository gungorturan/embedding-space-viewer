"""Backend gerektirmeyen statik site üretir (GitHub Pages için): docs/

Embedding'ler, tüm projeksiyonlar (3 yöntem × 3 centering) ve gap metrikleri önceden
hesaplanır. Metin sorgusu tarayıcıda transformers.js ile çalışır.

    uv run python -m scripts.export_static
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from backend import data, models
from backend.space import Space

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"

# (dataset, model, n) — n: caption dataset'inde görüntü, sınıf dataset'inde sınıf başına görüntü
COMBOS = [
    ("flickr1k", "clip-b32", 500),
    ("imagenette", "clip-b32", 100),
]
# Tarayıcıda çalışacak ONNX karşılıkları (Hugging Face Hub)
BROWSER_MODELS = {"clip-b32": "Xenova/clip-vit-base-patch32"}
METHODS = ["pca", "umap", "tsne"]
CENTERS = ["none", "global", "modality"]


def f32(a) -> list[float]:
    return np.asarray(a, np.float32).round(6).tolist()


def export_combo(dataset: str, model: str, n: int) -> dict:
    key = f"{dataset}_{model}"
    d = OUT / "data" / key
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    sp = Space(dataset, model, n, lambda m: print(f"  [{key}] {m}"))

    sp.unit.astype("<f4").tofile(d / "emb.bin")
    (d / "atlas.jpg").write_bytes(sp.atlas)
    for i, b in enumerate(sp.thumbs):
        (d / "thumbs" / f"{i}.jpg").write_bytes(b)

    centers = {}
    for c in CENTERS:
        mu_img, mu_txt = sp._prep_stats(True, c)
        proj = {}
        for m in METHODS:
            print(f"  [{key}] {m} / {c}")
            res = sp.project(m, True, c, {})
            res["_Y"].astype("<f4").tofile(d / f"proj_{m}_{c}.bin")
            p = {"file": f"proj_{m}_{c}.bin", "info": {**res["info"], "approx_query": m != "pca"}}
            if m == "pca":
                pca = res["_pca"]
                p["pca"] = {"mean": f32(pca.mean_), "components": [f32(r) for r in pca.components_],
                            "shift": f32(res["_shift"]), "scale": float(res["_scale"])}
            proj[m] = p
        gap = sp.gap(True, c, sp.project("pca", True, c, {}))
        gap = {k: v for k, v in gap.items() if not k.startswith("centroid3d")}
        centers[c] = {"mu_img": f32(mu_img), "mu_txt": f32(mu_txt), "gap": gap, "proj": proj}

    space = {
        "dataset": dataset, "model": model, "n": n,
        "n_img": sp.n_img, "n_txt": sp.n_txt, "dim": int(sp.raw.shape[1]),
        "classes": sp.ds.classes, "kind": data.DATASETS[dataset]["kind"],
        "atlas": sp.atlas_grid, "items": sp.items(),
        "pairs": sp.pairs if not sp.ds.classes else None,
        "centers": centers,
    }
    (d / "space.json").write_text(json.dumps(space, ensure_ascii=False, separators=(",", ":")))
    return {"key": key, "dataset": dataset, "model": model, "n": n,
            "title": data.DATASETS[dataset]["title"], "kind": data.DATASETS[dataset]["kind"]}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    combos = [export_combo(*c) for c in COMBOS]
    manifest = {
        "combos": combos,
        "models": {k: {"title": models.MODELS[k]["title"], "browser": BROWSER_MODELS[k]}
                   for k in {c["model"] for c in combos}},
    }
    (OUT / "data" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    # Arayüz dosyaları: aynı kod, statik modda çalışır
    front = ROOT / "frontend"
    shutil.copytree(front, OUT / "static", ignore=shutil.ignore_patterns("index.html"))
    html = (front / "index.html").read_text()
    (OUT / "index.html").write_text(html.replace('<html lang="tr">', '<html lang="tr" data-mode="static">'))
    (OUT / ".nojekyll").write_text("")  # GitHub Pages'in Jekyll işlemesini kapat

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"docs/ hazır: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
