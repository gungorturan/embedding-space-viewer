"""FastAPI sunucusu. Tek kullanıcılı yerel araç: aktif embedding uzayı sunucuda tutulur."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data, models
from .space import Space

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Embedding Space Viewer")

state: dict = {"space": None, "status": "idle", "message": "", "error": None, "version": 0}
lock = threading.Lock()


class LoadReq(BaseModel):
    dataset: str
    model: str
    n: int


class ProjReq(BaseModel):
    method: str = "umap"
    normalize: bool = True
    center: str = "none"  # none | global | modality
    params: dict = {}


class QueryReq(ProjReq):
    text: str
    k: int = 12
    scope: str = "all"


def _space() -> Space:
    sp = state["space"]
    if sp is None:
        raise HTTPException(409, "Önce bir dataset/model yükleyin")
    return sp


def _proj(sp: Space, r: ProjReq) -> dict:
    if r.method not in ("pca", "umap", "tsne", "rtdae") or r.center not in ("none", "global", "modality"):
        raise HTTPException(400, "Geçersiz projeksiyon parametresi")
    return sp.project(r.method, r.normalize, r.center, r.params)


@app.get("/api/config")
def config():
    return {
        "datasets": data.DATASETS,
        "models": {k: {"title": v["title"]} for k, v in models.MODELS.items()},
        "device": models.DEVICE,
    }


@app.post("/api/load")
def load(r: LoadReq):
    if r.dataset not in data.DATASETS or r.model not in models.MODELS:
        raise HTTPException(400, "Bilinmeyen dataset veya model")
    n = max(1, min(r.n, data.DATASETS[r.dataset]["max_n"]))
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "Zaten bir yükleme sürüyor")

    def progress(msg):
        state["message"] = msg

    def run():
        try:
            state.update(status="loading", error=None)
            sp = Space(r.dataset, r.model, n, progress)
            state.update(space=sp, status="ready", message="Hazır", version=state["version"] + 1)
        except Exception as e:  # noqa: BLE001 - hata arayüzde gösterilir
            traceback.print_exc()
            state.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def status():
    return {k: state[k] for k in ("status", "message", "error", "version")}


@app.get("/api/space")
def space_info():
    sp = _space()
    return {
        "dataset": sp.dataset, "model": sp.model, "n": sp.n,
        "n_img": sp.n_img, "n_txt": sp.n_txt, "dim": int(sp.raw.shape[1]),
        "classes": sp.ds.classes, "kind": data.DATASETS[sp.dataset]["kind"],
        "atlas": sp.atlas_grid, "version": state["version"],
        "items": sp.items(),
        # Sınıf dataset'lerinde eşler etiketten türetilir, büyük listeyi göndermeye gerek yok
        "pairs": sp.pairs if not sp.ds.classes else None,
    }


@app.post("/api/project")
def project(r: ProjReq):
    sp = _space()
    res = _proj(sp, r)
    return {"coords": res["coords"], "info": res["info"]}


@app.post("/api/gap")
def gap(r: ProjReq):
    sp = _space()
    return sp.gap(r.normalize, r.center, _proj(sp, r))


@app.post("/api/query")
def query(r: QueryReq):
    sp = _space()
    if not r.text.strip():
        raise HTTPException(400, "Boş sorgu")
    return sp.query(r.text.strip(), r.k, r.scope, _proj(sp, r))


@app.get("/api/item/{idx}")
def item(idx: int, k: int = 12, scope: str = "all"):
    sp = _space()
    if not 0 <= idx < len(sp.raw):
        raise HTTPException(404)
    return sp.item(idx, k, scope)


@app.get("/api/thumb/{idx}")
def thumb(idx: int):
    sp = _space()
    if not 0 <= idx < sp.n_img:
        raise HTTPException(404)
    return Response(sp.thumbs[idx], media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/atlas")
def atlas():
    return Response(_space().atlas, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/wiki.html")
def wiki():
    return FileResponse(FRONTEND / "wiki.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
