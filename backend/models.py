"""CLIP / SigLIP encoder sarmalayıcıları. Yeni model eklemek için MODELS'e kayıt eklemek yeterli."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODELS = {
    "clip-b32": {
        "title": "CLIP ViT-B/32 (OpenAI)",
        "hf": "openai/clip-vit-base-patch32",
        "text_kwargs": {"padding": True, "truncation": True},
    },
    "siglip-b16": {
        "title": "SigLIP ViT-B/16-224 (Google)",
        "hf": "google/siglip-base-patch16-224",
        # SigLIP metinleri eğitimdeki gibi sabit uzunluğa pad'lenmezse embedding'ler bozulur
        "text_kwargs": {"padding": "max_length", "max_length": 64, "truncation": True},
    },
}

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


def _features(out) -> torch.Tensor:
    # transformers 5.x: get_*_features BaseModelOutputWithPooling döndürür,
    # projekte edilmiş vektör pooler_output'tadır. Eski sürümler doğrudan tensor döndürür.
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class Encoder:
    def __init__(self, key: str):
        cfg = MODELS[key]
        self.key = key
        self.text_kwargs = cfg["text_kwargs"]
        self.processor = AutoProcessor.from_pretrained(cfg["hf"])
        self.model = AutoModel.from_pretrained(cfg["hf"]).to(DEVICE).eval()

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=images[i:i + batch_size], return_tensors="pt").to(DEVICE)
            out.append(_features(self.model.get_image_features(**inputs)).float().cpu())
        return torch.cat(out).numpy()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            inputs = self.processor(text=texts[i:i + batch_size], return_tensors="pt",
                                    **self.text_kwargs).to(DEVICE)
            out.append(_features(self.model.get_text_features(**inputs)).float().cpu())
        return torch.cat(out).numpy()


_cache: dict[str, Encoder] = {}


def get_encoder(key: str) -> Encoder:
    # 16 GB RAM'de iki modeli aynı anda tutmamak için tek model önbellekte kalır
    if key not in _cache:
        _cache.clear()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        _cache[key] = Encoder(key)
    return _cache[key]
