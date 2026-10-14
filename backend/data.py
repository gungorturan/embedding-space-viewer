"""Hazır dataset'leri yükleyip ortak bir formata (görüntüler + metinler) dönüştürür."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from datasets import load_dataset
from PIL import Image

# CLIP makalesindeki prompt ensemble'ından seçilmiş şablonlar
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a drawing of a {}.",
    "a black and white photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of the large {}.",
]

IMAGENETTE_NAMES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]


@dataclass
class Dataset:
    name: str
    images: list[Image.Image]
    image_labels: list[str | None]
    texts: list[str]
    text_labels: list[str | None]
    # Her metnin eşleştiği görüntü indeksi (caption dataset'lerinde). Sınıf
    # dataset'lerinde None; eşleşme etiket üzerinden yapılır.
    text_pairs: list[int | None]
    classes: list[str] = field(default_factory=list)


DATASETS = {
    "flickr1k": {
        "title": "Flickr30k-1K (görüntü + 5 caption)",
        "kind": "captions",
        "default_n": 500,
        "max_n": 1000,
    },
    "imagenette": {
        "title": "Imagenette (10 sınıf, 320px)",
        "kind": "classes",
        "default_n": 100,
        "max_n": 300,
    },
    "cifar10": {
        "title": "CIFAR-10 (10 sınıf, 32px)",
        "kind": "classes",
        "default_n": 100,
        "max_n": 500,
    },
}


def _stratified(labels: list[int], per_class: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    picked = []
    for y in sorted(by_class):
        idx = by_class[y]
        rng.shuffle(idx)
        picked.extend(idx[:per_class])
    return picked


def _class_dataset(name, hf_name, split, image_col, class_names, per_class, seed):
    ds = load_dataset(hf_name, split=split)
    picked = _stratified(ds["label"], per_class, seed)
    sub = ds.select(picked)
    images = [im.convert("RGB") for im in sub[image_col]]
    image_labels = [class_names[y] for y in sub["label"]]
    texts, text_labels = [], []
    for c in class_names:
        for t in PROMPT_TEMPLATES:
            texts.append(t.format(c))
            text_labels.append(c)
    return Dataset(name, images, image_labels, texts, text_labels,
                   [None] * len(texts), list(class_names))


def load(name: str, n: int, seed: int = 0) -> Dataset:
    """n: caption dataset'inde görüntü sayısı, sınıf dataset'inde sınıf başına görüntü sayısı."""
    if name == "flickr1k":
        ds = load_dataset(
            "parquet",
            data_files="hf://datasets/nlphuji/flickr_1k_test_image_text_retrieval"
            "@refs%2Fconvert%2Fparquet/TEST/test/0000.parquet",
            split="train",
        )
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        sub = ds.select(sorted(idx[:n]))
        images = [im.convert("RGB") for im in sub["image"]]
        texts, pairs = [], []
        for i, caps in enumerate(sub["caption"]):
            for c in caps:
                texts.append(c.strip())
                pairs.append(i)
        return Dataset(name, images, [None] * len(images), texts,
                       [None] * len(texts), pairs)
    if name == "imagenette":
        return _class_dataset(name, "johnowhitaker/imagenette2-320", "train",
                              "image", IMAGENETTE_NAMES, n, seed)
    if name == "cifar10":
        ds_names = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]
        return _class_dataset(name, "uoft-cs/cifar10", "test", "img",
                              ds_names, n, seed)
    raise ValueError(f"Bilinmeyen dataset: {name}")
