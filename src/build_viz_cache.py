import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_PATH = DATA / "viz_cache.json"
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# small aggregates so the deployed app can show the charts without shipping
# the full image dataset in the image


def build():
    rng = random.Random(0)
    dist, colors, dims = [], [], []
    for split in ("train", "val", "test"):
        for c in CLASS_NAMES:
            p = DATA / split / c
            n = len(list(p.glob("*.jpg"))) if p.exists() else 0
            dist.append({"split": split, "class": c, "count": n})

    for c in CLASS_NAMES:
        p = DATA / "train" / c
        if not p.exists():
            continue
        for f in rng.sample(list(p.glob("*.jpg")), min(40, len(list(p.glob("*.jpg"))))):
            with Image.open(f) as im:
                w, h = im.width, im.height
                arr = np.asarray(im.convert("RGB")) / 255.0
            dims.append({"class": c, "width": w, "height": h})
            colors.append({"class": c, "R": float(arr[..., 0].mean()),
                           "G": float(arr[..., 1].mean()),
                           "B": float(arr[..., 2].mean())})

    return {"class_distribution": dist, "color_signature": colors,
            "dimension_sample": dims}


if __name__ == "__main__":
    CACHE_PATH.write_text(json.dumps(build()), encoding="utf-8")
    print(f"wrote {CACHE_PATH}")
