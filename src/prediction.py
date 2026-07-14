from pathlib import Path

import torch

try:
    from .preprocessing import preprocess_image, CLASS_NAMES
    from .model import load_model, DEVICE
except ImportError:
    from preprocessing import preprocess_image, CLASS_NAMES
    from model import load_model, DEVICE

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "flowers_model.pth"

LABEL_DISPLAY = {
    "daisy": "Daisy",
    "dandelion": "Dandelion",
    "roses": "Roses",
    "sunflowers": "Sunflowers",
    "tulips": "Tulips",
}

_cache = {}


def _get_model(model_path=DEFAULT_MODEL_PATH):
    key = str(model_path)
    if key not in _cache:
        _cache[key] = load_model(model_path)
    return _cache[key]


def clear_model_cache():
    # call after retraining so the next prediction picks up the new weights
    _cache.clear()


def predict(image, model_path=DEFAULT_MODEL_PATH):
    net = _get_model(model_path)
    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(net(tensor), dim=1).squeeze(0).cpu()
    idx = int(probs.argmax())
    name = CLASS_NAMES[idx]
    return {
        "class": name,
        "label": LABEL_DISPLAY.get(name, name),
        "confidence": float(probs[idx]),
        "probabilities": {LABEL_DISPLAY.get(c, c): float(p)
                          for c, p in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        r = predict(sys.argv[1])
        print(f"{r['label']} ({r['confidence']:.1%})")
        for label, p in r["probabilities"].items():
            print(f"  {label:12s} {p:.1%}")
    else:
        print("usage: python prediction.py <image_path>")
