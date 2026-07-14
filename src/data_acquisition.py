import random
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def _download(url, dest):
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _find_class_root(extract_dir):
    for c in [extract_dir, *extract_dir.rglob("*")]:
        if c.is_dir() and all((c / n).is_dir() for n in CLASS_NAMES):
            return c
    raise FileNotFoundError(f"class folders not found under {extract_dir}")


def acquire(force=False):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not force and all((DATA_ROOT / s).exists() and any((DATA_ROOT / s).iterdir())
                         for s in SPLIT):
        print("splits already present, skipping")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tgz = tmp / "flowers.tgz"
        _download(URL, tgz)
        print("  extracting")
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp / "extracted")
        class_root = _find_class_root(tmp / "extracted")

        for s in SPLIT:
            target = DATA_ROOT / s
            if target.exists():
                shutil.rmtree(target)
            for c in CLASS_NAMES:
                (target / c).mkdir(parents=True)

        rng = random.Random(SEED)
        for c in CLASS_NAMES:
            files = sorted((class_root / c).glob("*.jpg"))
            rng.shuffle(files)
            n = len(files)
            n_train = int(n * SPLIT["train"])
            n_val = int(n * SPLIT["val"])
            buckets = {
                "train": files[:n_train],
                "val": files[n_train:n_train + n_val],
                "test": files[n_train + n_val:],
            }
            for s, flist in buckets.items():
                for f in flist:
                    shutil.copy(f, DATA_ROOT / s / c / f.name)
            print(f"  {c}: {n} -> " + ", ".join(f"{s} {len(v)}" for s, v in buckets.items()))
    print("done")


def summary():
    print("\ndataset summary:")
    for s in SPLIT:
        folder = DATA_ROOT / s
        if not folder.exists():
            continue
        counts = {c: len(list((folder / c).glob("*.jpg"))) for c in CLASS_NAMES}
        print(f"  {s:5s}: {counts} (total {sum(counts.values())})")


if __name__ == "__main__":
    acquire()
    summary()
