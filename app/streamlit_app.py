import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import database as db
import prediction
from preprocessing import CLASS_NAMES, ensure_class_dirs
from prediction import LABEL_DISPLAY

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
UPLOADS_DIR = DATA_DIR / "uploads"
MODEL_PATH = ROOT / "models" / "flowers_model.pth"
VIZ_CACHE_PATH = DATA_DIR / "viz_cache.json"

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

st.set_page_config(page_title="Flower Classifier", layout="wide")
db.init_db()


def uptime_str():
    secs = int(time.time() - st.session_state.start_time)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def model_available():
    return MODEL_PATH.exists()


@st.cache_data(show_spinner=False)
def _viz_cache():
    if VIZ_CACHE_PATH.exists():
        return json.loads(VIZ_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _has_raw_data():
    return TRAIN_DIR.exists() and any(TRAIN_DIR.rglob("*.jpg"))


@st.cache_data(show_spinner=False)
def class_distribution():
    if _has_raw_data():
        rows = []
        for split, d in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
            for c in CLASS_NAMES:
                p = d / c
                rows.append({"split": split, "class": c,
                             "count": len(list(p.glob("*.jpg"))) if p.exists() else 0})
        return pd.DataFrame(rows)
    return pd.DataFrame(_viz_cache().get("class_distribution", []))


@st.cache_data(show_spinner=False)
def color_signature():
    if _has_raw_data():
        import random
        rng = random.Random(0)
        rows = []
        for c in CLASS_NAMES:
            p = TRAIN_DIR / c
            if not p.exists():
                continue
            files = list(p.glob("*.jpg"))
            for f in rng.sample(files, min(40, len(files))):
                with Image.open(f) as im:
                    arr = np.asarray(im.convert("RGB")) / 255.0
                rows.append({"class": c, "R": arr[..., 0].mean(),
                             "G": arr[..., 1].mean(), "B": arr[..., 2].mean()})
        return pd.DataFrame(rows)
    return pd.DataFrame(_viz_cache().get("color_signature", []))


@st.cache_data(show_spinner=False)
def dimension_sample():
    if _has_raw_data():
        import random
        rng = random.Random(1)
        rows = []
        for c in CLASS_NAMES:
            p = TRAIN_DIR / c
            if not p.exists():
                continue
            files = list(p.glob("*.jpg"))
            for f in rng.sample(files, min(40, len(files))):
                with Image.open(f) as im:
                    rows.append({"class": c, "width": im.width, "height": im.height})
        return pd.DataFrame(rows)
    return pd.DataFrame(_viz_cache().get("dimension_sample", []))


st.sidebar.title("Flower Classifier")
page = st.sidebar.radio("Page", ["Status", "Predict", "Visualizations",
                                 "Upload Data", "Retrain"])
st.sidebar.metric("Uptime", uptime_str())
st.sidebar.write("Model: " + ("loaded" if model_available() else "missing"))


if page == "Status":
    st.title("System Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uptime", uptime_str())
    c2.metric("Model", "online" if model_available() else "missing")
    c3.metric("Uploaded images", db.total_uploads())
    if model_available():
        mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, tz=timezone.utc)
        c4.metric("Model updated", mtime.strftime("%Y-%m-%d %H:%M"))
    else:
        c4.metric("Model updated", "-")

    st.subheader("Dataset")
    dist = class_distribution()
    if dist.empty or dist["count"].sum() == 0:
        st.info("No dataset found. Run `python src/data_acquisition.py`.")
    else:
        st.bar_chart(dist.pivot(index="class", columns="split", values="count"))

    st.subheader("Recent retraining events")
    events = db.recent_retrains()
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.caption("No retraining runs yet.")


elif page == "Predict":
    st.title("Predict a flower species")
    if not model_available():
        st.error("Model not found. Train it with `python src/train.py`, then reload.")
    else:
        file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])
        if file is not None:
            image = Image.open(file).convert("RGB")
            left, right = st.columns(2)
            left.image(image, caption="Uploaded image", use_container_width=True)
            with st.spinner("Classifying"):
                result = prediction.predict(image, model_path=MODEL_PATH)
            right.success(f"Prediction: {result['label']}")
            right.metric("Confidence", f"{result['confidence']:.1%}")
            probs = pd.DataFrame({"probability": result["probabilities"]})
            right.bar_chart(probs.sort_values("probability", ascending=False))


elif page == "Visualizations":
    st.title("Dataset visualizations")
    dist = class_distribution()
    if dist.empty or dist["count"].sum() == 0:
        st.info("No dataset found. Run `python src/data_acquisition.py`.")
    else:
        st.header("Class balance")
        st.bar_chart(dist.pivot(index="class", columns="split", values="count"))
        st.caption("Classes are only mildly imbalanced, so I report macro-averaged "
                   "metrics rather than trusting raw accuracy.")

        st.header("Image dimensions")
        st.scatter_chart(dimension_sample(), x="width", y="height", color="class")
        st.caption("The raw photos vary a lot in size and aspect ratio, so a fixed "
                   "224x224 resize is a required preprocessing step.")

        st.header("Colour signature per class")
        st.bar_chart(color_signature().groupby("class")[["R", "G", "B"]].mean())
        st.caption("Colour separates the classes only partly (yellow flowers score "
                   "high on R+G), which is why a CNN reading shape and texture beats "
                   "a plain colour rule.")


elif page == "Upload Data":
    st.title("Upload data for retraining")
    cls = st.selectbox("Class label for these images", CLASS_NAMES,
                       format_func=lambda c: LABEL_DISPLAY.get(c, c))
    files = st.file_uploader("Select images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)

    if st.button("Save uploads", type="primary", disabled=not files):
        ensure_class_dirs(UPLOADS_DIR)
        dest_dir = UPLOADS_DIR / cls
        saved = 0
        for f in files:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            name = f"{stamp}_{f.name}"
            with open(dest_dir / name, "wb") as out:
                out.write(f.getbuffer())
            db.log_upload(name, cls, str(dest_dir / name))
            saved += 1
        st.success(f"Saved {saved} image(s) and logged them to the database.")

    st.subheader("Pending uploads by class")
    counts = db.upload_counts()
    if counts:
        st.dataframe(pd.DataFrame([{"class": c, "count": counts.get(c, 0)}
                                   for c in CLASS_NAMES]), use_container_width=True)
    else:
        st.caption("No uploads yet.")


elif page == "Retrain":
    st.title("Retrain the model")
    st.write("Retraining loads the current model and fine-tunes it on the original "
             "training data plus the uploaded images. It does not train from scratch.")

    total = db.total_uploads()
    st.metric("Uploaded images available", total)
    epochs = st.slider("Fine-tuning epochs", 1, 10, 3)

    if not model_available():
        st.error("No base model found. Train it first with `python src/train.py`.")
    elif st.button("Trigger retraining", type="primary"):
        from preprocessing import get_combined_loader, get_dataloader
        import model as model_mod
        import torch

        ensure_class_dirs(UPLOADS_DIR)
        event_id = db.start_retrain(total)
        try:
            with st.status("Retraining", expanded=True) as status:
                st.write("Building loaders (train + uploads)")
                train_loader = get_combined_loader([TRAIN_DIR, UPLOADS_DIR],
                                                   batch_size=32, train=True)
                val_loader = get_dataloader(VAL_DIR, batch_size=32, train=False)

                st.write(f"Fine-tuning for {epochs} epoch(s)")
                net, _ = model_mod.retrain_model(MODEL_PATH, train_loader, val_loader,
                                                 epochs=epochs, lr=5e-4, patience=2,
                                                 verbose=False)

                st.write("Evaluating on the test set")
                test_loader = get_dataloader(TEST_DIR, batch_size=32, train=False)
                _, test_acc = model_mod.evaluate_loss(
                    net, test_loader, torch.nn.CrossEntropyLoss())

                st.write("Saving the updated model")
                model_mod.save_model(net, MODEL_PATH)
                prediction.clear_model_cache()
                db.finish_retrain(event_id, float(test_acc), "success")
                status.update(label="Retraining complete", state="complete")
            st.success(f"New test accuracy: {test_acc:.1%}. The prediction page now "
                       "uses the retrained model.")
        except Exception as exc:
            db.finish_retrain(event_id, 0.0, f"failed: {exc}")
            st.exception(exc)
