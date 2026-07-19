# Flower species classifier

An end-to-end image classification pipeline. A fine-tuned MobileNetV2 classifies a
flower photo into one of five species, served through a Streamlit app and a FastAPI
service, with bulk data upload and one-click retraining, containerized with Docker
and load-tested with Locust.

Data type: images (non-tabular). Classes: daisy, dandelion, roses, sunflowers, tulips.

| | |
|---|---|
| Video demo (YouTube) | TODO |
| Live app (Streamlit) | https://machine-learning-cycle-3rjwyz84gzajk68kw9bnh2.streamlit.app/ |
| Notebook | [notebook/flowers_classification.ipynb](notebook/flowers_classification.ipynb) |

## What it does

The dataset is ~3,600 real flower photos. The app lets a user:

- predict the species of an uploaded photo, with confidence scores
- view visualizations that interpret the dataset (class balance, image size, colour)
- upload new labelled images in bulk (saved to disk and logged in SQLite)
- retrain the model on the original plus uploaded data from a button, fine-tuning
  the existing model instead of training from scratch

The notebook and the app share the same `src/` modules, so training and serving
don't drift apart.

## Model

MobileNetV2 pretrained on ImageNet, classifier head replaced with dropout and a
linear layer for 5 classes. Trained with transfer learning, data augmentation
(flip, rotation, colour jitter), Adam, L2 weight decay and early stopping on the
validation loss. Evaluated with accuracy, loss, precision, recall and F1, plus a
confusion matrix.

Test set results:

| Metric | Score |
|---|---|
| Accuracy | 0.880 |
| Precision (macro) | 0.883 |
| Recall (macro) | 0.881 |
| F1 (macro) | 0.876 |

Early stopping cut training at epoch 9 (best weights from epoch 5).

## Structure

```
Machine-learning-cycle/
├── notebook/flowers_classification.ipynb   full ML cycle + evaluation
├── src/
│   ├── data_acquisition.py   download + split the dataset
│   ├── preprocessing.py      transforms and dataloaders
│   ├── model.py              build / train / retrain / save / load
│   ├── prediction.py         single-image inference
│   ├── database.py           SQLite log of uploads + retrain events
│   ├── train.py              training entry point
│   └── build_viz_cache.py    precompute viz aggregates for deploy
├── app/
│   ├── streamlit_app.py      predict / visualize / upload / retrain / status
│   └── api.py                FastAPI: /predict, /health
├── locust/locustfile.py      flood test
├── data/                     train/ val/ test/ uploads/ (gitignored)
├── models/flowers_model.pth  trained checkpoint
├── Dockerfile                Streamlit image
├── Dockerfile.api            FastAPI image
├── docker-compose.yml        nginx + N api replicas
└── render.yaml               Render deployment
```

## Setup

Needs Python 3.12 to 3.14. Built and tested on 3.14 with PyTorch (TensorFlow has no
3.14 wheels).

```bash
git clone <repo-url>
cd Machine-learning-cycle

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

python src/data_acquisition.py  # downloads ~220 MB into data/train|val|test
python src/train.py --epochs 15 # optional, a trained model is already in models/

streamlit run app/streamlit_app.py   # http://localhost:8501
```

Prediction API and flood test need a couple of extra tools:

```bash
pip install -r requirements-dev.txt   # fastapi, uvicorn, locust

uvicorn app.api:app --host 0.0.0.0 --port 8000
curl -F "file=@data/test/roses/<some>.jpg" http://localhost:8000/predict
```

## Notebook

[notebook/flowers_classification.ipynb](notebook/flowers_classification.ipynb) covers
data acquisition, three interpreted EDA features, preprocessing, the model,
training, evaluation with five metrics and a confusion matrix, single-image
prediction, and retraining from the saved model.

## Docker

```bash
docker build -t flower-ui .
docker run -p 8501:8501 flower-ui

docker build -f Dockerfile.api -t flower-api .
docker run -p 8000:8000 flower-api
```

## Deployment (Streamlit Community Cloud)

The app is deployed on Streamlit Community Cloud from this GitHub repo:

1. Go to https://share.streamlit.io and sign in with GitHub.
2. New app, pick this repo, branch `main`, main file `app/streamlit_app.py`. Under
   Advanced settings, set the Python version to **3.12** (PyTorch ships stable CPU
   wheels for it; the app itself runs on 3.12-3.14).
3. Deploy. It installs `requirements.txt` and runs the app. Live URL above.

The model and `viz_cache.json` are committed, so prediction and the visualizations
work on the live app without shipping the full dataset. A Docker setup
([Dockerfile](Dockerfile), [render.yaml](render.yaml)) is also included for
container-based hosting.

## Flood test (Locust)

The FastAPI `/predict` endpoint was flooded with Locust while running with a
varying number of uvicorn worker processes (each worker is a separate process,
which mirrors scaling container replicas). The Docker-container version of the same
setup, nginx in front of N replicas, is in [docker-compose.yml](docker-compose.yml)
and [nginx.conf](nginx.conf). Commands are in
[locust/run_load_tests.md](locust/run_load_tests.md).

Results (100 users, 60s ramp-to-flood, 4 logical / 2 physical core machine, one
thread per worker, aggregated over `/predict` and `/health`):

| Workers | Requests | RPS | Median latency (ms) | 95%ile (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 587 | 10.2 | 4900 | 23000 | 0 |
| 2 | 1030 | 17.6 | 4000 | 11000 | 0 |
| 4 | 1310 | 22.0 | 3500 | 9600 | 0 |

Throughput rises from 10 to 22 req/s and the 95th-percentile latency drops from 23s
to 9.6s as workers scale from 1 to 4, with no failures at any level. Scaling past
the two physical cores still helps here because extra workers overlap image decode,
request I/O and inference, and hyperthreading keeps the cores busy.

One thing worth noting: each worker is pinned to a single thread
(`OMP_NUM_THREADS=1`) for this test. Without pinning, PyTorch spreads a single
inference across all cores by default, so running two unpinned workers just made
them contend for the same cores and throughput actually dropped below the
single-worker case (an early unpinned run gave 2 workers ~6 req/s versus ~11 for 1).
Pinning each worker to one thread is what lets them spread across cores and scale.

## Retraining flow

1. Upload Data page: pick a class, upload images in bulk. They are saved to
   `data/uploads/<class>/` and logged in SQLite.
2. Retrain page: press Trigger retraining. It builds a loader over the original
   train set plus the uploads, loads the current model, fine-tunes, evaluates on
   the test set, and saves the new model.
3. The Predict page then uses the retrained model.
