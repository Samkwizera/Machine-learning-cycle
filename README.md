# Flower species classifier

An end-to-end image classification pipeline. A fine-tuned MobileNetV2 classifies a
flower photo into one of five species, served through a Streamlit app and a FastAPI
service, with bulk data upload and one-click retraining, containerized with Docker
and load-tested with Locust.

Data type: images (non-tabular). Classes: daisy, dandelion, roses, sunflowers, tulips.

| | |
|---|---|
| Video demo (YouTube) | https://youtu.be/QxjZuCiTYcQ |
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

Test set results (556 held-out images):

| Metric | Score |
|---|---|
| Accuracy | 0.890 |
| Precision (macro) | 0.889 |
| Recall (macro) | 0.888 |
| F1 (macro) | 0.887 |
| Loss | 0.314 |

Early stopping cut training at epoch 7 (best weights from epoch 4, val loss 0.289).

The notebook then fine-tunes that model for 4 more epochs to demonstrate the retrain
path, which lifts test accuracy to **0.906**. That retrained checkpoint is the one in
`models/flowers_model.pth` and the one the app and API serve, so the numbers above
describe the base model and 0.906 describes what ships.

Per class, roses are hardest (0.77 recall — most often confused with tulips) and
dandelions easiest (0.94 F1). Full classification report and confusion matrix are in
the notebook.

The checkpoint is saved as `models/flowers_model.pth`. `.pth` is PyTorch's saved-model
format, the equivalent of `.h5` or `.tf` for Keras and `.pkl` for scikit-learn — the
stack here is PyTorch because TensorFlow has no Python 3.14 wheels. It holds the
`state_dict`, which `src/model.py` loads back into the MobileNetV2 architecture.

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
├── data/                     train/ val/ test/ uploads/ (images not committed)
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
prediction, and retraining from the saved model. It is committed with all outputs
saved, so the plots, metrics and confusion matrix are visible without re-running it.

One artefact worth explaining: epoch 7 reports 38368.5s against ~600-700s for every
other epoch. Training ran overnight on a laptop that suspended mid-epoch — the wall
clock kept counting while the machine slept. It affects that one timing figure and
nothing else.

## Docker

Both images are built and verified: the UI answers `/_stcore/health` on 8501, and the
API returns a real prediction on 8000.

```bash
docker build -t flower-ui .
docker run -p 8501:8501 flower-ui

docker build -f Dockerfile.api -t flower-api .
docker run -p 8000:8000 flower-api
curl -F "file=@data/test/daisy/<some>.jpg" http://localhost:8000/predict
```

The load-balanced topology used for the flood test (nginx in front of N API
replicas):

```bash
docker compose up -d --scale api=2
curl http://localhost:8000/health
docker compose down
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

The FastAPI `/predict` endpoint was flooded with Locust against a varying number of
**Docker container replicas** behind an nginx load balancer
([docker-compose.yml](docker-compose.yml), [nginx.conf](nginx.conf)). Commands are
in [locust/run_load_tests.md](locust/run_load_tests.md).

Results (100 users, 60s ramp-to-flood, 4 logical / 2 physical core machine, each
replica capped at 1.0 CPU and one thread, aggregated over `/predict` and `/health`):

| Replicas | Requests | RPS | Median latency (ms) | 95%ile (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 942 | 16.1 | 5600 | 6500 | 0 |
| 2 | 1224 | 20.5 | 4300 | 5500 | 0 |
| 4 | 731 | 12.2 | 6500 | 15000 | 0 |

Throughput rises from 16 to 20.5 req/s going from one replica to two, and the median
latency drops from 5.6s to 4.3s, with no failures. Going to four replicas makes
things **worse**, not better: throughput falls to 12.2 req/s and the 95th percentile
blows out from 5.5s to 15s. Four replicas each capped at 1.0 CPU want four cores,
and they are competing with nginx, the Docker VM and Locust itself for the same four
logical (two physical) cores, so the box is oversubscribed and time is lost to
context switching rather than spent on inference. On this hardware the sweet spot is
two replicas; scaling past the physical core count needs a bigger host, not more
containers.

nginx spread the load evenly at both levels (510/511 requests across two replicas,
155/166/164/158 across four), so the drop-off is CPU contention, not a balancing
artifact.

For reference, an earlier run of the same flood against N **uvicorn worker
processes** in a single container (no CPU caps, workers free to share all cores)
scaled differently — 10.2 → 17.6 → 22.0 req/s at 1, 2 and 4 workers:

| Workers | Requests | RPS | Median latency (ms) | 95%ile (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 587 | 10.2 | 4900 | 23000 | 0 |
| 2 | 1030 | 17.6 | 4000 | 11000 | 0 |
| 4 | 1310 | 22.0 | 3500 | 9600 | 0 |

Workers keep improving at 4 where containers regress because the OS is free to
schedule them across whatever cores are idle, while the container replicas are each
held to a hard 1.0 CPU limit and cannot borrow slack from one another.

One thing worth noting: every worker and every replica is pinned to a single thread
(`OMP_NUM_THREADS=1`, set in [docker-compose.yml](docker-compose.yml) for the
containers). Without pinning, PyTorch spreads a single inference across all cores by
default, so running two unpinned workers just made them contend for the same cores
and throughput actually dropped below the single-worker case (an early unpinned run
gave 2 workers ~6 req/s versus ~11 for 1). Pinning each unit to one thread is what
lets them spread across cores and scale.

## Retraining flow

1. Upload Data page: pick a class, upload images in bulk. They are saved to
   `data/uploads/<class>/` and logged in SQLite.
2. Retrain page: press Trigger retraining. It builds a loader over the original
   train set plus the uploads, loads the current model, fine-tunes, evaluates on
   the test set, and saves the new model.
3. The Predict page then uses the retrained model.
