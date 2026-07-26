# Flood-Test Runbook

Two variants of the same flood test. The container variant is the one reported in the
README; the worker variant is kept because it was run first and shows a different
scaling curve (no CPU caps).

## A. Container replicas behind nginx (primary)

`docker-compose.yml` runs N replicas of the FastAPI image behind nginx on port 8000.
Each replica is capped at 1.0 CPU and pinned to one thread.

### Terminal A — bring up the stack

```powershell
docker compose up -d --build --scale api=1   # then 2, then 4
```

Wait for the load balancer to answer before flooding — nginx returns 502 for a few
seconds while the replicas boot and warm the model:

```powershell
curl http://localhost:8000/health           # expect {"status":"ok","model_loaded":true}
```

Tear down between runs so each level starts clean:

```powershell
docker compose down
```

### Terminal B — flood it

```powershell
mkdir results -Force
.\.venv\Scripts\python.exe -m locust -f locust/locustfile.py --host http://localhost:8000 `
    --headless -u 100 -r 20 -t 60s --csv results/c1
# repeat with results/c2 and results/c4 for the other levels
```

Check that nginx actually spread the load, otherwise the numbers describe one replica
rather than N:

```powershell
docker logs machine-learning-cycle-api-1 | Select-String "POST /predict" | Measure-Object -Line
```

## B. Uvicorn worker processes (reference)

Each worker is a separate OS process sharing all cores, with no CPU limits.

### Terminal A — serve the API with N workers

Pin each worker to one thread first, otherwise PyTorch spreads one inference across
all cores and extra workers just contend for the same cores instead of scaling.

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1

# 1 worker
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1

# 2 workers  (stop the previous one first)
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 2

# 4 workers
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 4
```

### Terminal B — flood it

```powershell
.\.venv\Scripts\python.exe -m locust -f locust/locustfile.py --host http://127.0.0.1:8000 `
    --headless -u 100 -r 20 -t 60s --csv results/w1
# repeat with results/w2 and results/w4
```

Locust prints a summary and writes `results/<name>_stats.csv`. Record RPS, median
latency, 95th percentile and failures into the README tables.
