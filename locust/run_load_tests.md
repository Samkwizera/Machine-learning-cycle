# Flood-Test Runbook (uvicorn worker processes)

Since Docker isn't available locally, we approximate "different numbers of
containers" by running the FastAPI app with a varying number of **uvicorn worker
processes** — each worker is a separate OS process serving requests, so scaling
workers mirrors scaling container replicas behind a load balancer. The Docker
version of the same topology is provided in `docker-compose.yml` + `nginx.conf`.

## Steps

Open two terminals (both with the venv activated).

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

### Terminal B — flood it with Locust (headless, 100 users, 1 minute)
```powershell
mkdir results -Force
.\.venv\Scripts\python.exe -m locust -f locust/locustfile.py --host http://127.0.0.1:8000 `
    --headless -u 100 -r 20 -t 60s --csv results/workers_1
# repeat with results/workers_2 and results/workers_4 for the other runs
```

Locust prints a summary and writes `results/workers_N_stats.csv`. Record RPS,
median latency, 95th percentile, and failures into the README table.
```
