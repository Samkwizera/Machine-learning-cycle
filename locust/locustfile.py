import io
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent.parent


def _sample_image():
    test_dir = ROOT / "data" / "test"
    if test_dir.exists():
        for jpg in test_dir.rglob("*.jpg"):
            return jpg.read_bytes()
    # fall back to an in-memory image so the test runs without the dataset
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (34, 139, 34)).save(buf, format="JPEG")
    return buf.getvalue()


SAMPLE_IMAGE = _sample_image()


class FlowerUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self):
        files = {"file": ("sample.jpg", SAMPLE_IMAGE, "image/jpeg")}
        with self.client.post("/predict", files=files, catch_response=True) as resp:
            if resp.status_code == 200 and "class" in resp.text:
                resp.success()
            else:
                resp.failure(f"{resp.status_code} {resp.text[:80]}")

    @task(1)
    def health(self):
        self.client.get("/health")
