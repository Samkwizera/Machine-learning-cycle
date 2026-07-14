import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            class TEXT NOT NULL,
            path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS retrain_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            n_new_images INTEGER,
            test_accuracy REAL,
            status TEXT)""")


def log_upload(filename, cls, path):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO uploads (filename, class, path, uploaded_at) VALUES (?,?,?,?)",
            (filename, cls, path, datetime.now(timezone.utc).isoformat()))


def upload_counts():
    with _connect() as conn:
        rows = conn.execute(
            "SELECT class, COUNT(*) AS n FROM uploads GROUP BY class").fetchall()
    return {r["class"]: r["n"] for r in rows}


def total_uploads():
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]


def start_retrain(n_new):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO retrain_events (started_at, n_new_images, status) VALUES (?,?,?)",
            (datetime.now(timezone.utc).isoformat(), n_new, "running"))
        return cur.lastrowid


def finish_retrain(event_id, test_accuracy, status="success"):
    with _connect() as conn:
        conn.execute(
            "UPDATE retrain_events SET finished_at=?, test_accuracy=?, status=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), test_accuracy, status, event_id))


def recent_retrains(limit=5):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print("db ready at", DB_PATH)
