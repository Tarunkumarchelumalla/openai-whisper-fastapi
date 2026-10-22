"""Job metadata + a durable work queue, both backed by one SQLite table.

Why the queue lives in the database instead of an in-memory asyncio.Queue (the old design):
  * Jobs survive a crash/restart - nothing is lost if the process dies mid-transcription.
  * Claiming a job is a single atomic UPDATE ... RETURNING, so any number of worker threads
    (or worker processes on the same DB) can pull concurrently without double-processing.
  * Each claim takes a time-limited *lease*. Workers heartbeat while busy; if a worker dies,
    the lease expires and the reaper puts the job back on the queue (crash recovery).
  * Writes from a worker are fenced by worker_id, so a "zombie" worker whose lease already
    expired cannot overwrite the result of the worker that took the job over.

Lifecycle:  queued -> processing -> completed
                          |  \
                          |   -> retrying (transient error, backoff) -> processing ...
                          -> failed (permanent error, or attempts exhausted; manual retry re-queues)
"""
import random
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    filename         TEXT NOT NULL,
    media_kind       TEXT NOT NULL,           -- audio | video
    media_key        TEXT NOT NULL,           -- storage key of the original upload
    size_bytes       INTEGER NOT NULL,
    sha256           TEXT NOT NULL,
    model            TEXT NOT NULL,
    language         TEXT,                    -- requested (NULL = auto-detect)
    status           TEXT NOT NULL,           -- queued | processing | retrying | completed | failed
    stage            TEXT,                    -- human readable progress inside 'processing'
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL,
    next_run_at      REAL NOT NULL,           -- earliest time a worker may (re)claim it
    worker_id        TEXT,
    lease_expires_at REAL,
    last_error       TEXT,
    transcript_key   TEXT,
    detected_language TEXT,
    duration_sec     REAL,
    processing_sec   REAL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, next_run_at, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_sha ON jobs(sha256, model);
"""

ACTIVE = ("queued", "retrying", "processing")


class JobStore:
    def __init__(self, db_path: Path, retry_base_delay_sec: float = 5.0):
        self.db_path = db_path
        self.retry_base_delay_sec = retry_base_delay_sec
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")  # readers (API/UI) never block the writer (workers)
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    # ---------------- producer side (API) ----------------
    def create(self, job: dict) -> dict:
        now = time.time()
        row = {
            **job,
            "status": "queued",
            "attempts": 0,
            "next_run_at": now,
            "created_at": now,
            "updated_at": now,
        }
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        with self._conn() as c:
            c.execute(f"INSERT INTO jobs ({cols}) VALUES ({marks})", row)
        return self.get(job["id"])

    def get(self, job_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(r) if r else None

    def list(self, status: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        sql, args = "SELECT * FROM jobs", []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def find_completed_duplicate(self, sha256: str, model: str, language: str | None) -> dict | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT * FROM jobs WHERE sha256 = ? AND model = ? AND language IS ? AND status = 'completed' "
                "ORDER BY finished_at DESC LIMIT 1",
                (sha256, model, language),
            ).fetchone()
        return dict(r) if r else None

    def queue_depth(self) -> int:
        with self._conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'retrying')"
            ).fetchone()[0]

    def stats(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        counts = {s: 0 for s in ("queued", "processing", "retrying", "completed", "failed")}
        counts.update({r["status"]: r["n"] for r in rows})
        return counts

    def requeue(self, job_id: str) -> dict | None:
        """Manual retry of a failed job: fresh attempt budget, keeps last_error for context."""
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='queued', stage=NULL, attempts=0, next_run_at=?, updated_at=?, "
                "worker_id=NULL, lease_expires_at=NULL, finished_at=NULL WHERE id=? AND status='failed'",
                (now, now, job_id),
            )
        return self.get(job_id)

    def delete(self, job_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    # ---------------- consumer side (workers) ----------------
    def claim(self, worker_id: str, lease_sec: int) -> dict | None:
        """Atomically take the oldest runnable job. Safe under any number of concurrent workers."""
        now = time.time()
        with self._conn() as c:
            r = c.execute(
                """
                UPDATE jobs
                   SET status='processing', stage='starting', attempts=attempts+1,
                       worker_id=:w, lease_expires_at=:lease, started_at=:now, updated_at=:now
                 WHERE id = (SELECT id FROM jobs
                              WHERE status IN ('queued','retrying') AND next_run_at <= :now
                              ORDER BY next_run_at, created_at LIMIT 1)
                RETURNING *
                """,
                {"w": worker_id, "lease": now + lease_sec, "now": now},
            ).fetchone()
        return dict(r) if r else None

    def heartbeat(self, job_id: str, worker_id: str, lease_sec: int, stage: str | None = None) -> bool:
        """Extend the lease. Returns False if this worker no longer owns the job."""
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET lease_expires_at=?, updated_at=?, stage=COALESCE(?, stage) "
                "WHERE id=? AND worker_id=? AND status='processing'",
                (now + lease_sec, now, stage, job_id, worker_id),
            )
        return cur.rowcount == 1

    def complete(self, job_id: str, worker_id: str, **fields) -> bool:
        now = time.time()
        fields.update(status="completed", stage=None, finished_at=now, updated_at=now,
                      lease_expires_at=None, last_error=None)
        sets = ", ".join(f"{k}=:{k}" for k in fields)
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE jobs SET {sets} WHERE id=:id AND worker_id=:w AND status='processing'",
                {**fields, "id": job_id, "w": worker_id},
            )
        return cur.rowcount == 1

    def fail(self, job_id: str, worker_id: str, error: str, transient: bool) -> str | None:
        """Record a failed attempt. Transient errors are retried with exponential backoff + jitter
        until max_attempts is reached; permanent errors fail immediately. Returns the new status."""
        job = self.get(job_id)
        if not job or job["worker_id"] != worker_id or job["status"] != "processing":
            return None
        now = time.time()
        if transient and job["attempts"] < job["max_attempts"]:
            delay = self.retry_base_delay_sec * 2 ** (job["attempts"] - 1)
            delay *= random.uniform(0.8, 1.2)
            status, next_run, finished = "retrying", now + delay, None
        else:
            status, next_run, finished = "failed", job["next_run_at"], now
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, stage=NULL, last_error=?, next_run_at=?, finished_at=?, "
                "updated_at=?, lease_expires_at=NULL WHERE id=? AND worker_id=?",
                (status, error[:2000], next_run, finished, now, job_id, worker_id),
            )
        return status

    def reap_expired_leases(self) -> int:
        """Crash recovery: a 'processing' job whose lease ran out belongs to a dead/hung worker.
        Put it back on the queue (or fail it if it has used up its attempts)."""
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='failed', stage=NULL, finished_at=:now, updated_at=:now, "
                "last_error='Worker lost (lease expired) and no attempts left' "
                "WHERE status='processing' AND lease_expires_at < :now AND attempts >= max_attempts",
                {"now": now},
            )
            cur = c.execute(
                "UPDATE jobs SET status='retrying', stage=NULL, next_run_at=:now, updated_at=:now, "
                "last_error='Worker lost (lease expired); re-queued' "
                "WHERE status='processing' AND lease_expires_at < :now",
                {"now": now},
            )
        return cur.rowcount

    def recover_on_startup(self) -> int:
        """Anything 'processing' at boot was owned by the previous (dead) process: re-queue now
        instead of waiting for the lease to expire. Only valid with a single API+worker process."""
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status='retrying', stage=NULL, next_run_at=?, updated_at=?, "
                "last_error='Server restarted during processing; re-queued' WHERE status='processing'",
                (now, now),
            )
        return cur.rowcount
