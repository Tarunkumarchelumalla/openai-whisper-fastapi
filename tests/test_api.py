"""End-to-end tests: real HTTP API, real queue + workers, real ffmpeg, fake speech model.

    python -m pytest -q
"""
import dataclasses
import io
import math
import struct
import threading
import time
import wave

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.transcriber import FakeTranscriber, TransientError


def make_wav(seconds=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(int(seconds * rate))))
    return buf.getvalue()


class FlakyTranscriber(FakeTranscriber):
    """Fails the first `failures` calls with a transient error, then succeeds."""

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0
        self.lock = threading.Lock()

    def transcribe(self, *a, **kw):
        with self.lock:
            self.calls += 1
            if self.calls <= self.failures:
                raise TransientError("boom")
        return super().transcribe(*a, **kw)


def make_client(tmp_path, transcriber=None, **overrides):
    opts = dict(data_dir=tmp_path, transcriber="fake", worker_count=2, retry_base_delay_sec=0.1,
                poll_interval_sec=0.05, lease_sec=5, max_upload_mb=5, embedded_workers=True)
    s = dataclasses.replace(Settings(), **{**opts, **overrides})
    return TestClient(create_app(s, transcriber or FakeTranscriber()))


def wait_for(client, job_id, statuses=("completed", "failed"), timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/transcriptions/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stuck in {job['status']}")


def upload(client, name, data, **form):
    return client.post("/api/v1/transcriptions", files=[("files", (name, data))], data=form)


def test_upload_transcribe_and_download_formats(tmp_path):
    with make_client(tmp_path) as c:
        r = upload(c, "hello.wav", make_wav())
        assert r.status_code == 202
        job = wait_for(c, r.json()["jobs"][0]["id"])
        assert job["status"] == "completed" and job["attempts"] == 1
        assert job["duration_sec"] == pytest.approx(1.0, abs=0.05)
        t = c.get(f"/api/v1/transcriptions/{job['id']}/transcript").json()
        assert t["text"] and t["segments"]
        assert c.get(f"/api/v1/transcriptions/{job['id']}/transcript?format=srt").text.startswith("1\n00:00:00,000")
        assert c.get(f"/api/v1/transcriptions/{job['id']}/transcript?format=vtt").text.startswith("WEBVTT")
        assert c.get(f"/api/v1/transcriptions/{job['id']}/media").content[:4] == b"RIFF"


def test_concurrent_uploads_each_processed_exactly_once(tmp_path):
    fake = FlakyTranscriber(failures=0)
    with make_client(tmp_path, fake) as c:
        # Different bytes per file so dedupe doesn't collapse them.
        results = [None] * 8
        def go(i):
            results[i] = upload(c, f"f{i}.wav", make_wav(0.5 + i * 0.05))
        threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        ids = [r.json()["jobs"][0]["id"] for r in results]
        assert len(set(ids)) == 8
        for i in ids:
            assert wait_for(c, i)["status"] == "completed"
        assert fake.calls == 8  # no job claimed twice


def test_transient_failure_is_retried_with_backoff(tmp_path):
    with make_client(tmp_path, FlakyTranscriber(failures=2), max_attempts=3) as c:
        job_id = upload(c, "a.wav", make_wav()).json()["jobs"][0]["id"]
        job = wait_for(c, job_id)
        assert job["status"] == "completed" and job["attempts"] == 3


def test_exhausted_retries_fail_then_manual_retry_recovers(tmp_path):
    flaky = FlakyTranscriber(failures=2)
    with make_client(tmp_path, flaky, max_attempts=2) as c:
        job_id = upload(c, "a.wav", make_wav()).json()["jobs"][0]["id"]
        job = wait_for(c, job_id)
        assert job["status"] == "failed" and "boom" in job["last_error"]
        assert c.post(f"/api/v1/transcriptions/{job_id}/retry").status_code == 202
        assert wait_for(c, job_id)["status"] == "completed"


def test_corrupt_file_fails_permanently_without_retry(tmp_path):
    with make_client(tmp_path, max_attempts=3) as c:
        job_id = upload(c, "bad.mp3", b"this is not audio" * 100).json()["jobs"][0]["id"]
        job = wait_for(c, job_id)
        assert job["status"] == "failed" and job["attempts"] == 1


def test_validation_rejects_bad_type_and_oversize(tmp_path):
    with make_client(tmp_path) as c:
        r = upload(c, "notes.txt", b"hello")
        assert r.status_code == 400 and "Unsupported" in r.json()["errors"][0]["error"]
        r = upload(c, "big.wav", b"0" * (6 * 1024 * 1024))
        assert r.status_code == 400 and "limit" in r.json()["errors"][0]["error"]


def test_duplicate_upload_reuses_completed_result(tmp_path):
    with make_client(tmp_path) as c:
        data = make_wav()
        first = upload(c, "a.wav", data).json()["jobs"][0]["id"]
        wait_for(c, first)
        again = upload(c, "copy.wav", data).json()["jobs"][0]
        assert again["id"] == first and again["deduplicated"]


def test_crashed_worker_lease_is_reaped_and_job_requeued(tmp_path):
    with make_client(tmp_path, lease_sec=1, embedded_workers=False) as c:
        store = c.app.state.store
        job_id = upload(c, "a.wav", make_wav()).json()["jobs"][0]["id"]
        assert store.claim("dead-worker", lease_sec=1)["id"] == job_id  # worker takes it, then "dies"
        time.sleep(1.1)
        assert store.reap_expired_leases() == 1
        assert store.get(job_id)["status"] == "retrying"
        # the dead worker waking up later must not be able to overwrite anything
        assert store.complete(job_id, "dead-worker", transcript_key="x") is False
