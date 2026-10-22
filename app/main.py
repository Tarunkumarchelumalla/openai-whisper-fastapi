"""HTTP API + web UI for the transcription service.

    uvicorn app.main:app            (API + embedded workers + UI on http://localhost:8000)

REST surface (all under /api/v1, documented live at /docs):
    POST   /transcriptions                    multipart upload of 1..N files -> 202 + job(s)
    GET    /transcriptions?status=&limit=     list jobs
    GET    /transcriptions/{id}               job status
    GET    /transcriptions/{id}/transcript    ?format=json|txt|srt|vtt
    GET    /transcriptions/{id}/media         original upload (for playback)
    POST   /transcriptions/{id}/retry         re-queue a failed job
    DELETE /transcriptions/{id}               delete job + media + transcript
    GET    /health, /stats
"""
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, settings as default_settings
from .formats import to_srt, to_txt, to_vtt
from .jobs import JobStore
from .storage import LocalStorage, UploadTooLarge
from .worker import WorkerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("api")

STATIC = Path(__file__).parent / "static"
MEDIA_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4"}


def public(job: dict) -> dict:
    """Shape a DB row for API consumers (hide internal storage keys/lease details)."""
    base = f"/api/v1/transcriptions/{job['id']}"
    return {
        "id": job["id"],
        "filename": job["filename"],
        "media_kind": job["media_kind"],
        "size_bytes": job["size_bytes"],
        "model": job["model"],
        "language": job["language"],
        "status": job["status"],
        "stage": job["stage"],
        "attempts": job["attempts"],
        "max_attempts": job["max_attempts"],
        "next_run_at": job["next_run_at"] if job["status"] == "retrying" else None,
        "last_error": job["last_error"],
        "detected_language": job["detected_language"],
        "duration_sec": job["duration_sec"],
        "processing_sec": job["processing_sec"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "links": {
            "self": base,
            "media": f"{base}/media",
            "transcript": f"{base}/transcript" if job["status"] == "completed" else None,
        },
    }


def create_app(s: Settings = default_settings, transcriber=None) -> FastAPI:
    store = JobStore(s.db_path, s.retry_base_delay_sec)
    storage = LocalStorage(s.data_dir)
    pool = WorkerPool(s, store, storage, transcriber) if s.embedded_workers else None

    @asynccontextmanager
    async def lifespan(_app):
        if pool:
            pool.start()
        yield
        if pool:
            pool.stop()

    app = FastAPI(title="Transcription Service", version="1.0", lifespan=lifespan)
    app.state.store, app.state.storage, app.state.pool = store, storage, pool

    def get_or_404(job_id: str) -> dict:
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    # ---------------- uploads ----------------
    def ingest(f: UploadFile, model: str, language: str | None) -> dict:
        """Validate, stream to storage, dedupe and enqueue one file. Runs in a thread pool so many
        concurrent uploads don't block the event loop."""
        name = Path(f.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in s.allowed_extensions:
            return {"filename": name, "error": f"Unsupported type '{ext}'. Allowed: {', '.join(s.allowed_extensions)}"}
        job_id = str(uuid.uuid4())
        try:
            key, size, sha = storage.save_upload(job_id, ext, f.file, s.max_upload_bytes)
        except UploadTooLarge:
            return {"filename": name, "error": f"File exceeds {s.max_upload_mb} MB limit"}
        if size == 0:
            storage.delete_job(job_id, None)
            return {"filename": name, "error": "File is empty"}

        # Idempotency: identical bytes + same settings already transcribed -> reuse that result.
        dup = store.find_completed_duplicate(sha, model, language)
        if dup:
            storage.delete_job(job_id, None)
            return {**public(dup), "deduplicated": True}

        job = store.create({
            "id": job_id, "filename": name, "media_kind": "video" if ext == ".mp4" else "audio",
            "media_key": key, "size_bytes": size, "sha256": sha, "model": model,
            "language": language, "max_attempts": s.max_attempts,
        })
        log.info("queued %s (%s, %d bytes)", job_id, name, size)
        return public(job)

    @app.post("/api/v1/transcriptions", status_code=202)
    async def create_transcriptions(
        files: list[UploadFile] = File(..., description="One or more .wav / .mp3 / .mp4 files"),
        model: str = Form(None),
        language: str | None = Form(None, description="ISO code, e.g. 'en'. Empty = auto-detect"),
    ):
        model = model or s.default_model
        if model not in s.allowed_models:
            raise HTTPException(422, f"model must be one of {list(s.allowed_models)}")
        language = (language or "").strip().lower() or None

        # Backpressure: refuse new work instead of letting the queue grow without bound.
        if store.queue_depth() + len(files) > s.max_queue_depth:
            return JSONResponse({"detail": "Queue is full, try again later"}, status_code=429,
                                headers={"Retry-After": "30"})

        results = [await run_in_threadpool(ingest, f, model, language) for f in files]
        accepted = [r for r in results if "error" not in r]
        if not accepted:
            return JSONResponse({"jobs": [], "errors": results}, status_code=400)
        return {"jobs": accepted, "errors": [r for r in results if "error" in r]}

    # ---------------- read ----------------
    @app.get("/api/v1/transcriptions")
    def list_transcriptions(status: str | None = None, limit: int = Query(100, le=500), offset: int = 0):
        return {"jobs": [public(j) for j in store.list(status, limit, offset)], "stats": store.stats()}

    @app.get("/api/v1/transcriptions/{job_id}")
    def get_transcription(job_id: str):
        return public(get_or_404(job_id))

    @app.get("/api/v1/transcriptions/{job_id}/transcript")
    def get_transcript(job_id: str, format: str = Query("json", pattern="^(json|txt|srt|vtt)$")):
        job = get_or_404(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, f"Transcript not ready (status: {job['status']})")
        data = storage.load_transcript(job["transcript_key"])
        if format == "json":
            return data
        stem = Path(job["filename"]).stem
        body = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt}[format](data)
        mime = "text/vtt" if format == "vtt" else "text/plain"
        return PlainTextResponse(body, media_type=f"{mime}; charset=utf-8",
                                 headers={"Content-Disposition": f'inline; filename="{stem}.{format}"'})

    @app.get("/api/v1/transcriptions/{job_id}/media")
    def get_media(job_id: str):
        job = get_or_404(job_id)
        path = storage.path(job["media_key"])
        if not path.exists():
            raise HTTPException(404, "Media no longer stored")
        return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"))

    # ---------------- recovery / admin ----------------
    @app.post("/api/v1/transcriptions/{job_id}/retry", status_code=202)
    def retry(job_id: str):
        job = get_or_404(job_id)
        if job["status"] != "failed":
            raise HTTPException(409, f"Only failed jobs can be retried (status: {job['status']})")
        return public(store.requeue(job_id))

    @app.delete("/api/v1/transcriptions/{job_id}", status_code=204)
    def delete(job_id: str):
        job = get_or_404(job_id)
        if job["status"] == "processing":
            raise HTTPException(409, "Job is being processed; try again when it finishes")
        store.delete(job_id)
        storage.delete_job(job_id, job["transcript_key"])

    @app.get("/api/v1/health")
    def health():
        return {"ok": True, "workers": s.worker_count if pool else 0, "transcriber": s.transcriber}

    @app.get("/api/v1/stats")
    def stats():
        return {**store.stats(), "queue_depth": store.queue_depth(), "workers": s.worker_count if pool else 0}

    @app.get("/api/v1/config")
    def config():
        return {"models": list(s.allowed_models), "default_model": s.default_model,
                "max_upload_mb": s.max_upload_mb, "extensions": list(s.allowed_extensions)}

    # ---------------- UI ----------------
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
