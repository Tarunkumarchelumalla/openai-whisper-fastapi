# Caption Extractor: audio/video transcription service

Upload `.wav`, `.mp3` or `.mp4` files and get back timestamped transcripts (TXT / SRT / VTT / JSON).
It's built with FastAPI, OpenAI Whisper, ffmpeg and a durable SQLite-backed job queue, and it comes with a web UI.

![flow](https://img.shields.io/badge/upload-%E2%86%92%20queue%20%E2%86%92%20worker%20%E2%86%92%20storage-3b5bdb)

## Quick start

Requirements: Python 3.10+ and **ffmpeg** on `PATH`. A CUDA GPU is optional but much faster.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

- Web UI: http://localhost:8000. Drag in files, watch the status, play the media with synced captions and download transcripts.
- Interactive API docs: http://localhost:8000/docs
- Tests: `python -m pytest -q` (8 end-to-end tests that use real ffmpeg and a fake speech model, so no GPU is needed)

### Configuration (environment variables or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | Default model (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WORKER_COUNT` | `2` | Worker threads in this process |
| `EMBEDDED_WORKERS` | `1` | `0` = API only. Run `python -m app.worker` processes separately |
| `MAX_ATTEMPTS` | `3` | Attempts per job before it is marked `failed` |
| `RETRY_BASE_DELAY_SEC` | `5` | Backoff base (5s, 10s, 20s … ±20% jitter) |
| `LEASE_SEC` | `120` | How long a worker may hold a job without a heartbeat |
| `MAX_UPLOAD_MB` | `500` | Per-file size limit |
| `MAX_QUEUE_DEPTH` | `200` | Above this, uploads get `429 Retry-After` |
| `DATA_DIR` | `data` | Where the DB, media and transcripts live |
| `TRANSCRIBER` | `whisper` | `fake` = no model, for demos and tests |
| `SIMULATE_FAILURE_RATE` | `0` | e.g. `0.5`: randomly fails attempts so you can watch retries happen |

## Architecture

```
 Browser UI / API client
        │  multipart POST (1..N files)
        ▼
 ┌──────────────┐  stream to disk in 1 MB chunks, sha256, validate type/size
 │   FastAPI    │──────────────────────────────┐
 └──────┬───────┘                              ▼
        │ INSERT job (status=queued)     ┌────────────┐
        ▼                                │  Storage   │  media/<id>/original.ext
 ┌──────────────┐  atomic claim + lease  │ (local FS; │  transcripts/<id>.json
 │ SQLite queue │◄──────────────────┐    │  S3-ready) │
 │  (jobs table)│   heartbeat       │    └────────────┘
 └──────────────┘                   │          ▲
        ▲  reaper re-queues   ┌─────┴──────┐   │
        └──expired leases─────│ Worker pool│───┘ probe → ffmpeg 16 kHz wav → Whisper → save
                              └────────────┘
```

| File | Responsibility |
|---|---|
| [app/main.py](app/main.py) | REST API, upload ingestion, backpressure, dedupe, UI hosting |
| [app/jobs.py](app/jobs.py) | Job metadata **and** the durable queue: claim, lease, heartbeat, retry, reaper |
| [app/worker.py](app/worker.py) | Worker pool loop. Also runs standalone with `python -m app.worker` |
| [app/transcriber.py](app/transcriber.py) | ffprobe validation, ffmpeg audio extraction, cached Whisper models, error classification |
| [app/storage.py](app/storage.py) | Blob storage for media and transcripts (the only code that touches the filesystem) |
| [app/formats.py](app/formats.py) | Renders TXT / SRT / VTT from the canonical JSON |
| [app/static/](app/static/) | Web UI (plain HTML/CSS/JS, no build step) |
| [tests/test_api.py](tests/test_api.py) | End-to-end tests: concurrency, retries, crash recovery, validation, dedupe |
| [legacy/](legacy/) | The earlier Instagram-URL/Supabase pipeline, kept intact (`uvicorn legacy.instagram_main:app`) |

---

## Design questions

### 1. How would you handle concurrent uploads?

**Split accepting work from doing work.** Transcription takes seconds to minutes of GPU time, so the
request path never runs it. An upload is:

1. validated (extension allow-list and size limit),
2. streamed to storage in 1 MB chunks on a thread pool, so neither the event loop nor RAM is blocked by large files,
3. hashed (sha256) and inserted as a `queued` row,
4. answered with **`202 Accepted`** and a job id, usually within milliseconds.

**The queue is a database table, not an in-memory `asyncio.Queue`.** The original version of this
project used an in-memory queue, so a restart lost every pending job. Now workers claim jobs with a
single atomic statement:

```sql
UPDATE jobs SET status='processing', worker_id=?, lease_expires_at=?, attempts=attempts+1
 WHERE id = (SELECT id FROM jobs WHERE status IN ('queued','retrying') AND next_run_at <= now
             ORDER BY next_run_at, created_at LIMIT 1)
RETURNING *;
```

Any number of worker threads or processes can run this at the same time, and no job is handed out
twice. `test_concurrent_uploads_each_processed_exactly_once` checks this with 8 parallel uploads.

**GPU-aware concurrency.** Each Whisper model is loaded **once per process**. The old code reloaded
it for every job, which cost seconds and gigabytes each time. One GPU runs one inference efficiently,
so model calls are serialised with a lock. Having more than one worker still helps, because ffprobe,
ffmpeg decoding, file I/O and DB writes for the next job overlap with the current inference. To scale
further, set `EMBEDDED_WORKERS=0` and run one `python -m app.worker` per GPU or machine against the
same database and storage.

**Protecting the system:**
- Per-file size limit (the upload is aborted mid-stream as soon as it exceeds the limit) and a type allow-list. ffprobe then confirms the file really is media.
- **Backpressure:** when the queue is deeper than `MAX_QUEUE_DEPTH`, new uploads get `429` with `Retry-After` instead of piling up without limit.
- **Idempotency:** identical bytes with the same model and language return the existing completed job (`"deduplicated": true`), so double-clicks and client retries don't use GPU time twice.
- The UI uploads files in parallel (3 at a time), one request per file, so each file gets its own progress bar and its own error.

### 2. How would you store audio and transcripts?

Metadata and blobs are kept separate:

- **Metadata** goes in a `jobs` table: status, stage, attempts, last error, timings, detected language, duration, and storage *keys*. SQLite in WAL mode lets the UI and API read while workers write. In production I'd use **Postgres**. The claim query above works there unchanged with `FOR UPDATE SKIP LOCKED`.
- **Blobs** go through `LocalStorage`, keyed per job:
  - `media/<id>/original.<ext>`: the untouched upload. It's kept as the source of truth so jobs can be re-run later with a better model, and so the UI can play it back.
  - `media/<id>/audio.wav`: a 16 kHz mono derivative (exactly what Whisper wants). It's built once so retries skip the decode step, and deleted after success.
  - `transcripts/<id>.json`: the **canonical** transcript (full text, timed segments, language, model, device, timings). TXT, SRT and VTT are rendered from it on request, so only one copy is stored and the formats can't drift apart.
- Transcript writes are **atomic** (write to a temp file, then rename), so a crash never leaves half a JSON file.
- The DB holds keys, never absolute paths. Moving to S3/GCS means reimplementing one small class, and presigned URLs could then serve media directly.
- Deleting a job removes its row, media folder and transcript together.
- For production I'd add a retention policy (for example, delete originals after N days and keep transcripts), and full-text search over transcripts with Postgres `tsvector` or OpenSearch.

### 3. How do you retry or recover failed transcriptions?

**Errors are classified first, because blind retries waste GPU time:**

| Kind | Examples | Behaviour |
|---|---|---|
| **Permanent** | not a media file, no audio track, undecodable codec | `failed` immediately after 1 attempt |
| **Transient** | CUDA OOM or runtime error, ffmpeg killed, disk or I/O hiccup, anything unexpected | `retrying` with **exponential backoff + jitter** (5s → 10s → 20s, ±20%) up to `MAX_ATTEMPTS` |

**Crash recovery uses leases.** Each claim sets `lease_expires_at`, and a heartbeat thread extends it
while a long transcription runs. If a worker process dies or hangs, the lease expires and a **reaper**
moves the job back to the queue (or to `failed` if it has no attempts left). When the server boots,
any job left in `processing` by the previous process is re-queued immediately.

**Fencing.** Every write a worker makes is conditional on `worker_id = me AND status = 'processing'`.
Say a worker stalls, its lease expires, and another worker takes the job over. When the first worker
wakes up, its result is discarded instead of overwriting the newer one
(`test_crashed_worker_lease_is_reaped_and_job_requeued`).

**Dead-letter and manual retry.** Jobs that exhaust their attempts stay in `failed` with `last_error`
visible in the UI and API. The **Retry** button, or `POST /api/v1/transcriptions/{id}/retry`, re-queues
the job with a fresh attempt budget. The original upload is still in storage, so the user never has to
upload it again.

You can watch all of this live by starting the server with `SIMULATE_FAILURE_RATE=0.5`.

### 4. How would you expose this as an API?

It's a resource-oriented, **asynchronous** REST API, versioned under `/api/v1`, with an OpenAPI spec at `/docs`:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/transcriptions` | multipart `files` (1..N), optional `model`, `language` → **202** `{jobs:[…], errors:[…]}` |
| `GET` | `/api/v1/transcriptions?status=&limit=&offset=` | list jobs and per-status counts |
| `GET` | `/api/v1/transcriptions/{id}` | status, stage, attempts, error, timings, links |
| `GET` | `/api/v1/transcriptions/{id}/transcript?format=json\|txt\|srt\|vtt` | result (`409` until completed) |
| `GET` | `/api/v1/transcriptions/{id}/media` | original upload (supports range requests for seeking) |
| `POST` | `/api/v1/transcriptions/{id}/retry` | re-queue a failed job |
| `DELETE` | `/api/v1/transcriptions/{id}` | remove job, media and transcript |
| `GET` | `/api/v1/health`, `/api/v1/stats`, `/api/v1/config` | liveness, queue metrics, client config |

```bash
curl -F "files=@meeting.mp3" -F "files=@demo.mp4" -F model=small http://localhost:8000/api/v1/transcriptions
curl http://localhost:8000/api/v1/transcriptions/<id>
curl "http://localhost:8000/api/v1/transcriptions/<id>/transcript?format=srt"
```

Design choices:
- **202 + polling** instead of holding the HTTP connection open for minutes. Each job response includes `links` to its transcript and media.
- **Partial success:** in a multi-file upload, valid files are queued and invalid ones are reported in `errors`. One bad file doesn't reject the whole batch.
- **Meaningful status codes:** `400` validation, `404` unknown job, `409` wrong state (transcript not ready, retrying a non-failed job), `422` bad model, `429` backpressure.
- Internal details (storage keys, worker ids, leases) are never exposed.

**For production I'd add:** API-key or OAuth auth with per-tenant job isolation and rate limits;
**webhooks** (`callback_url`) or Server-Sent Events instead of polling; **presigned direct-to-S3
uploads** for very large files, so the API never streams the bytes; an `Idempotency-Key` header; and a
real broker (SQS, Redis Streams, RabbitMQ) once a single database becomes the bottleneck.

---

## Key decisions and trade-offs

- **SQLite queue rather than Redis/Celery.** It has zero extra infrastructure, it's durable, and it's transactional, which is the right size for one machine. The `JobStore` interface (claim, heartbeat, complete, fail, reap) maps directly onto Postgres `SKIP LOCKED` or SQS (visibility timeout = lease) when it needs to scale.
- **Threads, not asyncio, in workers.** Whisper and ffmpeg are blocking and release the GIL, so threads are simple and effective. The API itself stays async.
- **Normalise to 16 kHz mono WAV first.** One code path for wav, mp3 and mp4, a smaller input for the model, and retries skip the decode step.
- **Keep the original upload.** It costs more storage, but it enables re-processing, playback and manual retry without asking the user to upload again.
- **Default model `small`.** It's a good speed and accuracy balance on a consumer GPU. Callers can choose a model per upload.
- **No frontend build step.** The UI is plain HTML, CSS and JS served by FastAPI, so `uvicorn app.main:app` is the whole deployment.
