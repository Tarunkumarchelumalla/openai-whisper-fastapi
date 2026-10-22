# Caption Extractor

Upload a `.wav`, `.mp3` or `.mp4` file and get back a transcript with timestamps.
It's built with FastAPI, OpenAI Whisper and ffmpeg, and it comes with a simple web UI.

## How to run

You need Python 3.10+ and ffmpeg installed.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

- Open http://localhost:8000 to upload files and see transcripts.
- API docs are at http://localhost:8000/docs.
- To run the tests: `python -m pytest -q`

## How it works

```
Upload  →  Save file  →  Add job to queue  →  Worker runs Whisper  →  Save transcript
```

1. You upload one or more files.
2. The server saves each file and creates a job with status `queued`. It replies right away with a job id.
3. Background workers pick up jobs from the queue. Each worker converts the file to audio with ffmpeg, then transcribes it with Whisper.
4. The transcript is saved, and the job becomes `completed`.
5. The UI checks every 2 seconds and shows the progress.

## Project structure

| File | What it does |
|---|---|
| `app/main.py` | API endpoints and file uploads |
| `app/jobs.py` | Job database and queue |
| `app/worker.py` | Background workers that process jobs |
| `app/transcriber.py` | ffmpeg and Whisper |
| `app/storage.py` | Saves media files and transcripts |
| `app/static/` | Web UI |
| `tests/` | Tests |

## Design questions

### 1. How would you handle concurrent uploads?

- The upload request doesn't do the transcription. It only saves the file, adds a job to a queue, and replies immediately. Transcription happens in the background.
- The queue is stored in a database, so jobs aren't lost if the server restarts.
- Several workers can run at the same time. Each job is given to exactly one worker, so no file is transcribed twice.
- The Whisper model is loaded once and reused for every job, instead of being loaded again for each file.
- Files are saved in small chunks, so large uploads don't fill up memory.
- If too many jobs are waiting, new uploads are rejected with "try again later" (HTTP 429).
- If the same file is uploaded again, the old transcript is reused.

### 2. How would you store audio and transcripts?

- **Job details** (status, attempts, errors) are stored in a SQLite database.
- **The original upload** is saved in `data/media/<job_id>/`. It's kept so the job can be retried or re-run later.
- **The transcript** is saved as JSON in `data/transcripts/<job_id>.json`. The TXT, SRT and VTT downloads are created from this JSON.
- In production, I'd use Postgres for the database and S3 for the files.

### 3. How do you retry or recover failed transcriptions?

- **Bad files** (not real audio, or no sound track) fail right away, because retrying won't help.
- **Temporary errors** (for example, the GPU runs out of memory) are retried automatically, with a longer wait each time (5s, 10s, 20s), up to 3 attempts.
- **Worker crashes:** while a worker is processing a job, it keeps checking in. If it stops checking in, the job goes back into the queue for another worker.
- **Server restarts:** jobs that were in progress are put back into the queue on startup.
- **Manual retry:** failed jobs show their error in the UI and have a **Retry** button. The original file is still saved, so it doesn't need to be uploaded again.

### 4. How would you expose this as an API?

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/api/v1/transcriptions` | Upload one or more files and get back job ids |
| `GET` | `/api/v1/transcriptions` | List all jobs |
| `GET` | `/api/v1/transcriptions/{id}` | Check a job's status |
| `GET` | `/api/v1/transcriptions/{id}/transcript?format=txt` | Get the transcript (`json`, `txt`, `srt` or `vtt`) |
| `POST` | `/api/v1/transcriptions/{id}/retry` | Retry a failed job |
| `DELETE` | `/api/v1/transcriptions/{id}` | Delete a job and its files |

Example:

```bash
curl -F "files=@audio.mp3" http://localhost:8000/api/v1/transcriptions
curl http://localhost:8000/api/v1/transcriptions/<id>
curl "http://localhost:8000/api/v1/transcriptions/<id>/transcript?format=txt"
```

The API is asynchronous: you upload, get a job id, then check back for the result.
In production, I'd add API keys, and webhooks so clients are notified when a job is done instead of checking repeatedly.
