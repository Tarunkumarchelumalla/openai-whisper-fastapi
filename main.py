import uuid
import whisper
import yt_dlp
import asyncio
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# ---------------- Load Supabase Config ----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="Video Transcription Service")

# Jobs kept in memory (simple queue)
jobs: Dict[str, dict] = {}
queue = asyncio.Queue()

# ---------------- Download Video ----------------
def download_video(url, output_path="video.mp4"):
    """Download video using yt-dlp"""
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'mp4/best',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# ---------------- Run Whisper ----------------
def transcribe_video(video_path, model_size="medium"):
    """Transcribe video locally with Whisper"""
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, fp16=False)
    return result

# ---------------- Save TXT + SRT ----------------
def save_transcriptions(result, output_base="captions"):
    """Save transcriptions to .txt and .srt"""
    txt_path = Path(f"{output_base}.txt")
    srt_path = Path(f"{output_base}.srt")

    # Plain text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # Subtitles using whisper.get_writer
    srt_writer = whisper.get_writer("srt", ".")
    srt_writer(result, str(srt_path))

    return str(txt_path), str(srt_path)

# ---------------- Worker Loop ----------------
async def worker():
    """Background worker that processes queued jobs"""
    while True:
        job_id = await queue.get()
        job = jobs[job_id]
        job["status"] = "processing"

        video_path = f"{job_id}.mp4"

        try:
            # Download + transcribe
            video_path = download_video(job["url"], output_path=video_path)
            result = transcribe_video(video_path, model_size=job["model_size"])
      
            # Update job state
            job["status"] = "completed"
            job["result"] = {"txt_file": result['text']}

            print(result, job)

            # Update Supabase with results
            supabase.table("instagram_post").update({
                "scriptstatus": "completed",
                "script": result['text']
            }).eq("id", job["postId"]).execute()

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)

            # Update Supabase with error
            supabase.table("instagram_post").update({
                "scriptstatus": "failed",
                "error": str(e)
            }).eq("id", job["postId"]).execute()

        finally:
            # ✅ Clean up downloaded file
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"Deleted temporary file: {video_path}")
                except Exception as cleanup_err:
                    print(f"Failed to delete file {video_path}: {cleanup_err}")

            queue.task_done()
# Start background worker on API startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())

# ---------------- API Models ----------------
class TranscriptionRequest(BaseModel):
    url: str
    model_size: str = "medium"  # tiny, base, small, medium, large
    postId: int

# ---------------- API Routes ----------------
@app.post("/transcribe")
async def transcribe(req: TranscriptionRequest):
    """Submit a transcription job"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "url": req.url, "model_size": req.model_size, "postId": req.postId}
    # Put into queue
    await queue.put(job_id)

    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    """Check job status"""
    if job_id not in jobs:
        return {"error": "Invalid job_id"}
    return jobs[job_id]
