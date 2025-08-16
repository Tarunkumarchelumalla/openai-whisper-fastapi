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
import concurrent.futures
import time
import logging

# ---------------- Setup Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

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

# ---------------- Thread Pool ----------------
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# ---------------- Download Video ----------------
def download_video(url, output_path="video.mp4"):
    logger.info(f"[download_video] Start downloading video from {url}")
    try:
        ydl_opts = {
            'outtmpl': output_path,
            'format': 'mp4/best',
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"[download_video] Completed downloading video: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"[download_video] Failed to download video {url}: {e}")
        raise e

# ---------------- Run Whisper ----------------
def transcribe_video(video_path, model_size="medium"):
    logger.info(f"[transcribe_video] Start transcription for {video_path} with model '{model_size}'")
    try:
        start_time = time.time()
        model = whisper.load_model(model_size)
        logger.info(f"[transcribe_video] Model '{model_size}' loaded")
        
        result = model.transcribe(video_path, fp16=False)
        elapsed_time = time.time() - start_time
        logger.info(f"[transcribe_video] Completed transcription in {elapsed_time:.2f} seconds")

        result["transcription_time_sec"] = elapsed_time
        return result
    except Exception as e:
        logger.error(f"[transcribe_video] Transcription failed for {video_path}: {e}")
        raise e

# ---------------- Save TXT ----------------
def save_transcriptions(result, output_base="captions"):
    logger.info(f"[save_transcriptions] Start saving transcription to {output_base}.txt")
    try:
        txt_path = Path(f"{output_base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        logger.info(f"[save_transcriptions] Saved transcription TXT file: {txt_path}")
        return str(txt_path)
    except Exception as e:
        logger.error(f"[save_transcriptions] Failed to save transcription TXT: {e}")
        raise e

# ---------------- Worker Loop ----------------
async def worker():
    logger.info("[worker] Background worker started")
    while True:
        job_id = await queue.get()
        job = jobs[job_id]
        job["status"] = "processing"
        logger.info(f"[worker] Start processing job {job_id}")

        video_path = f"{job_id}.mp4"

        try:
            logger.info(f"[worker] Downloading video for job {job_id}")
            video_path = await asyncio.get_running_loop().run_in_executor(
                executor, download_video, job["url"], video_path
            )

            logger.info(f"[worker] Transcribing video for job {job_id}")
            result = await asyncio.get_running_loop().run_in_executor(
                executor, transcribe_video, video_path, job["model_size"]
            )

            job["status"] = "completed"
            job["result"] = {"txt_file": result['text']}
            job["transcription_time_sec"] = result.get("transcription_time_sec", 0)
            logger.info(f"[worker] Job {job_id} completed successfully")

            # Update Supabase
            logger.info(f"[worker] Updating Supabase for job {job_id}")
            try:
                supabase.table("instagram_post").update({
                    "scriptstatus": "completed",
                    "script": result['text']
                }).eq("id", job["postId"]).execute()
                logger.info(f"[worker] Supabase update successful for job {job_id}")
            except Exception as e:
                logger.error(f"[worker] Failed to update Supabase for job {job_id}: {e}")

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.error(f"[worker] Job {job_id} failed: {e}")
            try:
                supabase.table("instagram_post").update({
                    "scriptstatus": "failed",
                    "error": str(e)
                }).eq("id", job["postId"]).execute()
                logger.info(f"[worker] Supabase error update successful for job {job_id}")
            except Exception as supa_err:
                logger.error(f"[worker] Failed to update Supabase error for job {job_id}: {supa_err}")

        finally:
            # Clean up downloaded file
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"[worker] Deleted temporary file: {video_path}")
                else:
                    logger.warning(f"[worker] Temporary file not found for deletion: {video_path}")
            except Exception as cleanup_err:
                logger.error(f"[worker] Failed to delete file {video_path}: {cleanup_err}")

            queue.task_done()
            logger.info(f"[worker] Finished job {job_id}")

# ---------------- API Startup ----------------
@app.on_event("startup")
async def startup_event():
    logger.info("[startup] Starting background worker")
    asyncio.create_task(worker())

# ---------------- API Models ----------------
class TranscriptionRequest(BaseModel):
    url: str
    model_size: str = "medium"  # tiny, base, small, medium, large
    postId: int

# ---------------- API Routes ----------------
@app.post("/transcribe")
async def transcribe(req: TranscriptionRequest):
    logger.info(f"[API] Received transcription request for postId {req.postId}")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "url": req.url,
        "model_size": req.model_size,
        "postId": req.postId
    }
    await queue.put(job_id)
    logger.info(f"[API] Job {job_id} queued for transcription")
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    logger.info(f"[API] Status check requested for job_id {job_id}")
    if job_id not in jobs:
        logger.warning(f"[API] Invalid job_id {job_id} for status check")
        return {"error": "Invalid job_id"}
    return jobs[job_id]
