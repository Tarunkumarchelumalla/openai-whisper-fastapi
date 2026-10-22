import uuid
import whisper
import yt_dlp
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import concurrent.futures
import time
import logging
import torch
from virality_engine import ViralityEngine
from services.clip_service import VideoClippingService

# Initialize services
engine = ViralityEngine()
clip_service = VideoClippingService()


# ---------------- Setup Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------- Load Supabase Config ----------------ā
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
def transcribe_video(
    video_path: str,
    model_size: str = "medium",
    word_timestamps: bool = False,
):
    """
    Transcribes a video using OpenAI Whisper.

    Returns:
        dict containing:
        - full text
        - segments with timestamps
        - optional word-level timestamps
        - transcription_time_sec
    """

    logger.info(
        f"[transcribe_video] Starting transcription | "
        f"video={video_path} | model={model_size}"
    )

    try:
        start_time = time.time()

        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Whisper model on correct device
        model = whisper.load_model(model_size, device=device)
        logger.info(f"[transcribe_video] Model loaded on {device}")

        # Transcribe
        result = model.transcribe(
            video_path,
            fp16=(device == "cuda"),
            word_timestamps=word_timestamps
        )

        elapsed_time = round(time.time() - start_time, 2)
        logger.info(
            f"[transcribe_video] Transcription completed in {elapsed_time}s"
        )

        # Attach timing metadata
        result["transcription_time_sec"] = elapsed_time
        result["device"] = device
        result["model_size"] = model_size

        # Optional: persist captions with random filename
        random_caption_name = str(uuid.uuid4()) 
        save_transcriptions(result, output_base=random_caption_name,mode="text")

        return result

    except Exception as e:
        logger.exception(
            f"[transcribe_video] Failed transcription for {video_path}: {e}"
        )
        raise
# ---------------- Save TXT ----------------
def save_transcriptions(result, output_base="captions", mode="segments"):
    """
    Saves Whisper transcription as JSON.

    mode:
        - "segments" → full detailed output (default)
        - "text"     → only plain text JSON
    """

    logger.info(
        f"[save_transcriptions] Saving JSON ({mode} mode) to {output_base}.json"
    )

    try:
        output_path = Path(f"{output_base}.json")

        # TEXT ONLY MODE
        if mode == "text":
            data = {
                "language": result.get("language"),
                "duration": result.get("duration"),
                "transcription_time_sec": result.get("transcription_time_sec"),
                "text": result.get("text")
            }

        # FULL SEGMENT MODE
        else:
            data = {
                "language": result.get("language"),
                "duration": result.get("duration"),
                "transcription_time_sec": result.get("transcription_time_sec"),
                "text": result.get("text"),
                "segments": []
            }

            for seg in result.get("segments", []):
                segment_data = {
                    "id": seg.get("id"),
                    "start": round(seg.get("start", 0), 3),
                    "end": round(seg.get("end", 0), 3),
                    "text": seg.get("text", "").strip()
                }

                # Include word-level timestamps if present
                if "words" in seg:
                    segment_data["words"] = [
                        {
                            "word": w.get("word"),
                            "start": round(w.get("start", 0), 3),
                            "end": round(w.get("end", 0), 3)
                        }
                        for w in seg["words"]
                    ]

                data["segments"].append(segment_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[save_transcriptions] JSON saved successfully: {output_path}"
        )

        return str(output_path)

    except Exception:
        logger.exception(
            "[save_transcriptions] Failed to save JSON"
        )
        raise
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
    print(torch.cuda.is_available())   # should be True
    print(torch.cuda.get_device_name(0))  # should show your GPU
    asyncio.create_task(worker())

# ---------------- API Models ----------------
class TranscriptionRequest(BaseModel):
    url: str
    model_size: str = "medium"  # tiny, base, small, medium, large
    postId: int

class HandleRequest(BaseModel):
    handlename: str
    
class HandleRequestPath(BaseModel):
    path: str

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

@app.post("/transcribe/handle")
async def transcribe_by_handle(req: HandleRequest):
    logger.info(f"[API] Fetching posts for handle {req.handlename}")
    try:
        # Fetch records from Supabase where script is NULL or empty
        response = (
            supabase.table("instagram_post")
            .select("*")
            .eq("handlename", req.handlename)
            .or_("script.is.null,script.eq.")  # Only posts without script
            .limit(50)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            logger.warning(f"[API] No records found for handle {req.handlename}")
            return {"error": "No records found for this handlename"}

        queued_jobs = []

        for post in response.data:
            post_id = post["id"]
            video_url = post.get("videourl")
            script_status = post.get("scriptstatus", "pending")

            if not video_url:
                logger.warning(f"[API] No video URL found for post {post_id}")
                continue  # skip this one

            if script_status == "completed":
                logger.info(f"[API] Script already completed for post {post_id}")
                continue  # skip this one

            # Queue transcription job
            job_id = str(uuid.uuid4())
            jobs[job_id] = {
                "status": "queued",
                "url": video_url,
                "model_size": "medium",
                "postId": post_id
            }
            await queue.put(job_id)
            logger.info(f"[API] Job {job_id} queued for transcription (postId {post_id})")

            # Update Supabase status
            supabase.table("instagram_post").update({
                "scriptstatus": "queued"
            }).eq("id", post_id).execute()

            queued_jobs.append({"job_id": job_id, "status": "queued", "postId": post_id})

        if not queued_jobs:
            return {"message": "No new posts were queued (all completed/skipped)"}

        return {"queued_jobs": queued_jobs}

    except Exception as e:
        logger.error(f"[API] Failed to fetch and queue transcriptions for handle {req.handlename}: {e}")
        return {"error": str(e)}


@app.get("/transcribe/bypath")
async def transcribe_by_path():
    """Transcribe all videos found in the `videos/` folder.

    Each MP4 file will be passed through `transcribe_video` and the
    resulting JSON (and the saved file from `save_transcriptions`) will be
    returned as a dictionary keyed by filename.  This makes it easy to
    iterate through a directory of clips for batch processing.
    """
    try:
        results = {}
        video_dir = Path("videos")

        if not video_dir.exists() or not video_dir.is_dir():
            logger.warning("[transcribe_by_path] videos directory not found")
            return {"error": "videos directory does not exist"}

        for video_path in video_dir.glob("*.mp4"):
            logger.info(f"[transcribe_by_path] processing {video_path}")
            try:
                captions_json = transcribe_video(str(video_path))
                results[video_path.name] = captions_json
            except Exception as inner_exc:
                logger.error(
                    f"[transcribe_by_path] failed on {video_path}: {inner_exc}"
                )
                results[video_path.name] = {"error": str(inner_exc)}

        return {"captions": results}

    except Exception as e:
        logger.error(e)
        return {"error": str(e)}
