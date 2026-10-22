"""Blob storage for uploaded media and transcript artifacts.

Layout (one folder per job, so a job is easy to inspect, move or delete):

    data/
      jobs.db                      <- metadata (SQLite)
      media/<job_id>/original.mp4  <- the untouched upload (source of truth, enables re-runs)
      media/<job_id>/audio.wav     <- 16 kHz mono derivative fed to Whisper (deleted after success)
      transcripts/<job_id>.json    <- canonical transcript (text + timed segments)

Only this class touches the filesystem. Swapping it for S3/GCS means implementing the same
methods against object keys; the database only ever stores keys, never absolute paths.
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import BinaryIO

CHUNK = 1024 * 1024


class UploadTooLarge(Exception):
    pass


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        (root / "media").mkdir(parents=True, exist_ok=True)
        (root / "transcripts").mkdir(parents=True, exist_ok=True)

    # ---------- media ----------
    def media_dir(self, job_id: str) -> Path:
        return self.root / "media" / job_id

    def save_upload(self, job_id: str, ext: str, src: BinaryIO, max_bytes: int) -> tuple[str, int, str]:
        """Stream an upload to disk in 1 MB chunks (never fully in memory).

        Returns (storage_key, size_bytes, sha256). Aborts and cleans up once max_bytes is exceeded.
        """
        folder = self.media_dir(job_id)
        folder.mkdir(parents=True, exist_ok=True)
        key = f"media/{job_id}/original{ext}"
        dest = self.root / key
        digest = hashlib.sha256()
        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := src.read(CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLarge()
                    digest.update(chunk)
                    out.write(chunk)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return key, size, digest.hexdigest()

    def path(self, key: str) -> Path:
        return self.root / key

    # ---------- transcripts ----------
    def save_transcript(self, job_id: str, data: dict) -> str:
        key = f"transcripts/{job_id}.json"
        tmp = self.root / f"{key}.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.root / key)  # atomic: readers never see a half-written file
        return key

    def load_transcript(self, key: str) -> dict:
        return json.loads((self.root / key).read_text(encoding="utf-8"))

    # ---------- cleanup ----------
    def delete_job(self, job_id: str, transcript_key: str | None) -> None:
        shutil.rmtree(self.media_dir(job_id), ignore_errors=True)
        if transcript_key:
            (self.root / transcript_key).unlink(missing_ok=True)
