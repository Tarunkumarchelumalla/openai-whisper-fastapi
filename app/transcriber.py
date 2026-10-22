"""Media preprocessing (ffmpeg) and speech-to-text (Whisper).

Errors are split into two classes so the queue knows what to do with them:
  * PermanentError - retrying cannot help (corrupt file, no audio track, unsupported codec).
  * anything else  - treated as transient (CUDA OOM, ffmpeg killed, disk hiccup...) and retried.
"""
import json
import logging
import random
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class PermanentError(Exception):
    pass


class TransientError(Exception):
    pass


# ---------------- ffmpeg helpers ----------------
def probe(path: Path) -> dict:
    """Validate the file really is media (not just a renamed .exe) and read its duration."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError as e:
        raise TransientError("ffprobe not installed / not on PATH") from e
    if out.returncode != 0:
        raise PermanentError(f"Not a readable media file: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout or "{}")
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    if "audio" not in kinds:
        raise PermanentError("File has no audio track")
    duration = float(info.get("format", {}).get("duration") or 0)
    return {"duration": duration, "has_video": "video" in kinds}


def extract_audio(src: Path, dest: Path) -> Path:
    """Normalise any input (wav/mp3/mp4) to 16 kHz mono PCM - exactly what Whisper consumes.
    Doing it once up front means a retry of the (expensive) model step skips this work."""
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".tmp.wav")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise PermanentError(f"ffmpeg could not decode audio: {res.stderr.strip()[:300]}")
    tmp.replace(dest)
    return dest


# ---------------- transcription engines ----------------
class WhisperTranscriber:
    """Loads each Whisper model once per process and reuses it (the old code reloaded the model
    for every job, which cost several seconds + GBs of RAM each time).

    A single GPU can only run one inference efficiently, so model calls are serialised with a
    lock; worker concurrency still overlaps ffmpeg decoding, I/O and DB work with inference.
    Scale beyond one GPU by running more worker processes against the same queue."""

    def __init__(self):
        self._models: dict = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _model(self, size: str):
        with self._load_lock:
            if size not in self._models:
                import torch
                import whisper

                device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info("Loading whisper model %s on %s", size, device)
                self._models[size] = (whisper.load_model(size, device=device), device)
            return self._models[size]

    def transcribe(self, audio_path: Path, model: str, language: str | None) -> dict:
        m, device = self._model(model)
        with self._infer_lock:
            try:
                result = m.transcribe(str(audio_path), fp16=(device == "cuda"), language=language)
            except RuntimeError as e:  # CUDA OOM and friends: worth another try later
                raise TransientError(f"Whisper runtime error: {e}") from e
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language"),
            "segments": [
                {"id": s["id"], "start": round(s["start"], 3), "end": round(s["end"], 3),
                 "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
            "device": device,
        }


class FakeTranscriber:
    """Deterministic stand-in used by tests and for demoing the pipeline without a GPU."""

    def transcribe(self, audio_path: Path, model: str, language: str | None) -> dict:
        time.sleep(0.2)
        return {
            "text": f"Fake transcript of {audio_path.parent.name}.",
            "language": language or "en",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Fake transcript."}],
            "device": "fake",
        }


class ChaosTranscriber:
    """Wraps a transcriber and randomly raises transient errors - lets you watch the retry/backoff
    path in the UI (SIMULATE_FAILURE_RATE=0.5)."""

    def __init__(self, inner, rate: float):
        self.inner, self.rate = inner, rate

    def transcribe(self, *args, **kwargs) -> dict:
        if random.random() < self.rate:
            raise TransientError("Simulated transient failure (SIMULATE_FAILURE_RATE)")
        return self.inner.transcribe(*args, **kwargs)


def build_transcriber(kind: str, failure_rate: float = 0.0):
    engine = FakeTranscriber() if kind == "fake" else WhisperTranscriber()
    return ChaosTranscriber(engine, failure_rate) if failure_rate > 0 else engine
