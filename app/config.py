"""Runtime settings, all overridable through environment variables (or .env)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")).resolve())

    # Uploads
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 500))
    allowed_extensions: tuple = (".wav", ".mp3", ".mp4")
    max_queue_depth: int = field(default_factory=lambda: _int("MAX_QUEUE_DEPTH", 200))

    # Workers / queue
    # Run workers inside the API process (simple single-box deploy). Set to 0 when running
    # dedicated `python -m app.worker` processes instead.
    embedded_workers: bool = field(default_factory=lambda: os.getenv("EMBEDDED_WORKERS", "1") == "1")
    worker_count: int = field(default_factory=lambda: _int("WORKER_COUNT", 2))
    max_attempts: int = field(default_factory=lambda: _int("MAX_ATTEMPTS", 3))
    retry_base_delay_sec: float = field(default_factory=lambda: _float("RETRY_BASE_DELAY_SEC", 5))
    lease_sec: int = field(default_factory=lambda: _int("LEASE_SEC", 120))
    poll_interval_sec: float = field(default_factory=lambda: _float("POLL_INTERVAL_SEC", 1))

    # Transcription
    transcriber: str = field(default_factory=lambda: os.getenv("TRANSCRIBER", "whisper"))  # whisper | fake
    default_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    allowed_models: tuple = ("tiny", "base", "small", "medium", "large-v3")
    # Chaos knob for demos/tests: probability that a transcription attempt fails transiently.
    simulate_failure_rate: float = field(default_factory=lambda: _float("SIMULATE_FAILURE_RATE", 0))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
