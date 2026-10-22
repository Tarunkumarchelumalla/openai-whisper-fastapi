"""Worker pool: N threads pulling from the durable queue in app.jobs.

Each worker:  claim (lease) -> probe -> extract audio -> transcribe -> save transcript -> complete
A heartbeat thread keeps the lease alive while a long transcription runs; a reaper thread
re-queues jobs whose worker died. Workers are stateless, so the same loop can run as a separate
process (python -m app.worker) on other machines pointed at the same database/storage.
"""
import logging
import os
import socket
import threading
import time
import traceback

from .config import Settings
from .jobs import JobStore
from .storage import LocalStorage
from .transcriber import PermanentError, build_transcriber, extract_audio, probe

log = logging.getLogger(__name__)


class WorkerPool:
    def __init__(self, settings: Settings, store: JobStore, storage: LocalStorage, transcriber=None):
        self.s = settings
        self.store = store
        self.storage = storage
        self.transcriber = transcriber or build_transcriber(settings.transcriber, settings.simulate_failure_rate)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._prefix = f"{socket.gethostname()}:{os.getpid()}"

    # ---------------- lifecycle ----------------
    def start(self, recover_orphans: bool = True) -> None:
        # Only safe when this is the sole worker process; multi-process setups rely on the reaper.
        recovered = self.store.recover_on_startup() if recover_orphans else 0
        if recovered:
            log.warning("Re-queued %d job(s) interrupted by the previous shutdown", recovered)
        for i in range(self.s.worker_count):
            t = threading.Thread(target=self._loop, args=(f"{self._prefix}:w{i}",), daemon=True)
            t.start()
            self._threads.append(t)
        reaper = threading.Thread(target=self._reaper, daemon=True)
        reaper.start()
        self._threads.append(reaper)
        log.info("Started %d worker(s)", self.s.worker_count)

    def stop(self, timeout: float = 5) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)

    # ---------------- loops ----------------
    def _loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            try:
                job = self.store.claim(worker_id, self.s.lease_sec)
            except Exception:
                log.exception("claim failed")
                job = None
            if not job:
                self._stop.wait(self.s.poll_interval_sec)
                continue
            self._run(job, worker_id)

    def _reaper(self) -> None:
        while not self._stop.wait(max(self.s.lease_sec / 4, 1)):
            try:
                n = self.store.reap_expired_leases()
                if n:
                    log.warning("Reaper re-queued %d job(s) with expired leases", n)
            except Exception:
                log.exception("reaper failed")

    # ---------------- one job ----------------
    def _run(self, job: dict, worker_id: str) -> None:
        job_id = job["id"]
        log.info("[%s] claimed %s (attempt %d/%d)", worker_id, job_id, job["attempts"], job["max_attempts"])
        beat_stop = threading.Event()

        def heartbeat():
            while not beat_stop.wait(self.s.lease_sec / 3):
                self.store.heartbeat(job_id, worker_id, self.s.lease_sec)

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        started = time.time()
        try:
            stage = lambda name: self.store.heartbeat(job_id, worker_id, self.s.lease_sec, stage=name)

            stage("probing media")
            src = self.storage.path(job["media_key"])
            if not src.exists():
                raise PermanentError("Uploaded media is missing from storage")
            meta = probe(src)

            stage("extracting audio")
            wav = extract_audio(src, self.storage.media_dir(job_id) / "audio.wav")

            stage(f"transcribing ({job['model']})")
            result = self.transcriber.transcribe(wav, job["model"], job["language"])

            stage("saving transcript")
            elapsed = round(time.time() - started, 2)
            transcript = {
                "job_id": job_id,
                "filename": job["filename"],
                "model": job["model"],
                "language": result["language"],
                "duration_sec": meta["duration"],
                "processing_sec": elapsed,
                "device": result.get("device"),
                "text": result["text"],
                "segments": result["segments"],
            }
            key = self.storage.save_transcript(job_id, transcript)
            ok = self.store.complete(
                job_id, worker_id, transcript_key=key, detected_language=result["language"],
                duration_sec=meta["duration"], processing_sec=elapsed,
            )
            if ok:
                wav.unlink(missing_ok=True)  # derived file; the original upload is kept
                log.info("[%s] completed %s in %.1fs", worker_id, job_id, elapsed)
            else:
                log.warning("[%s] lost ownership of %s before completing; result discarded", worker_id, job_id)
        except PermanentError as e:
            self.store.fail(job_id, worker_id, str(e), transient=False)
            log.error("[%s] %s failed permanently: %s", worker_id, job_id, e)
        except Exception as e:
            status = self.store.fail(job_id, worker_id, f"{type(e).__name__}: {e}", transient=True)
            log.error("[%s] %s attempt failed (%s): %s\n%s", worker_id, job_id, status, e,
                      traceback.format_exc(limit=3))
        finally:
            beat_stop.set()


if __name__ == "__main__":
    # Standalone worker process: `python -m app.worker` (run as many as you have GPUs/machines).
    from .config import settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    pool = WorkerPool(settings, JobStore(settings.db_path, settings.retry_base_delay_sec),
                      LocalStorage(settings.data_dir))
    pool.start(recover_orphans=False)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pool.stop()
