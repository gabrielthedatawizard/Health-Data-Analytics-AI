from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path("data_store")
JOBS_PATH = DATA_DIR / "jobs.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_JOBS_LOCK = threading.Lock()


def _load_jobs(path: Path = JOBS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    # Retry on torn writes from concurrent API/worker processes.
    for attempt in range(5):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            if attempt == 4:
                return {}
            time.sleep(0.02 * (attempt + 1))
    return {}


def _save_jobs(jobs: dict[str, Any], path: Path = JOBS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JOBS_LOCK:
        tmp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(jobs, handle, ensure_ascii=False, indent=2)
        try:
            tmp_path.replace(path)
        except OSError:
            # Cross-process race: reload, merge, retry once.
            existing = _load_jobs(path)
            existing.update(jobs)
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(existing, handle, ensure_ascii=False, indent=2)
            tmp_path.replace(path)


def create_job(job_type: str, dataset_id: str, payload: dict[str, Any], path: Path = JOBS_PATH) -> dict[str, Any]:
    jobs = _load_jobs(path)
    job_id = str(uuid.uuid4())
    now = _utc_now_iso()
    job = {
        "job_id": job_id,
        "type": job_type,
        "dataset_id": dataset_id,
        "status": "queued",
        "progress": 0,
        "created_at": now,
        "updated_at": now,
        "payload": payload,
        "result": None,
        "artifacts": {"facts": None, "spec": None, "report_pdf": None},
        "error": None,
    }
    jobs[job_id] = job
    _save_jobs(jobs, path)
    return job


def update_job(
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | dict[str, Any] | None = None,
    progress: float | int | None = None,
    artifacts: dict[str, Any] | None = None,
    artifact_links: dict[str, Any] | None = None,  # backward compatibility
    path: Path = JOBS_PATH,
) -> dict[str, Any]:
    jobs = _load_jobs(path)
    if job_id not in jobs:
        raise KeyError("Job not found")
    jobs[job_id]["status"] = status
    jobs[job_id]["updated_at"] = _utc_now_iso()
    if result is not None:
        jobs[job_id]["result"] = result
    if progress is not None:
        jobs[job_id]["progress"] = int(max(0.0, min(100.0, float(progress))))
    resolved_artifacts = artifacts if artifacts is not None else artifact_links
    if resolved_artifacts is not None:
        existing = jobs[job_id].get("artifacts") or {}
        existing.update(resolved_artifacts)
        jobs[job_id]["artifacts"] = existing
    if error is not None:
        if isinstance(error, str):
            jobs[job_id]["error"] = {"code": "job_error", "message": error, "trace": ""}
        else:
            jobs[job_id]["error"] = {
                "code": str(error.get("code", "job_error")),
                "message": str(error.get("message", "Unknown job failure")),
                "trace": str(error.get("trace", "")),
            }
    _save_jobs(jobs, path)
    return jobs[job_id]


def get_job(job_id: str, path: Path = JOBS_PATH) -> dict[str, Any] | None:
    jobs = _load_jobs(path)
    return jobs.get(job_id)


def list_jobs(dataset_id: str | None = None, path: Path = JOBS_PATH) -> list[dict[str, Any]]:
    jobs = _load_jobs(path)
    items = list(jobs.values())
    if dataset_id:
        items = [job for job in items if job.get("dataset_id") == dataset_id]
    return sorted(items, key=lambda job: job.get("created_at", ""), reverse=True)
