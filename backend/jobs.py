from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path("data_store")
JOBS_PATH = DATA_DIR / "jobs.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_jobs(path: Path = JOBS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_jobs(jobs: dict[str, Any], path: Path = JOBS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(jobs, handle, ensure_ascii=False, indent=2)
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
        "progress": 0.0,
        "created_at": now,
        "updated_at": now,
        "payload": payload,
        "result": None,
        "artifact_links": {},
        "error": None,
    }
    jobs[job_id] = job
    _save_jobs(jobs, path)
    return job


def update_job(
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    progress: float | None = None,
    artifact_links: dict[str, Any] | None = None,
    path: Path = JOBS_PATH,
) -> dict[str, Any]:
    jobs = _load_jobs(path)
    if job_id not in jobs:
        raise KeyError("Job not found")
    jobs[job_id]["status"] = status
    jobs[job_id]["updated_at"] = _utc_now_iso()
    if result is not None:
        jobs[job_id]["result"] = result
    if error is not None:
        jobs[job_id]["error"] = error
    if progress is not None:
        jobs[job_id]["progress"] = max(0.0, min(1.0, float(progress)))
    if artifact_links is not None:
        jobs[job_id]["artifact_links"] = artifact_links
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
