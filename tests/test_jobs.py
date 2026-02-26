from pathlib import Path

from backend.jobs import create_job, get_job, list_jobs, update_job


def test_job_lifecycle(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.json"
    job = create_job("report", "ds_test", {"template": "health"}, path=jobs_path)
    assert job["status"] == "queued"

    fetched = get_job(job["job_id"], path=jobs_path)
    assert fetched is not None
    assert fetched["job_id"] == job["job_id"]

    updated = update_job(job["job_id"], status="completed", result={"report": "ok"}, path=jobs_path)
    assert updated["status"] == "completed"
    assert updated["result"]["report"] == "ok"

    jobs = list_jobs(path=jobs_path)
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job["job_id"]
