from backend.jobs import create_job, update_job


def test_job_progress_updates(tmp_path):
    jobs_path = tmp_path / "jobs.json"
    job = create_job("facts", "ds_1", {}, path=jobs_path)
    updated = update_job(job["job_id"], status="running", progress=0.5, path=jobs_path)
    assert updated["progress"] == 0.5
