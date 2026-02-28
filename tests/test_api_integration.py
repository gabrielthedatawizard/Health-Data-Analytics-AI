import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def _create_session() -> str:
    resp = client.post("/sessions", json={"created_by": "tester"})
    assert resp.status_code == 200
    return resp.json()["dataset_id"]


def _upload_small_csv(dataset_id: str) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = client.post(f"/sessions/{dataset_id}/upload", files=files, data={"uploaded_by": "tester"})
    assert resp.status_code == 200


def test_facts_async_enqueue(monkeypatch):
    dataset_id = _create_session()
    _upload_small_csv(dataset_id)
    monkeypatch.setattr(main, "SMALL_ROW_MAX", 1)
    resp = client.get(f"/sessions/{dataset_id}/facts")
    assert resp.status_code in (200, 202)
    payload = resp.json()
    assert "job_id" in payload or "facts_bundle" in payload
    if "job_id" in payload:
        status = client.get(f"/jobs/{payload['job_id']}")
        assert status.status_code == 200
        assert status.json()["status"] in {"queued", "running", "succeeded", "failed", "completed"}


def test_jobs_endpoint():
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()


def test_report_job_queue():
    dataset_id = _create_session()
    _upload_small_csv(dataset_id)
    resp = client.post(f"/sessions/{dataset_id}/report", json={})
    assert resp.status_code in (200, 202)
    assert "job_id" in resp.json()
