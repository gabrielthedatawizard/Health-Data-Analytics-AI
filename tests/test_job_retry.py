import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main
from backend.jobs import create_job, update_job


client = TestClient(main.app)
OWNER = "job_retry_owner"


def _create_session(actor: str = OWNER) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_dataset(dataset_id: str, actor: str = OWNER) -> None:
    df = pd.DataFrame(
        [
            {"facility": "North", "month": "2025-01-01", "visits": 120},
            {"facility": "South", "month": "2025-02-01", "visits": 98},
        ]
    )
    files = {"file": ("job_retry.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_failed_facts_job_can_be_retried() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id)
    meta = main._load_meta(dataset_id)

    job = create_job(
        "facts",
        dataset_id,
        {
            "mode": "full",
            "seed": 42,
            "file_hash": meta.get("file_hash", ""),
            "schema_hash": meta.get("schema_hash", ""),
            "force": True,
        },
    )
    update_job(job["job_id"], status="failed", error="Synthetic failure for retry coverage.")

    response = client.post(
        f"/jobs/{job['job_id']}/retry",
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] != job["job_id"]
    assert payload["type"] == "facts"
    assert payload["dataset_id"] == dataset_id
    assert payload["status"] in {"queued", "running", "succeeded"}

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": OWNER})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "job_retried" in actions


def test_viewer_cannot_retry_failed_job() -> None:
    dataset_id = _create_session("job_retry_view_owner")
    _upload_dataset(dataset_id, actor="job_retry_view_owner")
    meta = main._load_meta(dataset_id)

    job = create_job(
        "facts",
        dataset_id,
        {
            "mode": "full",
            "seed": 7,
            "file_hash": meta.get("file_hash", ""),
            "schema_hash": meta.get("schema_hash", ""),
            "force": True,
        },
    )
    update_job(job["job_id"], status="failed", error="Synthetic failure for auth coverage.")

    response = client.post(
        f"/jobs/{job['job_id']}/retry",
        headers={"X-API-Key": "job_retry_view_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
