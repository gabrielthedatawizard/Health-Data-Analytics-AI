import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
OWNER = "report_schedule_owner"
REVIEWER = "report_schedule_reviewer"


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
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("report_schedule.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_schedule_report_workflow_creates_and_runs_report_schedule() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id)

    draft = client.post(
        f"/sessions/{dataset_id}/workflow-actions/draft",
        json={
            "action_type": "schedule_report",
            "target": "Quality committee",
            "objective": "Send a recurring governed monthly summary.",
        },
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert draft.status_code == 200
    action_id = draft.json()["action_id"]

    review = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{action_id}/decision",
        json={"approved": True, "note": "Approved for governed reporting cadence."},
        headers={"X-API-Key": REVIEWER, "X-User-Role": "reviewer"},
    )
    assert review.status_code == 200

    execute = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{action_id}/execute",
        json={"note": "Create the report schedule artifact."},
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert execute.status_code == 200
    payload = execute.json()["payload"]
    assert payload["schedule_id"]
    assert payload["schedule_status"] == "active"

    schedules = client.get(
        f"/sessions/{dataset_id}/report-schedules",
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert schedules.status_code == 200
    schedule_payload = schedules.json()["schedules"][0]
    assert schedule_payload["schedule_id"] == payload["schedule_id"]
    assert schedule_payload["report_template"] == "health_report"

    run_response = client.post(
        f"/sessions/{dataset_id}/report-schedules/{payload['schedule_id']}/run",
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert run_response.status_code == 200
    assert run_response.json()["schedule_id"] == payload["schedule_id"]
    assert run_response.json()["job_id"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": OWNER})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "report_schedule_created" in actions
    assert "report_schedule_run" in actions


def test_viewer_cannot_run_report_schedule() -> None:
    dataset_id = _create_session("report_schedule_view_owner")
    _upload_dataset(dataset_id, actor="report_schedule_view_owner")

    response = client.post(
        f"/sessions/{dataset_id}/report-schedules/not-a-real-id/run",
        headers={"X-API-Key": "report_schedule_view_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
