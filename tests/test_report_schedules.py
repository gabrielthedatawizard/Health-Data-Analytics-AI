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
    assert schedule_payload["due_now"] is True
    assert schedule_payload["next_due_at"]

    run_response = client.post(
        f"/sessions/{dataset_id}/report-schedules/{payload['schedule_id']}/run",
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert run_response.status_code == 200
    assert run_response.json()["schedule_id"] == payload["schedule_id"]
    assert run_response.json()["job_id"]

    refreshed_schedules = client.get(
        f"/sessions/{dataset_id}/report-schedules",
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert refreshed_schedules.status_code == 200
    refreshed_payload = refreshed_schedules.json()["schedules"][0]
    assert refreshed_payload["due_now"] is False
    assert refreshed_payload["last_run_at"]
    assert refreshed_payload["next_due_at"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": OWNER})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "report_schedule_created" in actions
    assert "report_schedule_run" in actions


def test_run_due_report_schedules_only_triggers_due_items() -> None:
    dataset_id = _create_session("report_schedule_due_owner")
    _upload_dataset(dataset_id, actor="report_schedule_due_owner")

    def create_schedule(title: str) -> str:
        draft = client.post(
            f"/sessions/{dataset_id}/workflow-actions/draft",
            json={
                "action_type": "schedule_report",
                "title": title,
                "objective": "Send a governed recurring summary.",
            },
            headers={"X-API-Key": "report_schedule_due_owner", "X-User-Role": "analyst"},
        )
        assert draft.status_code == 200
        action_id = draft.json()["action_id"]

        review = client.post(
            f"/sessions/{dataset_id}/workflow-actions/{action_id}/decision",
            json={"approved": True},
            headers={"X-API-Key": REVIEWER, "X-User-Role": "reviewer"},
        )
        assert review.status_code == 200

        execute = client.post(
            f"/sessions/{dataset_id}/workflow-actions/{action_id}/execute",
            json={},
            headers={"X-API-Key": "report_schedule_due_owner", "X-User-Role": "analyst"},
        )
        assert execute.status_code == 200
        return execute.json()["payload"]["schedule_id"]

    first_schedule_id = create_schedule("Monthly governed summary")
    first_run = client.post(
        f"/sessions/{dataset_id}/report-schedules/{first_schedule_id}/run",
        headers={"X-API-Key": "report_schedule_due_owner", "X-User-Role": "analyst"},
    )
    assert first_run.status_code == 200

    second_schedule_id = create_schedule("Quarterly governance summary")

    run_due = client.post(
        f"/sessions/{dataset_id}/report-schedules/run-due",
        headers={"X-API-Key": "report_schedule_due_owner", "X-User-Role": "analyst"},
    )
    assert run_due.status_code == 200
    payload = run_due.json()
    assert payload["triggered_count"] == 1
    assert payload["runs"][0]["schedule_id"] == second_schedule_id

    schedules = client.get(
        f"/sessions/{dataset_id}/report-schedules",
        headers={"X-API-Key": "report_schedule_due_owner", "X-User-Role": "analyst"},
    )
    assert schedules.status_code == 200
    schedule_map = {item["schedule_id"]: item for item in schedules.json()["schedules"]}
    assert schedule_map[first_schedule_id]["due_now"] is False
    assert schedule_map[second_schedule_id]["due_now"] is False


def test_viewer_cannot_run_report_schedule() -> None:
    dataset_id = _create_session("report_schedule_view_owner")
    _upload_dataset(dataset_id, actor="report_schedule_view_owner")

    response = client.post(
        f"/sessions/{dataset_id}/report-schedules/not-a-real-id/run",
        headers={"X-API-Key": "report_schedule_view_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
