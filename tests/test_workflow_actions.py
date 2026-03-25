import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
OWNER = "workflow_owner"
REVIEWER = "workflow_reviewer"


def _create_session(actor: str = OWNER) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_workflow_dataset(dataset_id: str, actor: str = OWNER) -> None:
    df = pd.DataFrame(
        [
            {"facility": "North", "month": "2025-01-01", "visits": 120, "notes": "steady"},
            {"facility": "North", "month": "2025-02-01", "visits": 142, "notes": "spike"},
            {"facility": "South", "month": "2025-01-01", "visits": 98, "notes": "steady"},
            {"facility": "South", "month": "2025-02-01", "visits": 101, "notes": "steady"},
        ]
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("workflow.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_workflow_actions_require_review_before_execution() -> None:
    dataset_id = _create_session()
    _upload_workflow_dataset(dataset_id)

    draft = client.post(
        f"/sessions/{dataset_id}/workflow-actions/draft",
        json={
            "action_type": "create_ticket",
            "target": "Quality operations",
            "objective": "Investigate the spike in North facility visits.",
        },
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert draft.status_code == 200
    draft_payload = draft.json()
    assert draft_payload["status"] == "pending_approval"
    action_id = draft_payload["action_id"]

    premature_execute = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{action_id}/execute",
        json={},
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert premature_execute.status_code == 400
    assert "approved" in premature_execute.json()["detail"]

    review = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{action_id}/decision",
        json={"approved": True, "note": "Approved for operational handoff."},
        headers={"X-API-Key": REVIEWER, "X-User-Role": "reviewer"},
    )
    assert review.status_code == 200
    assert review.json()["status"] == "approved"

    execute = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{action_id}/execute",
        json={"note": "Logged as a governed manual workflow action."},
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert execute.status_code == 200
    assert execute.json()["status"] == "executed"
    assert execute.json()["payload"]["execution_mode"] == "governed_manual"

    listing = client.get(
        f"/sessions/{dataset_id}/workflow-actions",
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert listing.status_code == 200
    assert listing.json()["actions"][0]["action_id"] == action_id

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": OWNER})
    assert detail.status_code == 200
    assert "workflow_actions" in detail.json()["artifacts"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": OWNER})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "workflow_action_drafted" in actions
    assert "workflow_action_reviewed" in actions
    assert "workflow_action_executed" in actions


def test_viewer_cannot_draft_workflow_action() -> None:
    dataset_id = _create_session("workflow_view_owner")
    _upload_workflow_dataset(dataset_id, actor="workflow_view_owner")

    response = client.post(
        f"/sessions/{dataset_id}/workflow-actions/draft",
        json={"action_type": "draft_email"},
        headers={"X-API-Key": "workflow_view_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
    assert "draft workflow" in response.json()["detail"]


def test_analyst_cannot_review_workflow_action() -> None:
    dataset_id = _create_session("workflow_review_owner")
    _upload_workflow_dataset(dataset_id, actor="workflow_review_owner")

    draft = client.post(
        f"/sessions/{dataset_id}/workflow-actions/draft",
        json={"action_type": "action_plan"},
        headers={"X-API-Key": "workflow_review_owner", "X-User-Role": "analyst"},
    )
    assert draft.status_code == 200
    action_id = draft.json()["action_id"]

    response = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{action_id}/decision",
        json={"approved": True},
        headers={"X-API-Key": "workflow_review_owner", "X-User-Role": "analyst"},
    )
    assert response.status_code == 403
    assert "review workflow" in response.json()["detail"]
