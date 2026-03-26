import io
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def _create_session(actor: str) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_dataset(dataset_id: str, actor: str) -> None:
    df = pd.DataFrame(
        [
            {"facility": "North", "month": "2025-01-01", "visits": 120, "notes": "steady"},
            {"facility": "North", "month": "2025-02-01", "visits": 142, "notes": "spike"},
            {"facility": "South", "month": "2025-01-01", "visits": 98, "notes": "steady"},
        ]
    )
    files = {"file": ("review_queue.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def _upload_document(actor: str, filename: str, content: str, data: dict[str, str] | None = None) -> dict:
    response = client.post(
        "/documents",
        files={"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")},
        data=data or {},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200
    return response.json()


def test_owner_review_queue_surfaces_pending_actions_and_schedule_artifacts() -> None:
    owner = f"queue_owner_{uuid4().hex}"
    reviewer = f"queue_reviewer_{uuid4().hex}"
    dataset_id = _create_session(owner)
    _upload_dataset(dataset_id, owner)

    export_request = client.post(
        f"/sessions/{dataset_id}/sensitive-export/request",
        json={"justification": "Need reviewer sign-off for governed export."},
        headers={"X-API-Key": owner, "X-User-Role": "analyst"},
    )
    assert export_request.status_code == 200

    pending_workflow = client.post(
        f"/sessions/{dataset_id}/workflow-actions/draft",
        json={"action_type": "create_ticket", "objective": "Investigate facility spike."},
        headers={"X-API-Key": owner, "X-User-Role": "analyst"},
    )
    assert pending_workflow.status_code == 200

    schedule_workflow = client.post(
        f"/sessions/{dataset_id}/workflow-actions/draft",
        json={"action_type": "schedule_report", "objective": "Queue monthly quality review."},
        headers={"X-API-Key": owner, "X-User-Role": "analyst"},
    )
    assert schedule_workflow.status_code == 200
    schedule_action_id = schedule_workflow.json()["action_id"]

    review_schedule = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{schedule_action_id}/decision",
        json={"approved": True, "note": "Approved for governed scheduling."},
        headers={"X-API-Key": reviewer, "X-User-Role": "reviewer"},
    )
    assert review_schedule.status_code == 200

    execute_schedule = client.post(
        f"/sessions/{dataset_id}/workflow-actions/{schedule_action_id}/execute",
        json={"note": "Create the durable schedule artifact."},
        headers={"X-API-Key": owner, "X-User-Role": "analyst"},
    )
    assert execute_schedule.status_code == 200
    assert execute_schedule.json()["payload"]["schedule_id"]

    original = _upload_document(
        owner,
        "measure_policy_v1.md",
        "Original measure guidance for queue test.",
        {"title": "Measure Guidance", "version_label": "v1", "effective_date": "2025-01-01"},
    )
    _upload_document(
        owner,
        "measure_policy_v2.md",
        "Updated measure guidance for queue test.",
        {
            "title": "Measure Guidance",
            "version_label": "v2",
            "effective_date": "2026-01-01",
            "supersedes_document_id": original["document_id"],
        },
    )

    response = client.get("/system/review-queue", headers={"X-API-Key": owner, "X-User-Role": "analyst"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_items"] >= 4
    categories = {item["category"] for item in payload["items"]}
    assert "sensitive_export" in categories
    assert "workflow_review" in categories
    assert "report_schedule" in categories
    assert "document_freshness" in categories
    assert any(item["dataset_id"] == dataset_id for item in payload["items"] if item["category"] != "document_freshness")


def test_viewer_review_queue_stays_scoped_to_owned_dataset() -> None:
    owner = f"queue_scope_owner_{uuid4().hex}"
    outsider = f"queue_scope_outsider_{uuid4().hex}"

    owned_dataset_id = _create_session(owner)
    _upload_dataset(owned_dataset_id, owner)
    outsider_dataset_id = _create_session(outsider)
    _upload_dataset(outsider_dataset_id, outsider)

    own_export_request = client.post(
        f"/sessions/{owned_dataset_id}/sensitive-export/request",
        json={"justification": "Owner export request."},
        headers={"X-API-Key": owner, "X-User-Role": "analyst"},
    )
    assert own_export_request.status_code == 200

    outsider_export_request = client.post(
        f"/sessions/{outsider_dataset_id}/sensitive-export/request",
        json={"justification": "Outsider export request."},
        headers={"X-API-Key": outsider, "X-User-Role": "analyst"},
    )
    assert outsider_export_request.status_code == 200

    response = client.get("/system/review-queue", headers={"X-API-Key": owner, "X-User-Role": "viewer"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "viewer"
    visible_dataset_ids = {item["dataset_id"] for item in payload["items"] if item["dataset_id"] not in {"documents", "system"}}
    assert visible_dataset_ids == {owned_dataset_id}
    assert all(outsider_dataset_id != item["dataset_id"] for item in payload["items"])
