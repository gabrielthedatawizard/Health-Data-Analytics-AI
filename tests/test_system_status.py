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
            {"patient_name": "Asha", "facility": "North", "month": "2025-01-01", "visits": 120},
            {"patient_name": "Bakari", "facility": "North", "month": "2025-02-01", "visits": 135},
            {"patient_name": "Neema", "facility": "South", "month": "2025-03-01", "visits": 128},
        ]
    )
    files = {"file": ("status.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
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


def test_reviewer_system_status_surfaces_live_governance_counts() -> None:
    owner = f"status_owner_{uuid4().hex}"
    dataset_id = _create_session(owner)
    _upload_dataset(dataset_id, owner)

    export_request = client.post(
        f"/sessions/{dataset_id}/sensitive-export/request",
        json={"justification": "Need governed review for export."},
        headers={"X-API-Key": owner, "X-User-Role": "analyst"},
    )
    assert export_request.status_code == 200

    draft_workflow = client.post(
        f"/sessions/{dataset_id}/workflow-actions/draft",
        json={"action_type": "create_ticket", "objective": "Investigate visit spike."},
        headers={"X-API-Key": owner, "X-User-Role": "analyst"},
    )
    assert draft_workflow.status_code == 200

    original = _upload_document(
        owner,
        "policy_v1.md",
        "Original denominator policy requires supervisor review.",
        {"title": "Denominator Policy", "version_label": "v1", "effective_date": "2025-01-01"},
    )
    _upload_document(
        owner,
        "policy_v2.md",
        "Updated denominator policy requires clinical lead review.",
        {
            "title": "Denominator Policy",
            "version_label": "v2",
            "effective_date": "2026-01-01",
            "supersedes_document_id": original["document_id"],
        },
    )

    response = client.get("/system/status", headers={"X-API-Key": "manager_1", "X-User-Role": "reviewer"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["counts"]["visible_sessions"] >= 1
    assert payload["counts"]["visible_documents"] >= 2
    assert payload["counts"]["pending_sensitive_exports"] >= 1
    assert payload["counts"]["pending_workflow_reviews"] >= 1
    assert payload["counts"]["superseded_documents"] >= 1
    assert any(alert["level"] == "warning" for alert in payload["alerts"])


def test_viewer_system_status_stays_scoped_to_owned_workspace() -> None:
    owner = f"status_scope_owner_{uuid4().hex}"
    outsider = f"status_scope_other_{uuid4().hex}"

    owned_dataset_id = _create_session(owner)
    _upload_dataset(owned_dataset_id, owner)

    other_dataset_id = _create_session(outsider)
    _upload_dataset(other_dataset_id, outsider)
    _upload_document(outsider, "other.md", "Other user's private trusted policy.")

    response = client.get("/system/status", headers={"X-API-Key": owner, "X-User-Role": "viewer"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "viewer"
    assert payload["counts"]["visible_sessions"] == 1
    assert payload["counts"]["visible_documents"] == 0
