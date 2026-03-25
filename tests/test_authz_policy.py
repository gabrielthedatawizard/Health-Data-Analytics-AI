from uuid import uuid4

from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def _create_session(actor: str, display_name: str) -> str:
    response = client.post(
        "/sessions",
        json={"created_by": actor, "display_name": display_name},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200
    return response.json()["dataset_id"]


def test_auth_me_returns_inferred_and_explicit_role_context() -> None:
    inferred = client.get("/auth/me", headers={"X-API-Key": "manager_1"})
    assert inferred.status_code == 200
    assert inferred.json()["role"] == "reviewer"
    assert "sensitive_export:review" in inferred.json()["permissions"]

    explicit = client.get("/auth/me", headers={"X-API-Key": "manager_1", "X-User-Role": "viewer"})
    assert explicit.status_code == 200
    assert explicit.json()["role"] == "viewer"
    assert explicit.json()["permissions"] == ["sessions:read_own", "sessions:export_masked_own"]


def test_analyst_cannot_read_another_users_session() -> None:
    owner_a = f"owner_a_{uuid4().hex}"
    owner_b = f"owner_b_{uuid4().hex}"
    dataset_id_b = _create_session(owner_b, "Owner B Dataset")

    response = client.get(f"/sessions/{dataset_id_b}", headers={"X-API-Key": owner_a})
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]


def test_reviewer_can_list_and_read_all_sessions() -> None:
    owner_a = f"review_scope_a_{uuid4().hex}"
    owner_b = f"review_scope_b_{uuid4().hex}"
    dataset_id_a = _create_session(owner_a, "Reviewer Scope A")
    dataset_id_b = _create_session(owner_b, "Reviewer Scope B")

    response = client.get("/sessions", headers={"X-API-Key": "manager_1"})
    assert response.status_code == 200
    session_ids = {item["dataset_id"] for item in response.json()["sessions"]}
    assert dataset_id_a in session_ids
    assert dataset_id_b in session_ids

    detail = client.get(f"/sessions/{dataset_id_b}", headers={"X-API-Key": "manager_1"})
    assert detail.status_code == 200
    assert detail.json()["dataset_id"] == dataset_id_b


def test_non_reviewer_cannot_decide_sensitive_export() -> None:
    owner = f"sensitive_owner_{uuid4().hex}"
    dataset_id = _create_session(owner, "Sensitive Review Dataset")

    requested = client.post(
        f"/sessions/{dataset_id}/sensitive-export/request",
        json={"justification": "Need supervised export for governed case review."},
        headers={"X-API-Key": owner},
    )
    assert requested.status_code == 200

    denied = client.post(
        f"/sessions/{dataset_id}/sensitive-export/decision",
        json={"approved": True, "note": "Analyst should not be able to approve this."},
        headers={"X-API-Key": "analyst_peer"},
    )
    assert denied.status_code == 403
    assert "review sensitive export" in denied.json()["detail"]
