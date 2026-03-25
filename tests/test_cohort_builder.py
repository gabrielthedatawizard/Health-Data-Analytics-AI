import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
ACTOR = "cohort_analyst"


def _create_session(actor: str = ACTOR) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_cohort_dataset(dataset_id: str, actor: str = ACTOR) -> None:
    df = pd.DataFrame(
        [
            {
                "patient_id": "PAT-001",
                "facility": "North",
                "sex": "Female",
                "diagnosis": "CHF",
                "age": 61,
                "follow_up_days": 45,
                "visit_month": "2025-01-01",
            },
            {
                "patient_id": "PAT-002",
                "facility": "North",
                "sex": "Male",
                "diagnosis": "COPD",
                "age": 55,
                "follow_up_days": 14,
                "visit_month": "2025-01-01",
            },
            {
                "patient_id": "PAT-003",
                "facility": "South",
                "sex": "Female",
                "diagnosis": "CHF",
                "age": 49,
                "follow_up_days": 37,
                "visit_month": "2025-02-01",
            },
            {
                "patient_id": "PAT-004",
                "facility": "East",
                "sex": "Female",
                "diagnosis": "Diabetes",
                "age": 42,
                "follow_up_days": 12,
                "visit_month": "2025-02-01",
            },
            {
                "patient_id": "PAT-005",
                "facility": "South",
                "sex": "Male",
                "diagnosis": "CHF",
                "age": 67,
                "follow_up_days": 52,
                "visit_month": "2025-03-01",
            },
        ]
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("cohort.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_build_cohort_analysis_persists_governed_artifact() -> None:
    dataset_id = _create_session()
    _upload_cohort_dataset(dataset_id)

    response = client.post(
        f"/sessions/{dataset_id}/cohorts",
        json={
            "name": "Female follow-up cohort",
            "criteria": [
                {"field": "sex", "operator": "eq", "value": "Female"},
                {"field": "follow_up_days", "operator": "gt", "value": 30},
            ],
            "limit": 10,
        },
        headers={"X-API-Key": ACTOR},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["cohort"]["row_count"] == 2
    assert payload["cohort"]["criteria_count"] == 2
    assert "patient_id" not in payload["cohort"]["preview_columns"]
    assert payload["cohort"]["preview_rows"]

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": ACTOR})
    assert detail.status_code == 200
    assert "cohort_analysis" in detail.json()["artifacts"]

    saved = client.get(f"/sessions/{dataset_id}/cohorts", headers={"X-API-Key": ACTOR})
    assert saved.status_code == 200
    assert saved.json()["cohort"]["row_count"] == 2

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": ACTOR})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "cohort_analysis_generated" in actions


def test_cohort_builder_blocks_governance_restricted_fields() -> None:
    dataset_id = _create_session()
    _upload_cohort_dataset(dataset_id)

    response = client.post(
        f"/sessions/{dataset_id}/cohorts",
        json={
            "criteria": [
                {"field": "patient_id", "operator": "eq", "value": "PAT-001"},
            ]
        },
        headers={"X-API-Key": ACTOR},
    )
    assert response.status_code == 400
    assert "blocked by governance controls" in response.json()["detail"]


def test_viewer_cannot_build_cohort() -> None:
    dataset_id = _create_session("cohort_owner")
    _upload_cohort_dataset(dataset_id, actor="cohort_owner")

    response = client.post(
        f"/sessions/{dataset_id}/cohorts",
        json={
            "criteria": [
                {"field": "facility", "operator": "eq", "value": "North"},
            ]
        },
        headers={"X-API-Key": "cohort_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
    assert "compute" in response.json()["detail"]
