import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
OWNER = "analyst_owner"


def _create_pii_session() -> str:
    session = client.post("/sessions", json={"created_by": OWNER}, headers={"X-API-Key": OWNER})
    assert session.status_code == 200
    dataset_id = session.json()["dataset_id"]

    df = pd.DataFrame(
        {
            "patient_name": ["A", "B", "C"],
            "visit_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "opd_visits": [10, 11, 9],
        }
    )
    files = {"file": ("pii.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
    upload = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": OWNER},
        headers={"X-API-Key": OWNER},
    )
    assert upload.status_code == 200
    return dataset_id


def test_csv_export_masks_pii_by_default() -> None:
    dataset_id = _create_pii_session()

    exported = client.get(f"/sessions/{dataset_id}/export/csv", headers={"X-API-Key": OWNER})
    assert exported.status_code == 200
    assert "patient_name" not in exported.text
    assert "pii_field_1" in exported.text


def test_sensitive_export_requires_request_and_approval() -> None:
    dataset_id = _create_pii_session()

    direct_enable = client.post(
        f"/sessions/{dataset_id}/sensitive-export",
        json={"enabled": True},
        headers={"X-API-Key": OWNER},
    )
    assert direct_enable.status_code == 409

    requested = client.post(
        f"/sessions/{dataset_id}/sensitive-export/request",
        json={"justification": "Need reviewed patient outreach list for case investigation."},
        headers={"X-API-Key": OWNER},
    )
    assert requested.status_code == 200
    assert requested.json()["sensitive_export_approval"]["status"] == "pending"

    approved = client.post(
        f"/sessions/{dataset_id}/sensitive-export/decision",
        json={"approved": True, "note": "Approved for supervised operational investigation."},
        headers={"X-API-Key": "manager_1"},
    )
    assert approved.status_code == 200
    assert approved.json()["allow_sensitive_export"] is True
    assert approved.json()["sensitive_export_approval"]["status"] == "approved"

    exported = client.get(f"/sessions/{dataset_id}/export/csv", headers={"X-API-Key": OWNER})
    assert exported.status_code == 200
    assert "patient_name" in exported.text


def test_sensitive_export_rejection_keeps_masking() -> None:
    dataset_id = _create_pii_session()

    requested = client.post(
        f"/sessions/{dataset_id}/sensitive-export/request",
        json={"justification": "Need unmasked data for temporary quality validation."},
        headers={"X-API-Key": OWNER},
    )
    assert requested.status_code == 200

    rejected = client.post(
        f"/sessions/{dataset_id}/sensitive-export/decision",
        json={"approved": False, "note": "Use masked export until approval is granted."},
        headers={"X-API-Key": "manager_1"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["allow_sensitive_export"] is False
    assert rejected.json()["sensitive_export_approval"]["status"] == "rejected"

    exported = client.get(f"/sessions/{dataset_id}/export/csv", headers={"X-API-Key": OWNER})
    assert exported.status_code == 200
    assert "patient_name" not in exported.text
    assert "pii_field_1" in exported.text
