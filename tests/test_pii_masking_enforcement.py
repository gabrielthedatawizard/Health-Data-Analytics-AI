import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def test_csv_export_masks_pii_by_default() -> None:
    session = client.post("/sessions", json={"created_by": "tester"}, headers={"X-API-Key": "user_1"})
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
    upload = client.post(f"/sessions/{dataset_id}/upload", files=files, data={"uploaded_by": "tester"})
    assert upload.status_code == 200

    profile = client.get(f"/sessions/{dataset_id}/profile")
    assert profile.status_code == 200

    exported = client.get(f"/sessions/{dataset_id}/export/csv")
    assert exported.status_code == 200
    assert "patient_name" not in exported.text
    assert "pii_field_1" in exported.text
