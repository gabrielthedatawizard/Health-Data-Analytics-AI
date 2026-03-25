import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
ACTOR = "drift_analyst"


def _create_session(actor: str = ACTOR) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_dataset(dataset_id: str, values: list[int], actor: str = ACTOR) -> None:
    months = pd.date_range("2025-01-01", periods=len(values), freq="MS")
    df = pd.DataFrame(
        {
            "month": months.strftime("%Y-%m-%d"),
            "visits": values,
            "facility": ["North"] * len(values),
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("drift.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_forecast_drift_flags_stale_model_after_data_refresh() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id, [100, 104, 108, 112, 116, 120, 124, 128, 132, 136])

    trained = client.post(
        f"/sessions/{dataset_id}/ml-runs/forecast",
        json={"time_field": "month", "metric_field": "visits", "horizon": 3},
        headers={"X-API-Key": ACTOR},
    )
    assert trained.status_code == 200

    _upload_dataset(dataset_id, [100, 104, 108, 112, 116, 120, 124, 128, 220, 260, 310, 340])

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": ACTOR})
    assert detail.status_code == 200
    assert "ml_runs" in detail.json()["artifacts"]

    drift = client.get(f"/sessions/{dataset_id}/ml-runs/drift", headers={"X-API-Key": ACTOR})
    assert drift.status_code == 200
    payload = drift.json()["drift"]

    assert payload["stale_model"] is True
    assert payload["training_data_hash"] != payload["current_data_hash"]
    assert payload["drift_score"] >= 0.25
    signal_codes = {signal["code"] for signal in payload["signals"]}
    assert "data_refresh" in signal_codes

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": ACTOR})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "ml_drift_scanned" in actions


def test_viewer_cannot_scan_ml_drift() -> None:
    dataset_id = _create_session("drift_owner")
    _upload_dataset(dataset_id, [90, 92, 95, 97, 99, 101, 103, 106], actor="drift_owner")

    trained = client.post(
        f"/sessions/{dataset_id}/ml-runs/forecast",
        json={"time_field": "month", "metric_field": "visits", "horizon": 2},
        headers={"X-API-Key": "drift_owner"},
    )
    assert trained.status_code == 200

    response = client.get(
        f"/sessions/{dataset_id}/ml-runs/drift",
        headers={"X-API-Key": "drift_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
    assert "compute" in response.json()["detail"]
