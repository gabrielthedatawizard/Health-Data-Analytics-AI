import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
ACTOR = "forecast_analyst"


def _create_session(actor: str = ACTOR) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_forecast_dataset(dataset_id: str, actor: str = ACTOR) -> None:
    months = pd.date_range("2025-01-01", periods=10, freq="MS")
    df = pd.DataFrame(
        {
            "month": months.strftime("%Y-%m-%d"),
            "admissions": [120, 125, 129, 134, 141, 146, 154, 162, 171, 183],
            "facility": ["North"] * len(months),
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("forecast.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_forecast_training_persists_governed_model_run() -> None:
    dataset_id = _create_session()
    _upload_forecast_dataset(dataset_id)

    response = client.post(
        f"/sessions/{dataset_id}/ml-runs/forecast",
        json={
            "name": "Admissions baseline forecast",
            "time_field": "month",
            "metric_field": "admissions",
            "horizon": 3,
            "aggregation": "sum",
        },
        headers={"X-API-Key": ACTOR},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["model_kind"] == "forecast"
    assert payload["status"] == "succeeded"
    assert payload["payload"]["metric_field"] == "admissions"
    assert payload["payload"]["time_field"] == "month"
    assert len(payload["payload"]["candidate_models"]) >= 3
    assert len(payload["payload"]["forecast"]) == 3

    listed = client.get(f"/sessions/{dataset_id}/ml-runs", headers={"X-API-Key": ACTOR})
    assert listed.status_code == 200
    assert listed.json()["runs"][0]["run_id"] == payload["run_id"]

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": ACTOR})
    assert detail.status_code == 200
    assert "ml_runs" in detail.json()["artifacts"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": ACTOR})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "ml_forecast_trained" in actions


def test_viewer_cannot_train_forecast_run() -> None:
    dataset_id = _create_session("forecast_owner")
    _upload_forecast_dataset(dataset_id, actor="forecast_owner")

    response = client.post(
        f"/sessions/{dataset_id}/ml-runs/forecast",
        json={
            "time_field": "month",
            "metric_field": "admissions",
            "horizon": 2,
        },
        headers={"X-API-Key": "forecast_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
    assert "compute" in response.json()["detail"]
