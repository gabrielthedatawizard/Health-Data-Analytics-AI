import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
OWNER = "registry_owner"
REVIEWER = "registry_reviewer"


def _create_session(actor: str = OWNER) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_dataset(dataset_id: str, values: list[int], actor: str = OWNER) -> None:
    months = pd.date_range("2025-01-01", periods=len(values), freq="MS")
    df = pd.DataFrame(
        {
            "month": months.strftime("%Y-%m-%d"),
            "claims": values,
            "facility": ["North"] * len(values),
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("registry.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def _train_forecast(dataset_id: str, actor: str = OWNER) -> str:
    response = client.post(
        f"/sessions/{dataset_id}/ml-runs/forecast",
        json={"time_field": "month", "metric_field": "claims", "horizon": 3},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200
    return response.json()["run_id"]


def test_reviewer_can_promote_forecast_run_to_registry() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id, [100, 105, 109, 113, 118, 124, 129, 135, 142, 150])
    run_id = _train_forecast(dataset_id)

    promoted = client.post(
        f"/sessions/{dataset_id}/ml-registry/promote/{run_id}",
        json={"note": "Reviewed and approved for governed production use."},
        headers={"X-API-Key": REVIEWER, "X-User-Role": "reviewer"},
    )
    assert promoted.status_code == 200
    payload = promoted.json()
    assert payload["status"] == "active"
    assert payload["run_id"] == run_id
    assert payload["promoted_by"] == REVIEWER

    registry = client.get(f"/sessions/{dataset_id}/ml-registry", headers={"X-API-Key": OWNER})
    assert registry.status_code == 200
    assert registry.json()["entries"][0]["registry_id"] == payload["registry_id"]

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": OWNER})
    assert detail.status_code == 200
    assert "ml_registry" in detail.json()["artifacts"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": OWNER})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "ml_model_promoted" in actions


def test_stale_run_cannot_be_promoted_after_refresh() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id, [100, 105, 109, 113, 118, 124, 129, 135, 142, 150])
    run_id = _train_forecast(dataset_id)

    _upload_dataset(dataset_id, [100, 105, 109, 113, 118, 124, 129, 135, 220, 260, 310, 350])

    promoted = client.post(
        f"/sessions/{dataset_id}/ml-registry/promote/{run_id}",
        json={"note": "Attempting to promote a stale run."},
        headers={"X-API-Key": REVIEWER, "X-User-Role": "reviewer"},
    )
    assert promoted.status_code == 400
    assert "Stale forecast runs cannot be promoted" in promoted.json()["detail"]
