import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
OWNER = "evaluation_owner"
REVIEWER = "evaluation_reviewer"


def _create_session(actor: str = OWNER) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_dataset(dataset_id: str, actor: str = OWNER) -> None:
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "month": months.strftime("%Y-%m-%d"),
            "utilization": [100, 103, 106, 110, 115, 119, 122, 128, 132, 137, 141, 148],
            "facility": ["North"] * len(months),
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("evaluation.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def _train_run(dataset_id: str, horizon: int, name: str, actor: str = OWNER) -> str:
    response = client.post(
        f"/sessions/{dataset_id}/ml-runs/forecast",
        json={"time_field": "month", "metric_field": "utilization", "horizon": horizon, "name": name},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200
    return response.json()["run_id"]


def test_model_evaluation_compares_active_and_challenger_runs() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id)
    active_run_id = _train_run(dataset_id, 2, "Active forecast")
    challenger_run_id = _train_run(dataset_id, 3, "Challenger forecast")

    promoted = client.post(
        f"/sessions/{dataset_id}/ml-registry/promote/{active_run_id}",
        json={"note": "Approve active baseline model."},
        headers={"X-API-Key": REVIEWER, "X-User-Role": "reviewer"},
    )
    assert promoted.status_code == 200

    evaluation = client.get(
        f"/sessions/{dataset_id}/ml-evaluation",
        params={"challenger_run_id": challenger_run_id},
        headers={"X-API-Key": OWNER},
    )
    assert evaluation.status_code == 200
    payload = evaluation.json()["evaluation"]

    assert payload["active_run"]["run_id"] == active_run_id
    assert payload["challenger_run"]["run_id"] == challenger_run_id
    assert payload["winner"] in {"active", "challenger"}
    assert payload["rationale"]
    assert payload["suggested_actions"]

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": OWNER})
    assert detail.status_code == 200
    assert "ml_evaluation" in detail.json()["artifacts"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": OWNER})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "ml_models_evaluated" in actions


def test_viewer_cannot_run_model_evaluation() -> None:
    dataset_id = _create_session("evaluation_scope_owner")
    _upload_dataset(dataset_id, actor="evaluation_scope_owner")
    active_run_id = _train_run(dataset_id, 2, "Scope active", actor="evaluation_scope_owner")
    _train_run(dataset_id, 3, "Scope challenger", actor="evaluation_scope_owner")

    promoted = client.post(
        f"/sessions/{dataset_id}/ml-registry/promote/{active_run_id}",
        json={"note": "Approve baseline."},
        headers={"X-API-Key": "evaluation_reviewer", "X-User-Role": "reviewer"},
    )
    assert promoted.status_code == 200

    response = client.get(
        f"/sessions/{dataset_id}/ml-evaluation",
        headers={"X-API-Key": "evaluation_scope_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
    assert "compute" in response.json()["detail"]
