import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
ACTOR = "anomaly_analyst"


def _create_session(actor: str = ACTOR) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_anomaly_dataset(dataset_id: str, actor: str = ACTOR) -> None:
    rows: list[dict[str, object]] = []
    months = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01"]
    visit_map = {
        "2025-01-01": [100, 105, 98],
        "2025-02-01": [102, 103, 97],
        "2025-03-01": [220, 260, 210],
        "2025-04-01": [101, 104, 99],
        "2025-05-01": [99, 102, 100],
        "2025-06-01": [103, 101, 98],
    }
    cost_map = {
        "North": 110,
        "South": 320,
        "West": 105,
    }

    for month in months:
        for facility, visits in zip(["North", "South", "West"], visit_map[month], strict=True):
            rows.append(
                {
                    "month": month,
                    "facility": facility,
                    "visits": visits,
                    "cost": cost_map[facility],
                    "notes": "" if facility == "South" or month in {"2025-03-01", "2025-05-01"} else "complete",
                }
            )

    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("anomaly.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_anomaly_analysis_surfaces_governed_signals() -> None:
    dataset_id = _create_session()
    _upload_anomaly_dataset(dataset_id)

    response = client.get(f"/sessions/{dataset_id}/anomalies", headers={"X-API-Key": ACTOR})
    assert response.status_code == 200
    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["analysis"]["anomaly_count"] >= 2
    kinds = {item["kind"] for item in payload["analysis"]["anomalies"]}
    assert "time_spike" in kinds
    assert "quality" in kinds or "distribution" in kinds

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": ACTOR})
    assert detail.status_code == 200
    assert "anomaly_analysis" in detail.json()["artifacts"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": ACTOR})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "anomaly_analysis_generated" in actions


def test_viewer_cannot_run_anomaly_analysis() -> None:
    dataset_id = _create_session("anomaly_scope_owner")
    _upload_anomaly_dataset(dataset_id, actor="anomaly_scope_owner")

    response = client.get(
        f"/sessions/{dataset_id}/anomalies",
        headers={"X-API-Key": "anomaly_scope_owner", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
    assert "compute" in response.json()["detail"]
