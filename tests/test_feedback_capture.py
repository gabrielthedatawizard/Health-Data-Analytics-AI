import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
OWNER = "feedback_owner"
PEER = "feedback_peer"


def _create_session(actor: str = OWNER) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_dataset(dataset_id: str, actor: str = OWNER) -> None:
    df = pd.DataFrame(
        [
            {"facility": "North", "month": "2025-01-01", "visits": 120},
            {"facility": "North", "month": "2025-02-01", "visits": 145},
            {"facility": "South", "month": "2025-01-01", "visits": 90},
        ]
    )
    files = {"file": ("feedback.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_feedback_records_are_saved_and_audited() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id)

    submit = client.post(
        f"/sessions/{dataset_id}/feedback",
        json={
            "surface": "ask_data",
            "target_id": "current_ask",
            "rating": "positive",
            "question": "Show the main metric trend over time",
            "title": "Ask Your Data response",
        },
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert submit.status_code == 200
    payload = submit.json()
    assert payload["surface"] == "ask_data"
    assert payload["rating"] == "positive"

    listing = client.get(
        f"/sessions/{dataset_id}/feedback",
        headers={"X-API-Key": OWNER, "X-User-Role": "analyst"},
    )
    assert listing.status_code == 200
    assert listing.json()["feedback"][0]["feedback_id"] == payload["feedback_id"]

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": OWNER})
    assert detail.status_code == 200
    assert "feedback" in detail.json()["artifacts"]

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": OWNER})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "feedback_recorded" in actions


def test_other_analyst_cannot_record_feedback_on_unowned_session() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id)

    response = client.post(
        f"/sessions/{dataset_id}/feedback",
        json={"surface": "ask_data", "target_id": "current_ask", "rating": "negative"},
        headers={"X-API-Key": PEER, "X-User-Role": "analyst"},
    )
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]
