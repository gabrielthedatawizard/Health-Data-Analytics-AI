import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)
ACTOR = "investigation_analyst"


def _create_session(actor: str = ACTOR) -> str:
    response = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def _upload_dataset(dataset_id: str, actor: str = ACTOR) -> None:
    df = pd.DataFrame(
        [
            {"facility": "North", "month": "2025-01-01", "visits": 120},
            {"facility": "South", "month": "2025-02-01", "visits": 98},
        ]
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("investigation.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post(
        f"/sessions/{dataset_id}/upload",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert response.status_code == 200


def test_saved_investigations_and_playbooks_persist_for_session() -> None:
    dataset_id = _create_session()
    _upload_dataset(dataset_id)

    saved_investigation = client.post(
        f"/sessions/{dataset_id}/investigations",
        json={
            "title": "North facility trend check",
            "question": "Why did North facility visits rise in February?",
            "context_type": "ask",
            "note": "Review before the committee meeting.",
            "result": {"answer": "Visits increased after outreach expansion.", "facts_used": ["fact_001"]},
        },
        headers={"X-API-Key": ACTOR},
    )
    assert saved_investigation.status_code == 200
    investigation_payload = saved_investigation.json()
    assert investigation_payload["title"] == "North facility trend check"

    saved_playbook = client.post(
        f"/sessions/{dataset_id}/playbooks",
        json={
            "name": "Facility trend review",
            "question_template": "Compare visit trends by facility over the last 6 months.",
            "description": "Reusable committee prep question.",
            "context_type": "ask",
        },
        headers={"X-API-Key": ACTOR},
    )
    assert saved_playbook.status_code == 200
    playbook_payload = saved_playbook.json()
    assert playbook_payload["name"] == "Facility trend review"

    listed_investigations = client.get(f"/sessions/{dataset_id}/investigations", headers={"X-API-Key": ACTOR})
    assert listed_investigations.status_code == 200
    assert listed_investigations.json()["investigations"][0]["investigation_id"] == investigation_payload["investigation_id"]

    listed_playbooks = client.get(f"/sessions/{dataset_id}/playbooks", headers={"X-API-Key": ACTOR})
    assert listed_playbooks.status_code == 200
    assert listed_playbooks.json()["playbooks"][0]["playbook_id"] == playbook_payload["playbook_id"]

    detail = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": ACTOR})
    assert detail.status_code == 200
    artifacts = detail.json()["artifacts"]
    assert "investigations" in artifacts
    assert "playbooks" in artifacts

    audit = client.get(f"/sessions/{dataset_id}/audit", headers={"X-API-Key": ACTOR})
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "investigation_saved" in actions
    assert "playbook_saved" in actions


def test_viewer_cannot_save_investigations_or_playbooks() -> None:
    dataset_id = _create_session("investigation_owner")
    _upload_dataset(dataset_id, actor="investigation_owner")

    investigation_response = client.post(
        f"/sessions/{dataset_id}/investigations",
        json={
            "question": "Should not save",
            "result": {},
        },
        headers={"X-API-Key": "investigation_owner", "X-User-Role": "viewer"},
    )
    assert investigation_response.status_code == 403
    assert "write" in investigation_response.json()["detail"]

    playbook_response = client.post(
        f"/sessions/{dataset_id}/playbooks",
        json={
            "name": "Viewer playbook",
            "question_template": "Should not save",
        },
        headers={"X-API-Key": "investigation_owner", "X-User-Role": "viewer"},
    )
    assert playbook_response.status_code == 403
    assert "write" in playbook_response.json()["detail"]
