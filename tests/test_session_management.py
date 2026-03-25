from uuid import uuid4

from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def test_session_list_update_delete_are_user_scoped() -> None:
    actor_a = f"scope_a_{uuid4().hex}"
    actor_b = f"scope_b_{uuid4().hex}"

    created_a = client.post(
        "/sessions",
        json={"created_by": actor_a, "display_name": "Actor A Dataset"},
        headers={"X-API-Key": actor_a},
    )
    assert created_a.status_code == 200
    dataset_id_a = created_a.json()["dataset_id"]

    created_b = client.post(
        "/sessions",
        json={"created_by": actor_b, "display_name": "Actor B Dataset"},
        headers={"X-API-Key": actor_b},
    )
    assert created_b.status_code == 200
    dataset_id_b = created_b.json()["dataset_id"]

    list_a = client.get("/sessions", headers={"X-API-Key": actor_a})
    assert list_a.status_code == 200
    session_ids_a = {item["dataset_id"] for item in list_a.json()["sessions"]}
    assert dataset_id_a in session_ids_a
    assert dataset_id_b not in session_ids_a

    update_a = client.patch(
        f"/sessions/{dataset_id_a}",
        json={"display_name": "Renamed Dataset", "description": "Scoped session"},
        headers={"X-API-Key": actor_a},
    )
    assert update_a.status_code == 200
    payload = update_a.json()["session"]
    assert payload["display_name"] == "Renamed Dataset"
    assert payload["description"] == "Scoped session"

    delete_a = client.delete(f"/sessions/{dataset_id_a}", headers={"X-API-Key": actor_a})
    assert delete_a.status_code == 200
    assert delete_a.json()["deleted"] is True

    missing = client.get(f"/sessions/{dataset_id_a}")
    assert missing.status_code == 404

    cleanup_b = client.delete(f"/sessions/{dataset_id_b}", headers={"X-API-Key": actor_b})
    assert cleanup_b.status_code == 200
