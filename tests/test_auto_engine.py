import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def _upload_df(actor: str, df: pd.DataFrame, auto_run: bool = False) -> str:
    session = client.post("/sessions", json={"created_by": actor}, headers={"X-API-Key": actor})
    assert session.status_code == 200
    dataset_id = session.json()["dataset_id"]
    files = {"file": ("auto.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
    suffix = "?auto_run=true" if auto_run else ""
    upload = client.post(
        f"/sessions/{dataset_id}/upload{suffix}",
        files=files,
        data={"uploaded_by": actor},
        headers={"X-API-Key": actor},
    )
    assert upload.status_code == 200
    return dataset_id


def test_pii_defaults_masked_and_valid_not_flagged() -> None:
    actor = "auto_owner_pii"
    df = pd.DataFrame(
        {
            "visit_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "valid": ["a", "b", "c"],
            "opd_visits": [10, 20, 15],
        }
    )
    dataset_id = _upload_df(actor, df)
    meta = client.get(f"/sessions/{dataset_id}", headers={"X-API-Key": actor}).json()
    assert meta["pii_masking_enabled"] is True

    profile = client.get(f"/sessions/{dataset_id}/profile", headers={"X-API-Key": actor}).json()["profile"]
    assert "visit_date" not in (profile.get("pii_candidates") or [])
    assert "valid" not in (profile.get("pii_candidates") or [])


def test_explicit_role_cannot_escalate_to_admin() -> None:
    assert main._role_from_identity("bob", explicit_role="admin") == "analyst"
    assert main._role_from_identity("manager_1", explicit_role="viewer") == "viewer"
    assert main._role_from_identity("admin_root", explicit_role="admin") == "admin"


def test_upload_auto_run_builds_dashboard_version() -> None:
    actor = "auto_owner_run"
    df = pd.DataFrame(
        {
            "visit_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "district": ["A", "B", "A"],
            "opd_visits": [10, 20, 15],
        }
    )
    dataset_id = _upload_df(actor, df, auto_run=True)
    auto = client.post(f"/sessions/{dataset_id}/auto-run", headers={"X-API-Key": actor})
    assert auto.status_code == 200
    payload = auto.json()
    assert payload["status"] == "auto_analyzed"
    assert payload["dashboard_version"] is not None

    versions = client.get(f"/sessions/{dataset_id}/dashboard-versions", headers={"X-API-Key": actor})
    assert versions.status_code == 200
    assert len(versions.json()["versions"]) >= 1

    first_version = versions.json()["versions"][-1]["version"]
    restore = client.post(
        f"/sessions/{dataset_id}/dashboard-versions/{first_version}/restore",
        headers={"X-API-Key": actor},
    )
    assert restore.status_code == 200


def test_share_link_is_masked_and_token_gated() -> None:
    actor = "auto_owner_share"
    df = pd.DataFrame(
        {
            "patient_name": ["A", "B", "C"],
            "opd_visits": [10, 20, 15],
        }
    )
    dataset_id = _upload_df(actor, df, auto_run=True)
    share = client.post(f"/sessions/{dataset_id}/share", headers={"X-API-Key": actor})
    assert share.status_code == 200
    token = share.json()["share_token"]
    assert share.json()["masked"] is True

    denied = client.get(f"/sessions/{dataset_id}/shared", params={"token": "badtoken"})
    assert denied.status_code == 403

    public = client.get(f"/sessions/{dataset_id}/shared", params={"token": token})
    assert public.status_code == 200
    assert public.json()["masked"] is True
