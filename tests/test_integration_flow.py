import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main
from backend.tasks import generate_facts_task, generate_report_task


client = TestClient(main.app)


def test_session_upload_profile_facts_spec_report_flow() -> None:
    session_resp = client.post("/sessions", json={"created_by": "integration_tester"})
    assert session_resp.status_code == 200
    dataset_id = session_resp.json()["dataset_id"]

    df = pd.DataFrame(
        {
            "visit_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "district": ["North", "North", "South", "South"],
            "opd_visits": [12, 15, 9, 14],
            "patient_name": ["A", "B", "C", "D"],
        }
    )
    files = {"file": ("health.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
    upload_resp = client.post(f"/sessions/{dataset_id}/upload", files=files, data={"uploaded_by": "integration_tester"})
    assert upload_resp.status_code == 200

    profile_resp = client.get(f"/sessions/{dataset_id}/profile")
    assert profile_resp.status_code == 200
    assert "profile" in profile_resp.json()

    facts_job_resp = client.post(f"/sessions/{dataset_id}/facts", params={"mode": "sample"})
    assert facts_job_resp.status_code == 202
    facts_job_id = facts_job_resp.json()["job_id"]

    generate_facts_task(facts_job_id, dataset_id, "sample", 42)
    facts_status = client.get(f"/jobs/{facts_job_id}")
    assert facts_status.status_code == 200
    assert facts_status.json()["status"] == "succeeded"

    facts_resp = client.get(f"/sessions/{dataset_id}/facts")
    assert facts_resp.status_code in (200, 202)
    if facts_resp.status_code == 200:
        facts_bundle = facts_resp.json().get("facts_bundle", {})
        assert "data_coverage" in facts_bundle
        assert "insight_facts" in facts_bundle

    spec_resp = client.post(f"/sessions/{dataset_id}/dashboard-spec", params={"use_llm": "false"}, json={"template": "health_core"})
    assert spec_resp.status_code == 200
    assert "dashboard_spec" in spec_resp.json()

    report_resp = client.post(f"/sessions/{dataset_id}/report", json={})
    assert report_resp.status_code == 202
    report_job_id = report_resp.json()["job_id"]

    generate_report_task(report_job_id, dataset_id, None, None)
    report_status = client.get(f"/jobs/{report_job_id}")
    assert report_status.status_code == 200
    assert report_status.json()["status"] == "succeeded"

    pdf_resp = client.get(f"/sessions/{dataset_id}/export/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers.get("content-type", "").startswith("application/pdf")

    audit_resp = client.get(f"/sessions/{dataset_id}/audit")
    assert audit_resp.status_code == 200
    assert "events" in audit_resp.json()
