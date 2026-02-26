import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main
from backend.jobs import create_job, get_job
from backend.tasks import generate_facts_task


client = TestClient(main.app)


def test_generate_facts_task():
    resp = client.post("/sessions", json={"created_by": "tester"})
    dataset_id = resp.json()["dataset_id"]

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
    upload = client.post(f"/sessions/{dataset_id}/upload", files=files, data={"uploaded_by": "tester"})
    assert upload.status_code == 200

    job = create_job("facts", dataset_id, {"mode": "full", "seed": 42})
    result = generate_facts_task(job["job_id"], dataset_id, "full", 42)
    assert "facts_path" in result
    status = get_job(job["job_id"])
    assert status is not None
    assert status["status"] == "completed"
