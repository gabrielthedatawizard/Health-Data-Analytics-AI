import io

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def _create_dataset() -> str:
    session = client.post("/sessions", json={"created_by": "tester"}, headers={"X-API-Key": "user_1"})
    assert session.status_code == 200
    dataset_id = session.json()["dataset_id"]

    df = pd.DataFrame(
        {
            "visit_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "district": ["North", "North", "South"],
            "payer": ["Medicare", "Commercial", "Medicare"],
            "opd_visits": [10, 20, 15],
            "patient_name": ["A", "B", "C"],
        }
    )
    files = {"file": ("health.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")}
    upload = client.post(f"/sessions/{dataset_id}/upload", files=files, data={"uploaded_by": "tester"})
    assert upload.status_code == 200

    profile = client.get(f"/sessions/{dataset_id}/profile")
    assert profile.status_code == 200

    facts = client.get(f"/sessions/{dataset_id}/facts")
    assert facts.status_code == 200
    return dataset_id


def test_semantic_layer_endpoint_excludes_pii() -> None:
    dataset_id = _create_dataset()

    response = client.get(f"/sessions/{dataset_id}/semantic-layer")

    assert response.status_code == 200
    payload = response.json()["semantic_layer"]
    dimension_fields = {item["field"] for item in payload["dimensions"]}
    assert "patient_name" not in dimension_fields
    assert "patient_name" in payload["pii_blocked_fields"]


def test_ask_returns_resolved_metric_id_and_governance(monkeypatch) -> None:
    dataset_id = _create_dataset()

    monkeypatch.setattr(
        main,
        "_generate_query_plan_llm",
        lambda question, facts_bundle, profile, semantic_layer: {
            "intent": "compare",
            "metrics": [{"metric_id": "sum_opd_visits"}],
            "group_by": ["district"],
            "filters": [{"field": "payer", "op": "in", "value": ["Medicare"]}],
            "time": {"field": "visit_date", "grain": "month", "start": None, "end": None},
            "limit": 50,
            "chart_hint": "bar",
        },
    )
    monkeypatch.setattr(
        main,
        "_summarize_query_result_llm",
        lambda question, result_df, facts_bundle, candidate_fact_ids: ("Summary", candidate_fact_ids[:1]),
    )

    response = client.post(
        f"/sessions/{dataset_id}/ask",
        json={"question": "Show 30-day readmissions by district"},
        headers={"X-API-Key": "user_1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_plan"]["metrics"][0]["metric_id"] == "sum_opd_visits"
    assert payload["query_plan"]["metrics"][0]["field"] == "opd_visits"
    assert payload["query_plan"]["metrics"][0]["op"] == "sum"
    assert payload["governance"]["validation_mode"] == "semantic_strict"
    assert payload["governance"]["semantic_metric_ids"] == ["sum_opd_visits"]
    assert "patient_name" in payload["governance"]["blocked_fields"]


def test_ask_rejects_pii_metric_plan(monkeypatch) -> None:
    dataset_id = _create_dataset()

    monkeypatch.setattr(
        main,
        "_generate_query_plan_llm",
        lambda question, facts_bundle, profile, semantic_layer: {
            "intent": "aggregate",
            "metrics": [{"op": "count", "field": "patient_name"}],
            "group_by": [],
            "filters": [],
            "time": {"field": None, "grain": None, "start": None, "end": None},
            "limit": 25,
            "chart_hint": "table",
        },
    )

    response = client.post(
        f"/sessions/{dataset_id}/ask",
        json={"question": "Count patients by name"},
        headers={"X-API-Key": "user_1"},
    )

    assert response.status_code == 422
    assert "PII-sensitive" in response.json()["detail"]
