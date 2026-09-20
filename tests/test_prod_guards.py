import pandas as pd
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def test_security_posture_never_leaks_secrets() -> None:
    resp = client.get("/system/security")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["pii_masking_default"] is True
    assert payload["audit_hash_chained"] is True
    assert "JWT_SECRET" not in resp.text
    assert "AZURE_OPENAI" not in resp.text


def test_smarter_fallback_answers_correlation_and_segment() -> None:
    bundle = {
        "data_coverage": {"mode": "full"},
        "quality": {"score": 92, "issues": []},
        "kpis": [{"id": "kpi_rows", "value": 100, "facts_refs": ["fact_001"]}],
        "insight_facts": [
            {"id": "fact_002", "type": "comparison",
             "value": {"metric": "top_opd_segment", "segment_field": "district", "segment": "North", "value": 123.0}},
            {"id": "fact_003", "type": "comparison",
             "value": {"metric": "pair_correlation", "x": "opd_visits", "y": "anc_visits", "value": 0.82}},
        ],
    }
    ans, refs, conf, cov, mode = main._safe_answer_from_facts("which district leads?", bundle)
    assert "North" in ans
    assert refs == ["fact_002"]

    ans2, refs2, _, _, _ = main._safe_answer_from_facts("are they correlated?", bundle)
    assert "r=0.82" in ans2
    assert "not proof of causation" in ans2
    assert refs2 == ["fact_003"]


def test_fast_profile_samples_large_frames() -> None:
    df = pd.DataFrame({"a": list(range(150_000)), "b": ["x"] * 150_000})
    profile = main._build_profile_fast(df)
    assert profile["profile_sampled"] is True
    assert profile["shape"]["rows"] == 150_000

    small = pd.DataFrame({"a": [1, 2, 3]})
    assert main._build_profile_fast(small)["profile_sampled"] is False
