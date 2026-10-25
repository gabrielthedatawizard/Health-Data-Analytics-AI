"""
Integration Tests for Phase 6 Autonomous Analytics Agent
"""

import pandas as pd
import pytest
from backend.ai_analytics_service import build_facts_bundle, ask_data
from backend.investigation_agent import InvestigationAgent
from backend.proactive_intelligence import ProactiveIntelligence


def test_phase6_ask_data_triggers_autonomous_investigation():
    # Build dataset with clear trend and segment concentration
    dates = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({
        "facility": ["County Hospital"] * 6 + ["St. Jude"] * 6,
        "date": dates,
        "admissions": [100, 110, 120, 130, 140, 150, 60, 65, 70, 75, 80, 85],
    })
    bundle = build_facts_bundle(df, session_id="s_inv_test", dataset_id="ds_inv_test")

    # Ask root-cause "why" / "explain" question
    res = ask_data("Why did admissions change and what is the primary driver?", bundle)

    assert "Autonomous Investigation" in res["answer"]
    assert "County Hospital" in res["answer"] or "St. Jude" in res["answer"]
    assert res["query_plan"]["intent"] == "root_cause_investigation"
    assert res.get("mode") == "autonomous_investigation"


def test_phase6_investigation_agent_end_to_end_pipeline():
    df = pd.DataFrame({
        "subcounty": ["North", "North", "South", "South"],
        "period": ["2024-01", "2024-02", "2024-01", "2024-02"],
        "malaria_cases": [200, 450, 50, 60],
        "rapid_tests": [500, 900, 200, 210],
    })
    bundle = build_facts_bundle(df, session_id="s_agent", dataset_id="ds_agent")

    agent = InvestigationAgent()
    report = agent.investigate(
        "Why did malaria cases spike in North subcounty?",
        df,
        facts=bundle,
    )

    assert report.target_metric == "malaria_cases"
    assert len(report.evidence_chain) == 4
    # All 4 pillars completed
    for step in report.evidence_chain:
        assert step["status"] == "completed"

    # Playbook formulated
    assert len(report.recommended_playbook) == 3
    assert any("Days 1-7" in p["priority"] for p in report.recommended_playbook)
