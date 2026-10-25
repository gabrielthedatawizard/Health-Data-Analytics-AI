"""
Unit Tests for Phase 6.1 Autonomous Investigation Agent
"""

import numpy as np
import pandas as pd
import pytest
from backend.investigation_agent import (
    InvestigationAgent,
    InvestigationStep,
    InvestigationReport,
)


@pytest.fixture
def sample_health_df():
    # 24-month simulated facility dataset
    n = 24
    dates = pd.date_range("2023-01-01", periods=n, freq="MS").strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "facility": ["District Hospital"] * 12 + ["Regional Referral"] * 12,
        "date": dates,
        "maternal_admissions": [80 + i * 3 for i in range(12)] + [140 + i * 5 for i in range(12)],
        "deliveries": [40 + i * 2 for i in range(12)] + [70 + i * 3 for i in range(12)],
        "stockouts_oxytocin": [0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 2, 1] * 2,
    })


def test_question_decomposition(sample_health_df):
    agent = InvestigationAgent()
    question = "Why did maternal admissions surge in Regional Referral facility over the last year?"
    cols = list(sample_health_df.columns)

    metric, steps = agent.decompose_question(question, cols)

    assert "maternal_admissions" in metric
    assert len(steps) == 4
    pillars = [s.pillar for s in steps]
    assert "temporal" in pillars
    assert "segment_drivers" in pillars
    assert "cross_correlation" in pillars
    assert "data_quality" in pillars


def test_execute_investigation_steps(sample_health_df):
    agent = InvestigationAgent()
    metric, steps = agent.decompose_question(
        "Investigate drivers of maternal admissions",
        list(sample_health_df.columns),
    )

    facts_mock = {
        "trends": {
            "maternal_admissions_monthly": {
                "metric": "maternal_admissions",
                "significance_test": {
                    "is_significant": True,
                    "trend_direction": "increasing",
                    "p_value": 0.0012,
                },
                "changepoints": [{"period": "2024-01-01", "direction": "surge", "confidence": 0.95}],
            }
        },
        "anomaly_report": {"records": []},
    }

    executed_steps = [agent.execute_step(s, sample_health_df, facts_mock) for s in steps]

    assert all(s.status == "completed" for s in executed_steps)
    # Temporal step found changepoint
    t_step = next(s for s in executed_steps if s.pillar == "temporal")
    assert "increasing" in t_step.evidence

    # Segment step identified top driver
    seg_step = next(s for s in executed_steps if s.pillar == "segment_drivers")
    assert "Regional Referral" in seg_step.evidence


def test_investigate_full_report(sample_health_df):
    agent = InvestigationAgent()
    facts_mock = {
        "trends": {
            "maternal_admissions_monthly": {
                "metric": "maternal_admissions",
                "significance_test": {"trend_direction": "increasing", "is_significant": True, "p_value": 0.005},
            }
        }
    }

    report: InvestigationReport = agent.investigate(
        "Why did maternal admissions increase?",
        sample_health_df,
        facts=facts_mock,
    )

    assert report.investigation_id.startswith("inv_")
    assert report.target_metric == "maternal_admissions"
    assert "Autonomous Investigation" in report.executive_summary
    assert len(report.evidence_chain) == 4
    assert len(report.recommended_playbook) == 3
    assert report.steps_executed == 4
