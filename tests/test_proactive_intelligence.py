"""
Unit Tests for Phase 6.2 Proactive Intelligence System
"""

import pandas as pd
import pytest
from backend.proactive_intelligence import ProactiveIntelligence


@pytest.fixture
def proactive_engine():
    return ProactiveIntelligence()


def test_scan_dataset_target_breach(proactive_engine):
    # Simulated facts bundle where ANC 1 coverage trails the 95% target
    facts = {
        "dataset_id": "ds_scan_test",
        "metrics": {
            "kpis": {
                "anc_1_coverage": {"mean": 72.5, "total": 72.5, "count": 100},
                "facility_delivery_rate": {"mean": 94.0, "total": 94.0, "count": 100},
            }
        },
        "trends": {},
        "anomaly_report": {},
    }

    scan = proactive_engine.scan_dataset(pd.DataFrame(), facts)

    assert scan["total_findings"] >= 1
    target_findings = [f for f in scan["findings"] if f["category"] == "target_breach"]
    assert len(target_findings) >= 1
    assert "ANC 1 Coverage" in target_findings[0]["title"]
    assert "trailing national target" in target_findings[0]["detail"]


def test_scan_dataset_early_warning_trend(proactive_engine):
    facts = {
        "dataset_id": "ds_ew_test",
        "metrics": {"kpis": {"immunization_doses": {"total": 5000}}},
        "trends": {
            "immunization_doses_monthly": {
                "metric": "immunization_doses",
                "significance_test": {
                    "trend_direction": "decreasing",
                    "is_significant": True,
                    "p_value": 0.003,
                    "slope": -25.4,
                },
            }
        },
        "anomaly_report": {},
    }

    scan = proactive_engine.scan_dataset(pd.DataFrame(), facts)

    ew_findings = [f for f in scan["findings"] if f["category"] == "early_warning"]
    assert len(ew_findings) >= 1
    assert "Declining Trajectory" in ew_findings[0]["title"]
    assert ew_findings[0]["severity"] == "high"


def test_suggest_cross_dataset_analyses(proactive_engine):
    datasets_meta = [
        {
            "name": "EPI_Immunization_Records",
            "columns": ["facility_id", "reporting_month", "pentavalent_doses", "measles_doses"],
        },
        {
            "name": "Vaccine_Cold_Chain_Inventory",
            "columns": ["facility_id", "reporting_month", "fridge_temp", "stockout_days"],
        },
    ]

    suggestions = proactive_engine.suggest_cross_dataset_analyses(datasets_meta)

    assert len(suggestions) == 1
    sugg = suggestions[0]
    assert "facility_id" in sugg["suggested_join_keys"]
    assert sugg["impact_score"] == "High"
    assert "EPI_Immunization_Records" in sugg["dataset_pair"]


def test_generate_executive_briefing(proactive_engine):
    scan_results = {
        "findings": [
            {
                "category": "target_breach",
                "severity": "critical",
                "title": "Severe ANC Deficit",
                "detail": "Trailing target by 22 pts",
                "action": "Dispatch emergency mobile clinics",
            },
            {
                "category": "early_warning",
                "severity": "high",
                "title": "Immunization drop",
                "action": "Review vaccine depot stocks",
            },
        ]
    }

    briefing = proactive_engine.generate_executive_briefing(scan_results)

    assert "health_score" in briefing
    assert briefing["health_score"] < 80  # Penalized by critical + high findings
    assert "High Urgent Intervention" in briefing["posture"] or "Moderate" in briefing["posture"]
    assert len(briefing["priority_recommendations"]) == 2
