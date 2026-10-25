"""
Unit Tests for Phase 4.1 Analytics Knowledge Base
"""

import shutil
import tempfile
from pathlib import Path
import pytest
from backend.knowledge_base import AnalyticsKnowledgeBase


@pytest.fixture
def temp_kb():
    temp_dir = tempfile.mkdtemp()
    kb = AnalyticsKnowledgeBase(storage_dir=temp_dir)
    yield kb
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_indicator_learning_and_retrieval(temp_kb):
    # 1. Standard default indicators are present
    anc = temp_kb.get_indicator_definition("anc_1_coverage")
    assert anc is not None
    assert anc["target"] == 95.0
    assert "WHO_HMIS" in anc["source"]

    # 2. Learn a custom health indicator
    learned = temp_kb.learn_indicator_definition(
        name="TB Cure Rate",
        formula="(TB_Cured / Total_TB_Cohort) * 100",
        unit="percentage",
        target=85.0,
        direction="higher_is_better",
        source="National_TB_Program",
        tags=["tuberculosis", "treatment_outcomes"],
    )
    assert learned["name"] == "TB Cure Rate"

    # Exact retrieve
    retrieved = temp_kb.get_indicator_definition("tb_cure_rate")
    assert retrieved is not None
    assert retrieved["target"] == 85.0

    # Fuzzy / substring retrieve
    fuzzy = temp_kb.get_indicator_definition("tb cure")
    assert fuzzy is not None
    assert fuzzy["name"] == "TB Cure Rate"

    # List by tag
    tb_list = temp_kb.list_indicators(tag="tuberculosis")
    assert len(tb_list) >= 1
    assert tb_list[0]["name"] == "TB Cure Rate"


def test_dataset_archetype_learning_and_suggestions(temp_kb):
    # Learn an archetype for Malaria Outbreak Surveillance
    temp_kb.learn_dataset_pattern(
        archetype_name="malaria_surveillance",
        detected_categories=["epidemiology", "cases", "lab"],
        key_columns=["malaria_confirmed_rdt", "malaria_microscopy", "act_treatments"],
        recommended_questions=[
            "What is the test positivity rate across subcounties?",
            "Is there an ACT stockout correlating with high caseloads?",
        ],
        typical_dashboard_views=["Positivity Trendline", "Hotspot Geo Map"],
    )

    # Match against schema containing these columns
    cols = ["facility_name", "reporting_month", "malaria_confirmed_rdt", "act_treatments"]
    cats = ["epidemiology", "cases"]
    sugg = temp_kb.suggest_analyses_for_dataset(cols, detected_categories=cats)

    assert sugg["matched_archetype"] == "malaria_surveillance"
    assert sugg["confidence"] > 0.6
    assert any("test positivity" in q.lower() for q in sugg["suggested_questions"])


def test_user_preference_learning(temp_kb):
    temp_kb.learn_user_preference("dr_sarah", "default_chart", "stacked_bar")
    temp_kb.learn_user_preference("dr_sarah", "show_confidence_intervals", True)

    prefs = temp_kb.get_user_preferences("dr_sarah")
    assert prefs["default_chart"] == "stacked_bar"
    assert prefs["show_confidence_intervals"] is True


def test_recurring_issue_tracking_and_chronic_offenders(temp_kb):
    # Record multiple issues for St. Jude Hospital
    temp_kb.record_issue("St. Jude Hospital", "frozen_values", "ds_jan", "Repeated 120 visits")
    temp_kb.record_issue("St. Jude Hospital", "frozen_values", "ds_feb", "Repeated 120 visits")
    temp_kb.record_issue("St. Jude Hospital", "reporting_gap", "ds_mar", "Zero submissions")

    # Record 1 issue for Mercy Clinic
    temp_kb.record_issue("Mercy Clinic", "missing_columns", "ds_feb", "Missing coordinates")

    # St. Jude should have 3 issues
    st_jude_issues = temp_kb.get_recurring_issues("St. Jude Hospital")
    assert len(st_jude_issues) == 3

    # Detect chronic offenders (>= 2 incidents)
    chronic = temp_kb.detect_chronic_offenders(min_incidents=2)
    assert len(chronic) == 1
    assert chronic[0]["entity"] == "St. Jude Hospital"
    assert chronic[0]["total_incidents"] == 3
