"""
Unit Tests for Phase 4.2 Feedback Loop
"""

import shutil
import tempfile
from pathlib import Path
import pytest
from backend.feedback_loop import FeedbackLoop


@pytest.fixture
def temp_feedback():
    temp_dir = tempfile.mkdtemp()
    fb = FeedbackLoop(storage_dir=temp_dir)
    yield fb
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_capture_and_retrieve_corrections(temp_feedback):
    # Capture an analyst correction on ANC dropout calculation
    rec = temp_feedback.capture_correction(
        question="How is ANC dropout calculated for rural facilities?",
        original_answer="ANC dropout is calculated as ANC1 visits minus ANC4 visits.",
        user_correction="Exclude patients referred out from the denominator when calculating ANC dropout.",
        intent="definition_lookup",
        user_id="lead_epidemiologist",
    )
    assert rec["id"].startswith("corr_")

    # Ask a similar question
    query = "What is the formula to calculate ANC dropout rate in facilities?"
    applicable = temp_feedback.get_applicable_corrections(query, min_jaccard=0.20)

    assert len(applicable) >= 1
    assert "Exclude patients referred out" in applicable[0]["user_correction"]


def test_apply_learned_corrections_enhancement(temp_feedback):
    temp_feedback.capture_correction(
        question="Why did immunization drop in Western district?",
        original_answer="Immunization dropped by 12% in the district.",
        user_correction="The cold chain generator at Regional Depot was broken from May 4th to 18th.",
    )

    draft = "Immunization dropped by 12% in the district due to reporting variation."
    applicable = temp_feedback.get_applicable_corrections("Why did immunization drop in Western district?")
    enhanced = temp_feedback.apply_learned_corrections(draft, applicable)

    assert "Analyst Knowledge Guidance" in enhanced
    assert "cold chain generator" in enhanced


def test_chart_override_learning(temp_feedback):
    # Default is line
    default_chart = "line"

    # User overrides it once -> not yet threshold (min_votes=2)
    temp_feedback.capture_chart_override(
        metric="facility_beds",
        default_recommended="line",
        user_chosen="horizontal_bar",
    )
    rec_1 = temp_feedback.get_adjusted_chart_recommendation("facility_beds", default_chart, min_votes=2)
    assert rec_1 == "line"

    # User overrides it a second time -> meets threshold
    temp_feedback.capture_chart_override(
        metric="facility_beds",
        default_recommended="line",
        user_chosen="horizontal_bar",
    )
    rec_2 = temp_feedback.get_adjusted_chart_recommendation("facility_beds", default_chart, min_votes=2)
    assert rec_2 == "horizontal_bar"


def test_ratings_and_satisfaction_metrics(temp_feedback):
    temp_feedback.capture_rating("ask_data", "q_001", "positive")
    temp_feedback.capture_rating("ask_data", "q_002", "positive")
    temp_feedback.capture_rating("reports", "rep_001", "negative", feedback_text="Missing summary table")

    metrics = temp_feedback.get_feedback_metrics()
    assert metrics["total_ratings"] == 3
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 1
    assert metrics["satisfaction_rate"] == 66.7
