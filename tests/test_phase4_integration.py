"""
Integration Tests for Phase 4 Adaptive Learning System
"""

import numpy as np
import pandas as pd
import pytest
from backend.ai_analytics_service import (
    build_facts_bundle,
    recommend_charts,
    ask_data,
    generate_insights,
)
from backend.feedback_loop import FeedbackLoop
from backend.knowledge_base import AnalyticsKnowledgeBase


def test_phase4_chart_recommendation_adapts_to_overrides(tmp_path, monkeypatch):
    fb_dir = tmp_path / "feedback"
    orig_fb_init = FeedbackLoop.__init__
    monkeypatch.setattr(
        FeedbackLoop,
        "__init__",
        lambda self, storage_dir=str(fb_dir): orig_fb_init(self, storage_dir=str(fb_dir)),
    )

    fb = FeedbackLoop(storage_dir=str(fb_dir))
    # Override 'admissions' default chart from 'line' to 'horizontal_bar' twice
    fb.capture_chart_override("admissions trend", "line", "horizontal_bar")
    fb.capture_chart_override("admissions trend", "line", "horizontal_bar")

    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "admissions": [100, 110, 120, 130],
    })
    bundle = build_facts_bundle(df, session_id="s_chart_test", dataset_id="ds_chart_test")

    recs = recommend_charts(bundle)
    trend_rec = next((r for r in recs if "admissions trend" in r.get("title", "").lower()), None)
    assert trend_rec is not None
    # Learned chart override should be applied!
    assert trend_rec["chart_type"] == "horizontal_bar"


def test_phase4_ask_data_applies_learned_analyst_corrections(tmp_path, monkeypatch):
    fb_dir = tmp_path / "feedback"
    orig_fb_init = FeedbackLoop.__init__
    monkeypatch.setattr(
        FeedbackLoop,
        "__init__",
        lambda self, storage_dir=str(fb_dir): orig_fb_init(self, storage_dir=str(fb_dir)),
    )

    fb = FeedbackLoop(storage_dir=str(fb_dir))
    # Capture human correction on outlier driver
    fb.capture_correction(
        question="Which metric has the highest outlier rate?",
        original_answer="Outpatient visits has 3 outliers.",
        user_correction="The December peak was caused by the annual mass screening campaign.",
        intent="outlier_detection",
    )

    df = pd.DataFrame({
        "facility": [f"Facility_{i}" for i in range(10)],
        "visits": [100, 105, 110, 95, 102, 108, 98, 101, 500, 104],
    })
    bundle = build_facts_bundle(df, session_id="s_ask_corr", dataset_id="ds_ask_corr")

    resp = ask_data("Which metric has the highest outlier rate?", bundle)
    assert "Analyst Knowledge Guidance" in resp["answer"]
    assert "annual mass screening campaign" in resp["answer"]
    assert resp.get("feedback_corrections_applied", 0) >= 1


def test_phase4_insights_enriches_from_knowledge_base_archetypes(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    orig_kb_init = AnalyticsKnowledgeBase.__init__
    monkeypatch.setattr(
        AnalyticsKnowledgeBase,
        "__init__",
        lambda self, storage_dir=str(kb_dir): orig_kb_init(self, storage_dir=str(kb_dir)),
    )

    kb = AnalyticsKnowledgeBase(storage_dir=str(kb_dir))
    kb.learn_dataset_pattern(
        archetype_name="mch_antenatal",
        detected_categories=["maternal_health"],
        key_columns=["anc_visits", "deliveries"],
        recommended_questions=[
            "What proportion of mothers attended all 4 standard ANC visits?",
        ],
        typical_dashboard_views=["ANC Retention Funnel"],
    )

    df = pd.DataFrame({
        "facility": ["Hosp_A", "Hosp_B"],
        "anc_visits": [200, 250],
        "deliveries": [40, 50],
    })
    bundle = build_facts_bundle(df, session_id="s_mch", dataset_id="ds_mch")

    insights = generate_insights(bundle)
    assert "follow_up_questions" in insights
    questions = insights["follow_up_questions"]
    # Verify that the knowledge base suggested question was injected
    assert any("4 standard anc visits" in q.lower() for q in questions)
