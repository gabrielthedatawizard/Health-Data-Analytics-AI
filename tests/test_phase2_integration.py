"""
Phase 2 Integration Test: Statistical Reasoning & Contextual Interpretation Engine
Verifies that statistical_engine and interpretation_engine
seamlessly enrich the facts bundle with inferential testing and domain-aware narratives.
"""

import pytest
import pandas as pd
import numpy as np
from backend.ai_analytics_service import profile_dataset, build_facts_bundle


@pytest.fixture
def longitudinal_hmis_df():
    """Generates a multi-facility longitudinal dataset with clear trends and disparities."""
    np.random.seed(42)
    records = []
    # 8 monthly periods
    periods = [f"2024-{m:02d}-01" for m in range(1, 9)]

    for month_idx, period in enumerate(periods):
        # General hospital: high visits with upward trend
        # Rural dispensary: low visits
        records.append({
            "facility_name": "Coast Provincial Hospital",
            "setting": "Urban",
            "reporting_date": period,
            "opd_attendances": 400 + month_idx * 50,  # upward trend
            "measles_vaccine_coverage_pct": 85.0 + month_idx * 1.5,
        })
        records.append({
            "facility_name": "Kilifi Rural Dispensary",
            "setting": "Rural",
            "reporting_date": period,
            "opd_attendances": 80 + np.random.randint(-5, 5),
            "measles_vaccine_coverage_pct": 70.0 + np.random.randint(-2, 2),
        })

    return pd.DataFrame(records)


def test_phase2_facts_bundle_statistical_enrichment(longitudinal_hmis_df):
    """Verify facts bundle contains significance tests, changepoints, group tests, and executive narrative."""
    profile = profile_dataset(longitudinal_hmis_df)
    facts = build_facts_bundle(
        df=longitudinal_hmis_df,
        dataset_id="phase2_test_ds",
        session_id="phase2_test_sess",
        profile=profile,
    )

    # 1. Trends have statistical tests and clinical severity
    trends = facts["trends"]
    assert len(trends) > 0

    first_trend_key = list(trends.keys())[0]
    trend_entry = trends[first_trend_key]
    assert "significance_test" in trend_entry
    sig_test = trend_entry["significance_test"]
    assert "is_significant" in sig_test
    assert "p_value" in sig_test
    assert "trend_direction" in sig_test
    assert "changepoints" in trend_entry
    assert "clinical_severity" in trend_entry

    # 2. Segments have statistical hypothesis test & effect sizes
    segments = facts["segments"]
    assert len(segments) > 0

    first_seg_key = list(segments.keys())[0]
    seg_entry = segments[first_seg_key]
    assert "statistical_significance" in seg_entry
    stat_sig = seg_entry["statistical_significance"]
    assert "test_used" in stat_sig
    assert "is_significant" in stat_sig
    assert "p_value" in stat_sig
    assert "effect_size" in stat_sig
    assert "practical_significance" in stat_sig

    # 3. Global Executive Narrative exists and is non-empty
    assert "executive_narrative" in facts
    assert isinstance(facts["executive_narrative"], str)
    assert len(facts["executive_narrative"]) > 10
