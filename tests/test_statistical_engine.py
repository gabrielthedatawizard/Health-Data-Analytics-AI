"""
Tests for Statistical Reasoning Engine (Phase 2.1)
"""

import pytest
import pandas as pd
import numpy as np
from backend.statistical_engine import (
    StatisticalEngine,
    run_statistical_inference,
)


def test_mann_kendall_trend_significance():
    """Verify Mann-Kendall detection on upward, downward, and flat series."""
    # 1. Strictly increasing series
    increasing_vals = [10, 12, 15, 19, 24, 28, 35, 42, 50, 61, 70]
    res_inc = StatisticalEngine.test_trend_significance(increasing_vals)
    assert res_inc.is_significant is True
    assert res_inc.trend_direction == "increasing"
    assert res_inc.p_value < 0.05
    assert res_inc.slope > 0

    # 2. Strictly decreasing series
    decreasing_vals = [100, 92, 85, 78, 65, 54, 48, 35, 22, 10]
    res_dec = StatisticalEngine.test_trend_significance(decreasing_vals)
    assert res_dec.is_significant is True
    assert res_dec.trend_direction == "decreasing"
    assert res_dec.p_value < 0.05
    assert res_dec.slope < 0

    # 3. Series without trend (stationary equilibrium)
    flat_vals = [50.0, 50.2, 49.8, 50.1, 49.9, 50.0, 50.1, 49.9, 50.0, 50.2, 49.8, 50.0]
    res_flat = StatisticalEngine.test_trend_significance(flat_vals)
    assert res_flat.is_significant is False
    assert res_flat.trend_direction == "stable"


def test_group_difference_two_groups():
    """Verify Welch's t-test or Mann-Whitney U test between 2 cohorts."""
    np.random.seed(42)
    # Group A: Urban facilities with high attendances
    # Group B: Rural facilities with low attendances
    urban = np.random.normal(500, 40, size=30)
    rural = np.random.normal(200, 30, size=30)

    df_2g = pd.DataFrame({
        "opd_attendances": np.concatenate([urban, rural]),
        "setting": ["Urban"] * 30 + ["Rural"] * 30,
    })

    res = StatisticalEngine.test_group_difference(df_2g, "opd_attendances", "setting")
    assert res.is_significant is True
    assert res.p_value < 0.001
    assert res.practical_significance == "large"
    assert "Urban" in res.group_means
    assert "Rural" in res.group_means
    assert res.group_means["Urban"] > res.group_means["Rural"]


def test_group_difference_multiple_groups_anova_or_kruskal():
    """Verify multi-group variance evaluation (ANOVA / Kruskal-Wallis)."""
    np.random.seed(42)
    df_multi = pd.DataFrame({
        "score": np.concatenate([
            np.random.normal(85, 5, 20),
            np.random.normal(65, 6, 20),
            np.random.normal(45, 5, 20),
        ]),
        "cadre": ["Doctor"] * 20 + ["Nurse"] * 20 + ["Community Health Worker"] * 20,
    })

    res = StatisticalEngine.test_group_difference(df_multi, "score", "cadre")
    assert res.is_significant is True
    assert res.p_value < 0.001
    assert res.effect_size_metric == "eta_squared"
    assert res.effect_size > 0.14  # Large effect size
    assert res.practical_significance == "large"


def test_changepoint_detection():
    """Verify CUSUM changepoint detection at structural shifts."""
    # Series with a clear jump at index 10 (e.g. intervention start)
    series = [50.0] * 10 + [120.0] * 10
    periods = [f"2024-M{i+1:02d}" for i in range(20)]

    changepoints = StatisticalEngine.detect_changepoints(series, labels=periods, min_segment_length=3)
    assert len(changepoints) >= 1

    first_cp = changepoints[0]
    assert first_cp.direction == "upward_shift"
    assert first_cp.percent_change > 100.0
    assert first_cp.period_label == "2024-M11"


def test_segmentation_analysis():
    """Verify segmentation driver ranking by impact."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "visits": np.concatenate([np.random.normal(800, 50, 30), np.random.normal(300, 30, 30)]),
        "facility_level": ["Level 5"] * 30 + ["Level 2"] * 30,
        "region": (["North", "South"] * 30)[:n],
    })

    ranked = StatisticalEngine.run_segmentation_analysis(df, "visits")
    assert len(ranked) >= 1
    top_driver = ranked[0]
    assert top_driver["dimension"] == "facility_level"
    assert top_driver["variance_explained_pct"] > 50.0
