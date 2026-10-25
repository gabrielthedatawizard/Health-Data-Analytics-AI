"""
Tests for Contextual Interpretation & Benchmark Engine (Phase 2.2)
"""

import pytest
from backend.interpretation_engine import (
    InterpretationEngine,
    interpret_analytics,
    HEALTH_BENCHMARKS,
)


def test_benchmark_database():
    """Verify health benchmarks covering MNCH, disease control, and hospital systems."""
    assert "vaccination_coverage" in HEALTH_BENCHMARKS
    assert "bed_occupancy_rate" in HEALTH_BENCHMARKS
    assert "malaria_positivity_rate" in HEALTH_BENCHMARKS
    assert "stock_out_days" in HEALTH_BENCHMARKS


def test_benchmark_interpretation_thresholds():
    """Verify threshold evaluation for higher-is-better, lower-is-better, and target-range."""
    engine = InterpretationEngine()

    # 1. Higher is better (Vaccination target >= 90%)
    res_high_opt = engine.interpret_with_benchmark("measles_vaccination_rate", 94.0)
    assert res_high_opt["status"] in ("on_target", "exceeds_target")
    assert res_high_opt["gap"] >= 0

    res_high_bad = engine.interpret_with_benchmark("measles_vaccination_rate", 62.0)
    assert res_high_bad["status"] == "critical_alert"
    assert res_high_bad["gap"] < -20

    # 2. Lower is better (Malaria TPR target <= 5%)
    res_low_good = engine.interpret_with_benchmark("malaria_tpr", 3.2)
    assert res_low_good["status"] == "on_target"
    assert res_low_good["gap"] <= 0

    res_low_bad = engine.interpret_with_benchmark("malaria_tpr", 18.5)
    assert res_low_bad["status"] == "critical_alert"
    assert res_low_bad["gap"] > 10

    # 3. Target Range (Bed Occupancy Rate 80% - 85%)
    res_range_opt = engine.interpret_with_benchmark("bed_occupancy_rate", 82.0)
    assert res_range_opt["status"] == "optimal"

    res_range_over = engine.interpret_with_benchmark("bed_occupancy_rate", 98.0)
    assert res_range_over["status"] in ("warning", "concerning")


def test_trend_severity_assessment():
    """Verify polarity-aware severity for improving vs worsening trends."""
    engine = InterpretationEngine()

    # Increasing vaccination = improving
    sev_vax = engine.assess_trend_severity("measles_vaccination", "increasing", True, 12.0)
    assert sev_vax["severity"] == "improving"

    # Increasing malaria = critical warning
    sev_malaria = engine.assess_trend_severity("malaria_cases", "increasing", True, 25.0)
    assert sev_malaria["severity"] == "critical"
    assert sev_malaria["urgency"] in ("high", "emergency")


def test_executive_narrative_generation():
    """Verify grounded natural language narrative synthesis."""
    trend_data = {
        "trend_direction": "increasing",
        "p_value": 0.002,
        "is_significant": True,
        "slope": 5.4,
    }
    changepoints = [{
        "period_label": "2024-06",
        "mean_before": 120.0,
        "mean_after": 185.0,
        "percent_change": 54.2,
        "confidence": 0.95,
    }]
    group_data = {
        "is_significant": True,
        "test_used": "Mann-Whitney U Test",
        "p_value": 0.0001,
        "effect_size": 1.25,
        "effect_size_metric": "hedges_g",
        "practical_significance": "large",
    }

    res = interpret_analytics(
        metric="opd_attendances",
        observed_value=850,
        trend_result=trend_data,
        group_result=group_data,
        changepoints=changepoints,
    )

    narrative = res["executive_narrative"]
    assert "statistically significant increasing trajectory" in narrative
    assert "p = 0.0020" in narrative
    assert "2024-06" in narrative
    assert "+54.2%" in narrative
    assert "Mann-Whitney U Test" in narrative
