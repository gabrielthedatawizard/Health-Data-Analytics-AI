"""
Tests for Data Quality Intelligence Engine (Phase 1.3)
"""

import pytest
import pandas as pd
import numpy as np
from backend.quality_intelligence import (
    QualityIntelligence,
    analyze_data_quality,
)


def test_missingness_mcar_and_mar():
    """Verify MCAR and MAR classification."""
    np.random.seed(42)
    n = 200

    # Age and Income, where missingness in Income depends heavily on Age (MAR)
    age = np.random.randint(18, 70, size=n)
    income = np.random.normal(50000, 15000, size=n)

    # Induce MAR: older patients (> 50) have 70% missing income, younger have 5% missing
    income_mar = income.copy()
    for i in range(n):
        if age[i] > 50 and np.random.rand() < 0.70:
            income_mar[i] = np.nan
        elif age[i] <= 50 and np.random.rand() < 0.05:
            income_mar[i] = np.nan

    # Random column with purely random missingness (MCAR)
    random_col = np.random.normal(10, 2, size=n)
    for i in range(n):
        if np.random.rand() < 0.15:
            random_col[i] = np.nan

    df = pd.DataFrame({
        "age": age,
        "income": income_mar,
        "random_metric": random_col,
    })

    engine = QualityIntelligence(df)

    res_mar = engine.classify_missingness_mechanism("income")
    assert res_mar["missingness_type"] == "MAR"
    assert len(res_mar["associated_columns"]) >= 1

    res_mcar = engine.classify_missingness_mechanism("random_metric")
    assert res_mcar["missingness_type"] == "MCAR"


def test_data_entry_errors():
    """Verify negative count and impossible vital measurement detection."""
    df_errors = pd.DataFrame({
        "patient_id": [f"P_{i}" for i in range(50)],
        "opd_attendances": [20, 15, -5, 40, -1] + [25]*45,  # 2 negative counts
        "body_temp_c": [36.5, 37.0, 52.0, 36.8, 12.0] + [37.0]*45,  # 2 impossible temps (>44 or <32)
        "coverage_pct": [50.0, 85.0, 125.0, -10.0] + [75.0]*46,  # 2 invalid percentage bounds
    })

    engine = QualityIntelligence(df_errors)
    errors = engine.detect_data_entry_errors()
    issue_types = [e.issue_type for e in errors]

    assert "negative_count_error" in issue_types
    assert "implausible_vital_measurement" in issue_types
    assert "invalid_percentage_range" in issue_types


def test_reporting_gaps_diagnosis():
    """Verify detection of missing months and attrition across facilities."""
    rows = []
    # Facility A: Reports all 4 months
    for m in ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]:
        rows.append({"facility_name": "Hospital A", "reporting_date": m, "visits": 100})

    # Facility B: Intermittent (misses Feb)
    for m in ["2024-01-01", "2024-03-01", "2024-04-01"]:
        rows.append({"facility_name": "Hospital B", "reporting_date": m, "visits": 80})

    # Facility C: Dropout after Feb
    for m in ["2024-01-01", "2024-02-01"]:
        rows.append({"facility_name": "Hospital C", "reporting_date": m, "visits": 50})

    df_long = pd.DataFrame(rows)
    engine = QualityIntelligence(df_long)
    gaps = engine.detect_reporting_gaps("reporting_date", "facility_name")

    assert gaps["status"] == "diagnosed"
    assert gaps["total_expected_periods"] == 4
    assert gaps["entities_with_gaps_count"] == 2

    worst_entities = {e["entity"]: e["gap_pattern"] for e in gaps["worst_reporting_entities"]}
    assert "Hospital C" in worst_entities
    assert worst_entities["Hospital C"] == "attrition_dropout"
    assert worst_entities["Hospital B"] == "intermittent_reporting"


def test_imputation_strategy_recommendation():
    """Verify imputation strategies for different column characteristics."""
    df_impute = pd.DataFrame({
        "facility": ["Fac 1", "Fac 2", "Fac 3", None, "Fac 5"],
        "temp_reading": [36.5, 37.0, None, 36.8, 37.2],
        "reporting_month": pd.date_range("2024-01-01", periods=5, freq="MS"),
    })
    engine = QualityIntelligence(df_impute)

    # Categorical
    strat_fac = engine.suggest_imputation_strategy("facility")
    assert strat_fac["strategy"] in ("explicit_missing_category", "mode_impute")

    # Numeric
    strat_temp = engine.suggest_imputation_strategy("temp_reading")
    assert "strategy" in strat_temp


def test_analytical_validity_and_report_card():
    """Verify analytical validity check and quality report card."""
    df_valid = pd.DataFrame({
        "encounter_id": [f"ID_{i}" for i in range(40)],
        "patient_age": np.random.randint(10, 80, size=40),
        "systolic_bp": np.random.randint(110, 160, size=40),
        "diastolic_bp": np.random.randint(70, 100, size=40),
    })
    res = analyze_data_quality(df_valid, dataset_name="vitals_sample")

    report = res["report_card"]
    assert "summary" in report
    assert report["summary"]["grade"] in ("A", "B", "C")
    assert report["summary"]["overall_score"] > 80.0

    validity = res["validity_assessment"]
    assert "target_analysis" in validity
    assert validity["is_valid"] is True
