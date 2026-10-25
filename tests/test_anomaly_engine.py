"""
Unit Tests for Phase 3.2 Anomaly Engine
"""

import numpy as np
import pandas as pd
import pytest
from backend.anomaly_engine import (
    AnomalyEngine,
    AnomalyRecord,
    DriftAlert,
)


def test_iqr_and_mad_scoring():
    # 20 regular points + 1 massive extreme outlier
    data = [20.0] * 20
    data[10] = 500.0  # extreme outlier
    s = pd.Series(data)

    iqr_scores = AnomalyEngine._compute_iqr_scores(s)
    mad_scores = AnomalyEngine._compute_mad_scores(s)

    assert iqr_scores[10] > 0.7
    assert mad_scores[10] > 0.7
    assert iqr_scores[0] < 0.2


def test_domain_anomaly_classification():
    col_stats = {"median": 100.0, "q75": 120.0, "iqr": 30.0, "max": 150.0}

    # Negative volume count
    atype, sev, hyp, act = AnomalyEngine.classify_anomaly_type(
        "patient_visits", -15.0, col_stats, iqr_score=0.9, mad_score=0.9
    )
    assert atype == "data_entry_error"
    assert sev == "critical"
    assert "Negative value" in hyp

    # Impossible body temp
    atype, sev, hyp, act = AnomalyEngine.classify_anomaly_type(
        "patient_temperature", 98.6, col_stats, iqr_score=0.9, mad_score=0.9
    )
    assert atype == "data_entry_error"
    assert sev == "critical"
    assert "Fahrenheit" in act

    # Epidemic surge
    atype, sev, hyp, act = AnomalyEngine.classify_anomaly_type(
        "malaria_cases", 450.0, col_stats, iqr_score=0.9, mad_score=0.9
    )
    assert atype == "genuine_spike"
    assert sev == "critical"
    assert "outbreak" in act.lower()


def test_detect_anomalies_dataframe():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "facility": [f"Facility_{i % 5}" for i in range(n)],
        "outpatient_visits": np.random.normal(200, 15, n),
        "deliveries": np.random.normal(40, 5, n),
    })

    # Inject deliberate anomalies
    df.loc[12, "outpatient_visits"] = 2500.0  # huge 10x surge
    df.loc[25, "deliveries"] = -8.0           # negative deliveries

    records = AnomalyEngine.detect_anomalies(df, numeric_cols=["outpatient_visits", "deliveries"])

    assert len(records) >= 2
    row_indices = [r.row_index for r in records]
    assert 12 in row_indices
    assert 25 in row_indices

    # Check that row 25 was identified as critical data entry error
    rec_25 = next(r for r in records if r.row_index == 25)
    assert rec_25.severity == "critical"
    assert rec_25.anomaly_type == "data_entry_error"


def test_detect_reporting_anomalies_frozen_and_catchup():
    df = pd.DataFrame({
        "facility": ["Hospital_A"] * 6 + ["Hospital_B"] * 6,
        "month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"] * 2,
        # Hospital_A has frozen numbers in periods 2, 3, 4, 5
        "immunizations": [120, 150, 150, 150, 150, 160, 100, 0, 0, 480, 110, 115],
    })

    issues = AnomalyEngine.detect_reporting_anomalies(df, facility_col="facility", time_col="month")

    assert len(issues) >= 2
    issue_types = [iss["issue_type"] for iss in issues]
    assert "frozen_repeated_values" in issue_types
    assert "batch_catch_up_spike" in issue_types


def test_distribution_drift_monitoring_psi():
    np.random.seed(42)
    # Baseline normal distribution
    baseline = np.random.normal(100, 15, 200)

    # Identical distribution -> stable
    curr_stable = np.random.normal(100, 15, 100)
    res_stable = AnomalyEngine.monitor_metric_drift(curr_stable, baseline, metric_name="opd")
    assert res_stable.drift_status == "stable"
    assert res_stable.psi_score < 0.15

    # Severely shifted distribution (mean jumped from 100 to 250)
    curr_drift = np.random.normal(250, 25, 100)
    res_drift = AnomalyEngine.monitor_metric_drift(curr_drift, baseline, metric_name="opd")
    assert res_drift.drift_status == "significant_drift"
    assert res_drift.psi_score > 0.25


def test_generate_anomaly_report_card():
    df = pd.DataFrame({
        "facility": [f"Facility_{i}" for i in range(25)],
        "cases": [10.0] * 24 + [950.0],
    })
    report = AnomalyEngine.generate_anomaly_report(df)

    assert "total_anomalies" in report
    assert report["total_anomalies"] >= 1
    assert "risk_index" in report
    assert "executive_summary" in report
    assert len(report["records"]) >= 1
