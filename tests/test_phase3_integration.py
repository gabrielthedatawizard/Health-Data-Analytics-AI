"""
Integration Tests for Phase 3 Predictive Intelligence Layer
"""

import numpy as np
import pandas as pd
import pytest
from backend.ai_analytics_service import (
    build_facts_bundle,
    ask_question,
    get_proactive_radar,
)


def test_phase3_facts_bundle_enrichment():
    # Construct 12 months of health facility data
    dates = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d").tolist()
    admissions = [100, 110, 115, 125, 130, 140, 150, 160, 175, 185, 190, 210]

    df = pd.DataFrame({
        "facility": [f"Hospital_{i % 3}" for i in range(12)],
        "reporting_date": dates,
        "admissions": admissions,
    })
    # Inject one negative value as an anomaly in another column
    df["deliveries"] = [25, 30, 28, -5, 35, 40, 38, 42, 45, 50, 48, 55]

    bundle = build_facts_bundle(df, session_id="s1", dataset_id="ds_p3_test")

    # 1. Verify Phase 3 Forecasting is populated in trends
    trends = bundle.get("trends", {})
    adm_trend = trends.get("admissions_monthly", {})
    assert "forecast" in adm_trend
    fc = adm_trend["forecast"]
    assert fc["horizon"] == 6
    assert len(fc["predictions"]) == 6
    assert "scenarios" in fc
    assert "explanation" in fc
    assert len(fc["explanation"]) > 20

    # 2. Verify Phase 3 Anomaly Report is populated in facts bundle
    assert "anomaly_report" in bundle
    anom_rep = bundle["anomaly_report"]
    assert anom_rep["total_anomalies"] >= 1
    assert "risk_index" in anom_rep

    # Row with deliveries = -5 should be flagged as critical data entry error
    flagged_rows = [r["row_index"] for r in anom_rep["records"]]
    assert 3 in flagged_rows
    deliv_anom = next(r for r in anom_rep["records"] if r["row_index"] == 3)
    assert deliv_anom["severity"] == "critical"
    assert deliv_anom["type"] == "data_entry_error"


def test_phase3_local_ask_forecasting_and_anomalies():
    dates = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({
        "date": dates,
        "cases": [50, 55, 60, 65, 70, 80, 85, 90, 95, 105, 115, 130],
        "deaths": [2, 3, 2, -1, 4, 3, 5, 4, 6, 5, 7, 8],
    })
    bundle = build_facts_bundle(df, session_id="s1", dataset_id="ds_ask_p3")

    # Ask about future / forecasting
    ans_forecast = ask_question("What is the projected forecast for the next 6 months?", bundle)
    assert "Forecast" in ans_forecast["answer"]
    assert "cases" in ans_forecast["answer"].lower() or "projection" in ans_forecast["answer"].lower()
    assert ans_forecast["chart_recommendation"]["type"] == "line"

    # Ask about anomalies / data audit
    ans_anom = ask_question("Are there any suspicious anomalies or data entry errors?", bundle)
    assert "Detected" in ans_anom["answer"] or "anomal" in ans_anom["answer"].lower()
    assert ans_anom["chart_recommendation"]["type"] == "table"


def test_phase3_proactive_radar_surfaces_critical_anomalies():
    df = pd.DataFrame({
        "facility": [f"Clinic_{i}" for i in range(15)],
        "visits": [100] * 14 + [-50],  # Critical negative count
    })
    bundle = build_facts_bundle(df, session_id="s1", dataset_id="ds_radar_p3")

    radar = get_proactive_radar(bundle)
    assert "alerts" in radar
    assert len(radar["alerts"]) >= 1

    # Should have a critical anomaly alert
    critical_alerts = [a for a in radar["alerts"] if a.get("severity") == "critical"]
    assert len(critical_alerts) >= 1
    assert "Data Entry Error" in critical_alerts[0]["title"] or "Anomaly" in critical_alerts[0]["title"]
