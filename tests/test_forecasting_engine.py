"""
Unit Tests for Phase 3.1 Forecasting Engine
"""

import math
import numpy as np
import pytest
from backend.forecasting_engine import (
    ForecastingEngine,
    ForecastResult,
)


def test_seasonality_detection_cyclical():
    # Construct 24 months of data with strong annual 12-month seasonality
    t = np.arange(24)
    # Sine wave with period 12
    seasonal_pattern = 100 + 10 * t + 30 * np.sin(2 * np.pi * t / 12)
    diag = ForecastingEngine.detect_seasonality(seasonal_pattern)

    assert diag["has_seasonality"] is True
    assert diag["period"] == 12
    assert diag["strength"] > 0.3
    assert diag["type"] in ("additive", "multiplicative")


def test_seasonality_detection_flat_noise():
    np.random.seed(42)
    noise = np.random.normal(50, 2, 20)
    diag = ForecastingEngine.detect_seasonality(noise)

    assert diag["has_seasonality"] is False
    assert diag["period"] is None


def test_simple_exponential_smoothing():
    arr = np.array([10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0])
    forecast, se, meta = ForecastingEngine.fit_simple_exponential_smoothing(arr, horizon=4)

    assert len(forecast) == 4
    assert len(se) == 4
    assert "alpha" in meta
    # SES forecast should be flat level
    assert np.all(np.isclose(forecast, forecast[0]))
    # Standard error should grow or stay >= initial
    assert se[-1] >= se[0]


def test_holt_linear_trend():
    # Pure linear upward trend: 10, 20, 30, 40, 50, 60, 70, 80
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    forecast, se, meta = ForecastingEngine.fit_holt_linear(arr, horizon=3, damped=False)

    assert len(forecast) == 3
    # Next should project upward
    assert forecast[0] > arr[-1]
    assert forecast[1] > forecast[0]
    assert forecast[2] > forecast[1]
    assert meta["final_slope"] > 0


def test_holt_winters_seasonal():
    # 24 periods with period 12
    t = np.arange(24)
    arr = 50 + 2 * t + 15 * np.sin(2 * np.pi * t / 12)
    forecast, se, meta = ForecastingEngine.fit_holt_winters(arr, horizon=6, period=12)

    assert len(forecast) == 6
    assert meta["period"] == 12
    assert not np.isnan(forecast).any()


def test_auto_forecast_end_to_end():
    # Monthly facility admissions data with upward trajectory
    monthly_data = [120, 128, 135, 142, 138, 150, 155, 162, 170, 168, 180, 192]
    res: ForecastResult = ForecastingEngine.auto_forecast(
        series=monthly_data,
        horizon=6,
        metric_name="facility_admissions",
    )

    assert res.metric == "facility_admissions"
    assert res.horizon == 6
    assert len(res.predictions) == 6

    # Verify prediction intervals logic: CI_95_low <= CI_80_low <= point <= CI_80_high <= CI_95_high
    for p in res.predictions:
        pt = p["predicted_value"]
        assert p["ci_lower_95"] <= p["ci_lower_80"] <= pt or math.isclose(p["ci_lower_80"], pt, rel_tol=0.1)
        assert pt <= p["ci_upper_80"] <= p["ci_upper_95"] or math.isclose(p["ci_upper_80"], pt, rel_tol=0.1)

    # Scenarios exist
    assert "baseline" in res.scenarios
    assert "optimistic" in res.scenarios
    assert "pessimistic" in res.scenarios

    # Explanation text generated
    assert "Facility Admissions" in res.explanation
    assert len(res.explanation) > 50


def test_auto_forecast_short_series_graceful():
    short = [50.0, 52.0]
    res = ForecastingEngine.auto_forecast(short, horizon=3, metric_name="beds")
    assert res.horizon == 3
    assert len(res.predictions) == 3
    assert res.model_name == "naive_persistence"
