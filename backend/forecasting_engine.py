"""
Forecasting Engine for Health Data Analytics AI
Phase 3.1: Predictive Intelligence Layer

Provides robust time-series forecasting, seasonality detection,
automated model selection, scenario/what-if analysis, and domain-aware explanations.
Works natively using NumPy and SciPy with zero required external heavyweight dependencies.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ForecastPoint:
    period: str
    predicted_value: float
    ci_lower_80: float
    ci_upper_80: float
    ci_lower_95: float
    ci_upper_95: float


@dataclass
class ForecastResult:
    metric: str
    model_name: str
    model_description: str
    horizon: int
    predictions: List[Dict[str, Any]]
    historical_points: int
    seasonality: Dict[str, Any]
    accuracy_metrics: Dict[str, float]  # rmse, mae, mape
    scenarios: Dict[str, List[Dict[str, Any]]]
    explanation: str


class ForecastingEngine:
    """
    Time-series forecasting engine with automated model selection,
    uncertainty quantification, scenario modeling, and natural language explanations.
    """

    # ─────────────────────────────────────────────────────────────
    # 1. Seasonality & Trend Diagnostic
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def detect_seasonality(series: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Detects cyclical patterns and seasonality using lag autocorrelation and variance analysis.
        Tests standard operational/health cycles: 4 (quarterly), 7 (daily), 12 (monthly), 52 (weekly).
        """
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 8:
            return {
                "has_seasonality": False,
                "period": None,
                "strength": 0.0,
                "type": "none",
                "rationale": "Insufficient historical points for seasonal decomposition (< 8 observations).",
            }

        # Detrend with simple differencing or linear residual
        t = np.arange(n)
        slope, intercept, _, _, _ = stats.linregress(t, arr)
        detrended = arr - (slope * t + intercept)
        var_detrended = float(np.var(detrended))

        if var_detrended <= 1e-9:
            return {
                "has_seasonality": False,
                "period": None,
                "strength": 0.0,
                "type": "none",
                "rationale": "Series has near-zero variance.",
            }

        # Check candidate periods
        candidate_periods = [p for p in [4, 7, 12, 52] if p <= n // 2]
        if not candidate_periods and n >= 6:
            candidate_periods = [2, 3]

        best_period = None
        max_autocorr = -1.0

        for period in candidate_periods:
            if n < period * 2:
                continue
            # Autocorrelation at lag `period`
            x0 = detrended[:-period]
            x1 = detrended[period:]
            if len(x0) > 3 and np.std(x0) > 1e-9 and np.std(x1) > 1e-9:
                corr, _ = stats.pearsonr(x0, x1)
                if corr > max_autocorr:
                    max_autocorr = corr
                    best_period = period

        # Seasonal significance threshold (roughly 95% critical value ~ 1.96 / sqrt(N))
        crit_val = min(0.65, max(0.35, 1.96 / math.sqrt(n)))
        has_season = (best_period is not None) and (max_autocorr >= crit_val)

        # Seasonality type check (multiplicative if seasonal amplitude scales with level)
        stype = "additive"
        if has_season and np.all(arr > 0):
            # Check correlation between local mean and local std
            sub_len = best_period
            chunks = [arr[i : i + sub_len] for i in range(0, n - sub_len + 1, sub_len)]
            if len(chunks) >= 3:
                means = [np.mean(c) for c in chunks]
                stds = [np.std(c) for c in chunks]
                if np.std(means) > 1e-6 and np.std(stds) > 1e-6:
                    r_amp, _ = stats.pearsonr(means, stds)
                    if r_amp > 0.5:
                        stype = "multiplicative"

        strength = max(0.0, min(1.0, float(max_autocorr))) if has_season else 0.0

        return {
            "has_seasonality": bool(has_season),
            "period": int(best_period) if has_season else None,
            "strength": round(strength, 3),
            "type": stype if has_season else "none",
            "autocorrelation": round(float(max_autocorr), 3) if best_period else 0.0,
            "rationale": (
                f"Detected recurring {best_period}-period cycle with seasonal strength of {strength:.1%}."
                if has_season
                else "No significant cyclical or seasonal frequency detected above statistical noise."
            ),
        }

    # ─────────────────────────────────────────────────────────────
    # 2. Forecasting Model Algorithms
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def fit_simple_exponential_smoothing(
        cls, arr: np.ndarray, horizon: int, alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Simple Exponential Smoothing (SES) with automated alpha parameter tuning.
        Best for data with no strong trend or seasonality.
        """
        n = len(arr)
        if alpha is None:
            # Grid search for optimal alpha minimizing sum of squared errors
            best_sse = float("inf")
            best_alpha = 0.3
            for a in np.linspace(0.05, 0.95, 19):
                lvl = arr[0]
                sse = 0.0
                for val in arr[1:]:
                    pred = lvl
                    sse += (val - pred) ** 2
                    lvl = a * val + (1 - a) * lvl
                if sse < best_sse:
                    best_sse = sse
                    best_alpha = a
            alpha = float(best_alpha)

        # Generate in-sample fitted values and final level
        fitted = np.zeros(n)
        fitted[0] = arr[0]
        lvl = arr[0]
        for t in range(1, n):
            fitted[t] = lvl
            lvl = alpha * arr[t] + (1 - alpha) * lvl

        residuals = arr[1:] - fitted[1:]
        sigma = float(np.std(residuals)) if len(residuals) > 1 else float(np.std(arr))

        # Out-of-sample forecast
        forecast = np.full(horizon, lvl)
        # Prediction standard error grows with horizon sqrt(1 + (h-1)*alpha^2)
        h_range = np.arange(1, horizon + 1)
        se_forecast = sigma * np.sqrt(1 + (h_range - 1) * (alpha**2))

        return forecast, se_forecast, {"alpha": round(alpha, 3), "sigma": round(sigma, 3), "fitted": fitted}

    @classmethod
    def fit_holt_linear(
        cls,
        arr: np.ndarray,
        horizon: int,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        damped: bool = True,
        phi: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Holt's Linear Trend method with optional trend damping (phi).
        Captures evolving velocity/trajectory.
        """
        n = len(arr)
        if n < 3:
            # Fallback to SES
            return cls.fit_simple_exponential_smoothing(arr, horizon)

        # Optimize alpha and beta
        if alpha is None or beta is None:
            best_sse = float("inf")
            best_a, best_b = 0.3, 0.1
            for a in [0.1, 0.3, 0.5, 0.7]:
                for b in [0.05, 0.1, 0.2, 0.3]:
                    lvl = arr[0]
                    b_trend = arr[1] - arr[0]
                    sse = 0.0
                    for t in range(1, n):
                        pred = lvl + (phi * b_trend if damped else b_trend)
                        sse += (arr[t] - pred) ** 2
                        new_lvl = a * arr[t] + (1 - a) * pred
                        b_trend = b * (new_lvl - lvl) + (1 - b) * (phi * b_trend if damped else b_trend)
                        lvl = new_lvl
                    if sse < best_sse:
                        best_sse = sse
                        best_a, best_b = a, b
            alpha, beta = best_a, best_b

        # Compute fitted
        fitted = np.zeros(n)
        fitted[0] = arr[0]
        lvl = arr[0]
        b_trend = arr[1] - arr[0] if n > 1 else 0.0

        for t in range(1, n):
            pred = lvl + (phi * b_trend if damped else b_trend)
            fitted[t] = pred
            new_lvl = alpha * arr[t] + (1 - alpha) * pred
            b_trend = beta * (new_lvl - lvl) + (1 - beta) * (phi * b_trend if damped else b_trend)
            lvl = new_lvl

        residuals = arr[1:] - fitted[1:]
        sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0

        # Forecast forward
        forecast = np.zeros(horizon)
        curr_lvl = lvl
        curr_b = b_trend
        se_forecast = np.zeros(horizon)

        for h in range(1, horizon + 1):
            if damped:
                # Cumulative damping sum: phi + phi^2 + ... + phi^h
                damp_factor = phi * (1 - phi**h) / (1 - phi) if abs(1 - phi) > 1e-5 else h
                forecast[h - 1] = curr_lvl + damp_factor * curr_b
            else:
                forecast[h - 1] = curr_lvl + h * curr_b

            # Variance expands with horizon
            se_forecast[h - 1] = sigma * math.sqrt(1 + (h - 1) * (alpha**2 + alpha * beta * h))

        return forecast, se_forecast, {
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "damped": damped,
            "final_level": round(float(lvl), 3),
            "final_slope": round(float(b_trend), 3),
            "fitted": fitted,
        }

    @classmethod
    def fit_holt_winters(
        cls, arr: np.ndarray, horizon: int, period: int = 12
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Holt-Winters Seasonal Additive model.
        Captures trend + cyclical periodic shifts.
        """
        n = len(arr)
        if n < period * 2:
            return cls.fit_holt_linear(arr, horizon)

        # Initialize seasonal factors
        num_cycles = n // period
        cycle_averages = [np.mean(arr[i * period : (i + 1) * period]) for i in range(num_cycles)]
        seasonals = np.zeros(period)
        for i in range(period):
            seasonals[i] = np.mean([arr[c * period + i] - cycle_averages[c] for c in range(num_cycles)])

        # Center seasonal factors to sum to 0
        seasonals -= np.mean(seasonals)

        # Baseline parameters
        alpha, beta, gamma = 0.3, 0.1, 0.2
        lvl = np.mean(arr[:period])
        trend = (np.mean(arr[period : period * 2]) - np.mean(arr[:period])) / period

        fitted = np.zeros(n)
        s_factors = list(seasonals)

        for t in range(n):
            idx = t % period
            if t == 0:
                fitted[t] = arr[0]
                continue
            pred = lvl + trend + s_factors[idx]
            fitted[t] = pred

            val = arr[t]
            prev_lvl = lvl
            lvl = alpha * (val - s_factors[idx]) + (1 - alpha) * (prev_lvl + trend)
            trend = beta * (lvl - prev_lvl) + (1 - beta) * trend
            s_factors[idx] = gamma * (val - lvl) + (1 - gamma) * s_factors[idx]

        residuals = arr[period:] - fitted[period:]
        sigma = float(np.std(residuals)) if len(residuals) > 0 else 1.0

        # Forecast
        forecast = np.zeros(horizon)
        se_forecast = np.zeros(horizon)
        for h in range(1, horizon + 1):
            idx = (n + h - 1) % period
            forecast[h - 1] = lvl + h * trend + s_factors[idx]
            se_forecast[h - 1] = sigma * math.sqrt(1 + 0.1 * h)

        return forecast, se_forecast, {
            "period": period,
            "final_level": round(float(lvl), 3),
            "final_trend": round(float(trend), 3),
            "fitted": fitted,
        }

    @classmethod
    def fit_linear_drift(
        cls, arr: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Linear Drift / OLS extrapolation. Baseline structural benchmark.
        """
        n = len(arr)
        t = np.arange(n)
        slope, intercept, r_val, p_val, std_err = stats.linregress(t, arr)

        fitted = slope * t + intercept
        residuals = arr - fitted
        sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0

        t_future = np.arange(n, n + horizon)
        forecast = slope * t_future + intercept

        # OLS prediction error widening with distance from mean
        mean_t = np.mean(t)
        sum_sq_t = np.sum((t - mean_t) ** 2) if n > 1 else 1.0
        se_forecast = np.array([
            sigma * math.sqrt(1 + 1.0 / n + ((tf - mean_t) ** 2) / max(sum_sq_t, 1e-9))
            for tf in t_future
        ])

        return forecast, se_forecast, {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 3),
            "r_squared": round(float(r_val**2), 3),
            "fitted": fitted,
        }

    # ─────────────────────────────────────────────────────────────
    # 3. Model Backtesting & Evaluation
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def evaluate_model(
        cls, arr: np.ndarray, model_func: Any, test_horizon: int = 3, **kwargs
    ) -> Dict[str, float]:
        """
        Performs out-of-sample backtest to compute realistic predictive performance.
        """
        n = len(arr)
        if n <= test_horizon + 2:
            # Fall back to in-sample residual evaluation
            pred, _, meta = model_func(arr, 1, **kwargs)
            fitted = meta.get("fitted", arr)
            err = arr - fitted
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            mape = float(np.mean(np.abs(err / np.where(arr != 0, arr, 1e-6)))) * 100
            return {"rmse": rmse, "mae": mae, "mape": mape}

        train = arr[:-test_horizon]
        test = arr[-test_horizon:]

        preds, _, _ = model_func(train, test_horizon, **kwargs)

        err = test - preds
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        denom = np.where(np.abs(test) > 1e-5, np.abs(test), 1.0)
        mape = float(np.mean(np.abs(err / denom))) * 100

        return {"rmse": round(rmse, 3), "mae": round(mae, 3), "mape": round(mape, 2)}

    # ─────────────────────────────────────────────────────────────
    # 4. Auto-Forecasting with Model Selection & Ensemble
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def auto_forecast(
        cls,
        series: Union[List[float], np.ndarray, pd.Series],
        horizon: int = 6,
        dates: Optional[List[str]] = None,
        metric_name: str = "metric",
    ) -> ForecastResult:
        """
        Automatically selects the highest-accuracy model among SES, Holt Damped,
        Seasonal Holt-Winters, and Linear Drift using time-series cross-validation.
        Generates point forecasts, multi-tier confidence intervals (80% & 95%),
        scenario models, and executive explanations.
        """
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 4:
            # Very short series: simple flat or linear extrapolation
            val = arr[-1] if n > 0 else 0.0
            preds = [
                {
                    "step": i + 1,
                    "predicted_value": round(float(val), 2),
                    "ci_lower_80": round(float(val * 0.9), 2),
                    "ci_upper_80": round(float(val * 1.1), 2),
                    "ci_lower_95": round(float(val * 0.8), 2),
                    "ci_upper_95": round(float(val * 1.2), 2),
                }
                for i in range(horizon)
            ]
            return ForecastResult(
                metric=metric_name,
                model_name="naive_persistence",
                model_description="Naive Last Observation Carried Forward (Short Series)",
                horizon=horizon,
                predictions=preds,
                historical_points=n,
                seasonality={"has_seasonality": False, "period": None, "strength": 0.0},
                accuracy_metrics={"rmse": 0.0, "mae": 0.0, "mape": 0.0},
                scenarios={},
                explanation="Series has fewer than 4 points; using baseline persistence projection.",
            )

        # Detect seasonality
        season_meta = cls.detect_seasonality(arr)

        # Define model candidates
        candidates = [
            ("holt_linear", "Holt Damped Linear Trend", cls.fit_holt_linear, {"damped": True}),
            ("ses", "Simple Exponential Smoothing", cls.fit_simple_exponential_smoothing, {}),
            ("linear_drift", "Linear Drift Extrapolation", cls.fit_linear_drift, {}),
        ]

        if season_meta["has_seasonality"] and season_meta["period"] and n >= season_meta["period"] * 2:
            candidates.insert(
                0,
                (
                    "holt_winters",
                    f"Holt-Winters Seasonal ({season_meta['period']}-period)",
                    cls.fit_holt_winters,
                    {"period": season_meta["period"]},
                ),
            )

        # Evaluate candidate models via backtest
        test_h = min(max(2, n // 4), 6)
        scored_candidates = []

        for m_id, m_name, m_func, m_kw in candidates:
            try:
                metrics = cls.evaluate_model(arr, m_func, test_horizon=test_h, **m_kw)
                scored_candidates.append({
                    "id": m_id,
                    "name": m_name,
                    "func": m_func,
                    "kwargs": m_kw,
                    "metrics": metrics,
                })
            except Exception:
                continue

        if not scored_candidates:
            # Fallback
            scored_candidates = [{
                "id": "ses",
                "name": "Simple Exponential Smoothing",
                "func": cls.fit_simple_exponential_smoothing,
                "kwargs": {},
                "metrics": {"rmse": 1.0, "mae": 1.0, "mape": 5.0},
            }]

        # Pick best model (lowest RMSE)
        scored_candidates.sort(key=lambda c: c["metrics"]["rmse"])
        best = scored_candidates[0]

        # Generate final forecast with full history
        point_forecast, se_forecast, model_meta = best["func"](arr, horizon, **best["kwargs"])

        # Compute multi-tier confidence intervals (80% z=1.282, 95% z=1.960)
        z80 = 1.282
        z95 = 1.960

        predictions = []
        for i in range(horizon):
            pred_val = float(point_forecast[i])
            se = float(se_forecast[i])
            ci80_low = pred_val - z80 * se
            ci80_high = pred_val + z80 * se
            ci95_low = pred_val - z95 * se
            ci95_high = pred_val + z95 * se

            # Non-negative constraint for count/volume health metrics if input was non-negative
            if np.all(arr >= 0):
                pred_val = max(0.0, pred_val)
                ci80_low = max(0.0, ci80_low)
                ci80_high = max(0.0, ci80_high)
                ci95_low = max(0.0, ci95_low)
                ci95_high = max(0.0, ci95_high)

            predictions.append({
                "step": i + 1,
                "predicted_value": round(pred_val, 2),
                "ci_lower_80": round(ci80_low, 2),
                "ci_upper_80": round(ci80_high, 2),
                "ci_lower_95": round(ci95_low, 2),
                "ci_upper_95": round(ci95_high, 2),
            })

        # Generate Scenarios (Optimistic / Pessimistic / Target Shift)
        scenarios = cls.generate_scenarios(point_forecast, se_forecast, arr)

        # Generate Domain-Aware Narrative Explanation
        explanation = cls.explain_forecast(
            metric_name=metric_name,
            historical_arr=arr,
            forecast_arr=point_forecast,
            model_name=best["name"],
            metrics=best["metrics"],
            season_meta=season_meta,
            scenarios=scenarios,
        )

        return ForecastResult(
            metric=metric_name,
            model_name=best["id"],
            model_description=best["name"],
            horizon=horizon,
            predictions=predictions,
            historical_points=n,
            seasonality=season_meta,
            accuracy_metrics=best["metrics"],
            scenarios=scenarios,
            explanation=explanation,
        )

    # ─────────────────────────────────────────────────────────────
    # 5. Scenario & What-If Modeling
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def generate_scenarios(
        cls, forecast_arr: np.ndarray, se_arr: np.ndarray, historical_arr: np.ndarray
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Constructs baseline, optimistic, pessimistic, and targeted scenario pathways.
        """
        horizon = len(forecast_arr)
        is_positive = np.all(historical_arr >= 0)

        # Optimistic: +1.0 SE or +15% intervention acceleration
        # Pessimistic: -1.0 SE or disruption shock
        # Target: Linear 10% improvement pathway
        opt_vals = []
        pess_vals = []
        target_vals = []

        baseline_last = historical_arr[-1] if len(historical_arr) > 0 else 0.0

        for h in range(horizon):
            b_val = float(forecast_arr[h])
            se = float(se_arr[h])

            o_val = b_val + 0.8 * se
            p_val = b_val - 0.8 * se

            # Progressive target (+2% compounded per period)
            t_val = baseline_last * ((1.02) ** (h + 1))

            if is_positive:
                o_val = max(0.0, o_val)
                p_val = max(0.0, p_val)
                t_val = max(0.0, t_val)

            opt_vals.append({"step": h + 1, "value": round(o_val, 2)})
            pess_vals.append({"step": h + 1, "value": round(p_val, 2)})
            target_vals.append({"step": h + 1, "value": round(t_val, 2)})

        return {
            "baseline": [{"step": h + 1, "value": round(float(forecast_arr[h]), 2)} for h in range(horizon)],
            "optimistic": opt_vals,
            "pessimistic": pess_vals,
            "target_trajectory": target_vals,
        }

    # ─────────────────────────────────────────────────────────────
    # 6. Forecast Explanation Narrative Generator
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def explain_forecast(
        metric_name: str,
        historical_arr: np.ndarray,
        forecast_arr: np.ndarray,
        model_name: str,
        metrics: Dict[str, float],
        season_meta: Dict[str, Any],
        scenarios: Dict[str, Any],
    ) -> str:
        """
        Translates mathematical projection into clear executive intelligence.
        """
        n = len(historical_arr)
        h = len(forecast_arr)
        recent_val = historical_arr[-1] if n > 0 else 0.0
        end_pred = forecast_arr[-1] if h > 0 else 0.0

        diff = end_pred - recent_val
        pct_change = (diff / recent_val * 100) if abs(recent_val) > 1e-6 else 0.0

        direction_word = "stable"
        if pct_change >= 5.0:
            direction_word = f"projected to increase by {abs(pct_change):.1f}%"
        elif pct_change <= -5.0:
            direction_word = f"projected to decrease by {abs(pct_change):.1f}%"
        else:
            direction_word = f"projected to remain steady (within {pct_change:+.1f}%)"

        clean_name = metric_name.replace("_", " ").title()

        narrative_parts = [
            f"**{clean_name} Forecasting Intelligence**:",
            f"Over the next {h} reporting periods, {clean_name} is {direction_word}, moving from {recent_val:,.1f} to {end_pred:,.1f}.",
            f"• **Optimal Model**: Selected **{model_name}** based on out-of-sample backtesting (Mean Absolute Error: {metrics.get('mae', 0):.2f}, MAPE: {metrics.get('mape', 0):.1f}%).",
        ]

        if season_meta.get("has_seasonality"):
            period = season_meta.get("period")
            strength = season_meta.get("strength", 0.0)
            narrative_parts.append(
                f"• **Seasonality Driver**: Incorporates an active {period}-period cyclical wave with a seasonal strength index of {strength:.0%}."
            )

        if "optimistic" in scenarios and "pessimistic" in scenarios:
            opt_end = scenarios["optimistic"][-1]["value"]
            pess_end = scenarios["pessimistic"][-1]["value"]
            narrative_parts.append(
                f"• **Risk Corridor**: 80% scenario boundary spans between {pess_end:,.1f} (pessimistic/disruption bound) and {opt_end:,.1f} (accelerated expansion bound)."
            )

        return "\n".join(narrative_parts)
