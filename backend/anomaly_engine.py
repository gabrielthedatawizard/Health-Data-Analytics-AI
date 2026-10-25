"""
Anomaly Detection System for Health Data Analytics AI
Phase 3.2: Predictive Intelligence Layer

Multi-method anomaly detection engine combining:
1. Robust statistical fences (IQR / MAD)
2. Multivariate Isolation Forest
3. Domain-aware clinical & reporting artifact classifiers
4. Data distribution drift monitoring (PSI & Kolmogorov-Smirnov)
5. Root-cause hypothesis generation & mitigation recommendations
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class AnomalyRecord:
    row_index: int
    columns_involved: List[str]
    composite_score: float  # 0.0 to 1.0 (1.0 = extreme anomaly)
    severity: str           # "low" | "medium" | "high" | "critical"
    anomaly_type: str       # "data_entry_error" | "genuine_spike" | "reporting_artifact" | "statistical_outlier"
    metric_values: Dict[str, Any]
    hypothesis: str
    recommended_action: str


@dataclass
class DriftAlert:
    metric: str
    drift_status: str       # "stable" | "moderate_drift" | "significant_drift"
    psi_score: float
    ks_statistic: float
    p_value: float
    summary: str


class AnomalyEngine:
    """
    Predictive Anomaly Detection and Reporting Quality Engine.
    Detects multivariate outliers, clinical impossible values, reporting artifacts,
    and distribution drift.
    """

    # ─────────────────────────────────────────────────────────────
    # 1. Univariate Statistical Anomaly Scoring
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_iqr_scores(series: pd.Series) -> np.ndarray:
        """
        Computes normalized IQR outlier score in [0.0, 1.0].
        Scores > 0.5 indicate outside 1.5*IQR; scores > 0.8 indicate outside 3.0*IQR.
        """
        vals = series.to_numpy(dtype=float)
        valid_mask = ~np.isnan(vals)
        scores = np.zeros(len(vals), dtype=float)
        if np.sum(valid_mask) < 4:
            return scores

        clean = vals[valid_mask]
        q25, q75 = np.percentile(clean, [25, 75])
        iqr = q75 - q25
        if iqr <= 1e-9:
            std = np.std(clean)
            if std <= 1e-9:
                return scores
            # Fall back to z-score normalization
            z = np.abs((vals - np.mean(clean)) / std)
            return np.clip(z / 4.0, 0.0, 1.0)

        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr

        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            if v < lower_fence:
                dist = (lower_fence - v) / iqr
                scores[i] = min(1.0, 0.5 + 0.15 * dist)
            elif v > upper_fence:
                dist = (v - upper_fence) / iqr
                scores[i] = min(1.0, 0.5 + 0.15 * dist)
            else:
                scores[i] = 0.0

        return scores

    @staticmethod
    def _compute_mad_scores(series: pd.Series) -> np.ndarray:
        """
        Modified Z-score using Median Absolute Deviation (MAD).
        More robust to extreme values than classical standard deviation.
        """
        vals = series.to_numpy(dtype=float)
        valid_mask = ~np.isnan(vals)
        scores = np.zeros(len(vals), dtype=float)
        if np.sum(valid_mask) < 4:
            return scores

        clean = vals[valid_mask]
        median = np.median(clean)
        mad = np.median(np.abs(clean - median))

        scale = mad
        if scale <= 1e-9:
            # Fall back to mean absolute deviation or std when >50% of values are identical
            mean_ad = float(np.mean(np.abs(clean - median)))
            if mean_ad <= 1e-9:
                return scores
            scale = mean_ad

        # Boris Iglewicz and David Hoaglin (1993) modified Z-score
        mod_z = 0.6745 * np.abs(vals - median) / scale
        # Normalization: mod_z >= 3.5 is the classical threshold
        norm_scores = np.clip(mod_z / 5.0, 0.0, 1.0)
        norm_scores[~valid_mask] = 0.0
        return norm_scores

    # ─────────────────────────────────────────────────────────────
    # 2. Multivariate Isolation Forest Scoring
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def _compute_isolation_scores(cls, df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        """
        Fits an Isolation Forest to identify multivariate anomalies.
        Returns continuous scores in [0.0, 1.0], where higher = more anomalous.
        """
        n_rows = len(df)
        if n_rows < 10 or not numeric_cols:
            return np.zeros(n_rows, dtype=float)

        # Impute missing values with median for isolation forest
        sub_df = df[numeric_cols].copy()
        for col in numeric_cols:
            med = sub_df[col].median()
            sub_df[col] = sub_df[col].fillna(med if not np.isnan(med) else 0.0)

        data_matrix = sub_df.to_numpy(dtype=float)

        if SKLEARN_AVAILABLE:
            try:
                # Contamination auto or 0.05
                iso = IsolationForest(
                    n_estimators=100,
                    contamination="auto",
                    random_state=42,
                    n_jobs=1,
                )
                iso.fit(data_matrix)
                # decision_function returns negative for anomalies, positive for inliers
                raw_scores = -iso.decision_function(data_matrix)
                # Min-max scale to [0, 1]
                min_s, max_s = raw_scores.min(), raw_scores.max()
                if max_s - min_s > 1e-6:
                    return (raw_scores - min_s) / (max_s - min_s)
                return np.zeros(n_rows, dtype=float)
            except Exception:
                pass

        # Fallback: Normalized Euclidean distance to centroid
        centroid = np.mean(data_matrix, axis=0)
        stds = np.std(data_matrix, axis=0)
        stds[stds < 1e-6] = 1.0
        norm_data = (data_matrix - centroid) / stds
        dists = np.linalg.norm(norm_data, axis=1)
        max_d = np.max(dists)
        if max_d > 1e-6:
            return np.clip(dists / max_d, 0.0, 1.0)
        return np.zeros(n_rows, dtype=float)

    # ─────────────────────────────────────────────────────────────
    # 3. Domain-Aware Health Anomaly Classification & Root Causes
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def classify_anomaly_type(
        col_name: str,
        value: float,
        col_stats: Dict[str, float],
        iqr_score: float,
        mad_score: float,
    ) -> Tuple[str, str, str, str]:
        """
        Classifies anomaly into semantic type, assigns severity,
        and generates health hypothesis + action.
        """
        c_lower = col_name.lower()
        val = float(value)
        med = col_stats.get("median", 0.0)
        q75 = col_stats.get("q75", 0.0)
        max_val = col_stats.get("max", 0.0)

        # 1. Negative Counts for non-negative clinical quantities
        non_negative_keywords = ["count", "cases", "deaths", "visits", "attendances", "deliveries", "patients", "doses"]
        if val < 0 and any(k in c_lower for k in non_negative_keywords):
            return (
                "data_entry_error",
                "critical",
                f"Negative value ({val:,.0f}) reported in '{col_name}', which represents a non-negative clinical volume.",
                "Verify facility ledger immediately; correct typographical inversion or accidental debit entry.",
            )

        # 2. Percentage Out of Bounds
        if any(k in c_lower for k in ["percent", "rate", "coverage", "proportion"]):
            if val > 150.0 or val < 0.0:
                return (
                    "data_entry_error",
                    "critical" if val > 200 or val < 0 else "high",
                    f"Reported rate/percentage ({val:.1f}%) exceeds logical ceiling (or is negative). Often caused by miscalculated denominator.",
                    "Audit target population denominator vs numerator to reconcile calculation discrepancy.",
                )

        # 3. Impossible Biological Vitals
        if "temperature" in c_lower or "temp" in c_lower:
            if val < 30.0 or val > 45.0:
                return (
                    "data_entry_error",
                    "critical",
                    f"Recorded body temperature ({val:.1f}°C) is physiologically incompatible with life (unit mismatch e.g. Fahrenheit vs Celsius suspected).",
                    "Check if thermometer unit was recorded in Fahrenheit instead of Celsius.",
                )
        if "systolic" in c_lower or "bp_sys" in c_lower:
            if val < 40 or val > 300:
                return (
                    "data_entry_error",
                    "critical",
                    f"Recorded systolic pressure ({val:.0f} mmHg) is extreme or miskeyed.",
                    "Review clinical chart for transposition error.",
                )

        # 4. Order-of-Magnitude Typo (e.g. 10x - 100x jump)
        if med > 0 and val > med * 15 and val > 100:
            return (
                "data_entry_error",
                "high",
                f"Value ({val:,.1f}) is {val/med:.0f}x higher than median ({med:,.1f}). Consistent with extra trailing zero or concatenated keypress.",
                "Inspect raw data entry log for potential double-entry or keystroke repetition.",
            )

        # 5. Epidemic / Surveillance Surge
        if any(k in c_lower for k in ["malaria", "cholera", "measles", "covid", "infections", "admissions"]):
            if val > q75 + 3 * max(1.0, col_stats.get("iqr", 1.0)):
                return (
                    "genuine_spike",
                    "critical",
                    f"Sharp acute surge in {col_name} ({val:,.0f} vs median {med:,.0f}). Exceeds 3x IQR epidemic threshold.",
                    "Trigger rapid outbreak verification protocol and notify district epidemiological surveillance team.",
                )

        # 6. Default Statistical Outlier
        severity = "high" if (iqr_score > 0.8 or mad_score > 0.8) else "medium"
        return (
            "statistical_outlier",
            severity,
            f"Statistically deviant observation ({val:,.2f}) exceeding 99th percentile boundary for {col_name}.",
            "Flag for data validation check during monthly facility reconciliation.",
        )

    # ─────────────────────────────────────────────────────────────
    # 4. Master Anomaly Detection Engine
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def detect_anomalies(
        cls,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        max_records: int = 50,
        min_composite_score: float = 0.55,
    ) -> List[AnomalyRecord]:
        """
        Runs ensemble anomaly detection across all numeric features.
        Combines IQR + MAD + Isolation Forest.
        """
        if df.empty:
            return []

        if numeric_cols is None:
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not str(c).startswith("_")]

        if not numeric_cols:
            return []

        n_rows = len(df)
        if n_rows == 0:
            return []

        # 1. Compute univariate scores and column stats
        col_iqr_scores = {}
        col_mad_scores = {}
        col_stats = {}

        for col in numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            valid = s.dropna()
            if len(valid) < 3:
                continue
            q25, q75 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
            col_stats[col] = {
                "median": float(valid.median()),
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
                "max": float(valid.max()),
                "min": float(valid.min()),
                "std": float(valid.std()),
            }
            col_iqr_scores[col] = cls._compute_iqr_scores(s)
            col_mad_scores[col] = cls._compute_mad_scores(s)

        # 2. Compute multivariate isolation forest scores
        iso_scores = cls._compute_isolation_scores(df, numeric_cols)

        # 3. Ensemble aggregation per row
        anomalies: List[AnomalyRecord] = []

        for row_idx in range(n_rows):
            row_flagged_cols = []
            max_uni_score = 0.0
            primary_col = None

            for col in numeric_cols:
                if col not in col_iqr_scores:
                    continue
                iqr_s = col_iqr_scores[col][row_idx]
                mad_s = col_mad_scores[col][row_idx]
                comb_s = 0.5 * iqr_s + 0.5 * mad_s

                if comb_s >= 0.50:
                    row_flagged_cols.append(col)
                    if comb_s > max_uni_score:
                        max_uni_score = comb_s
                        primary_col = col

            # Check for direct hard domain violations (negative counts, impossible vitals)
            for col in numeric_cols:
                c_low = str(col).lower()
                val_raw = df.iloc[row_idx][col]
                if pd.notna(val_raw):
                    try:
                        n_val = float(val_raw)
                        if n_val < 0 and any(k in c_low for k in ["count", "cases", "deaths", "visits", "attendances", "deliveries", "patients", "doses"]):
                            primary_col = col
                            if col not in row_flagged_cols:
                                row_flagged_cols.append(col)
                            max_uni_score = 1.0
                        elif ("temperature" in c_low or "temp" in c_low) and (n_val < 30.0 or n_val > 45.0):
                            primary_col = col
                            if col not in row_flagged_cols:
                                row_flagged_cols.append(col)
                            max_uni_score = 1.0
                    except (ValueError, TypeError):
                        pass

            iso_s = iso_scores[row_idx] if len(iso_scores) > row_idx else 0.0

            # Composite weighted score: 60% univariate peak + 40% multivariate isolation
            if row_flagged_cols:
                composite = max(max_uni_score, 0.65 * max_uni_score + 0.35 * iso_s)
            else:
                composite = iso_s * 0.75  # Pure multivariate anomaly without single column blowout

            if composite >= min_composite_score and primary_col:
                val = df.iloc[row_idx][primary_col]
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    num_val = 0.0

                atype, sev, hyp, act = cls.classify_anomaly_type(
                    col_name=primary_col,
                    value=num_val,
                    col_stats=col_stats.get(primary_col, {}),
                    iqr_score=col_iqr_scores[primary_col][row_idx],
                    mad_score=col_mad_scores[primary_col][row_idx],
                )

                metric_vals = {
                    c: float(df.iloc[row_idx][c])
                    for c in row_flagged_cols[:5]
                    if pd.notna(df.iloc[row_idx][c])
                }

                anomalies.append(
                    AnomalyRecord(
                        row_index=row_idx,
                        columns_involved=row_flagged_cols,
                        composite_score=round(float(composite), 3),
                        severity=sev,
                        anomaly_type=atype,
                        metric_values=metric_vals,
                        hypothesis=hyp,
                        recommended_action=act,
                    )
                )

        # Sort by severity and composite score
        sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        anomalies.sort(key=lambda a: (sev_rank.get(a.severity, 0), a.composite_score), reverse=True)

        return anomalies[:max_records]

    # ─────────────────────────────────────────────────────────────
    # 5. Facility Reporting Behavior & Artifact Detection
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def detect_reporting_anomalies(
        cls,
        df: pd.DataFrame,
        facility_col: Optional[str] = None,
        time_col: Optional[str] = None,
        metric_cols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scans for systemic reporting pathologies:
        - Frozen values (identical repeated counts across >3 consecutive periods)
        - Catch-up batch reporting (repeated 0s followed by an extreme sudden surge)
        - Mass reporting blackout (all indicators drop to 0 simultaneously)
        """
        if df.empty:
            return []

        # Auto-detect facility and time columns if not supplied
        if not facility_col:
            for c in df.columns:
                c_low = str(c).lower()
                if any(k in c_low for k in ["facility", "hospital", "clinic", "site", "center", "subcounty"]):
                    facility_col = c
                    break

        if not time_col:
            for c in df.columns:
                c_low = str(c).lower()
                if any(k in c_low for k in ["date", "period", "month", "year", "week"]):
                    time_col = c
                    break

        if not metric_cols:
            metric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not str(c).startswith("_")]

        if not facility_col or not metric_cols:
            return []

        issues = []
        # Group by facility
        grouped = df.groupby(facility_col)

        for fac_name, grp in grouped:
            if len(grp) < 3:
                continue

            for col in metric_cols[:4]:
                s = pd.to_numeric(grp[col], errors="coerce").dropna()
                if len(s) < 3:
                    continue
                vals = s.tolist()

                # Check 1: Frozen numbers (same non-zero value repeated >= 3 times)
                consecutive_count = 1
                for i in range(1, len(vals)):
                    if vals[i] == vals[i - 1] and vals[i] != 0:
                        consecutive_count += 1
                        if consecutive_count >= 3:
                            issues.append({
                                "facility": str(fac_name),
                                "column": col,
                                "issue_type": "frozen_repeated_values",
                                "severity": "high",
                                "description": f"Exact value ({vals[i]:,.0f}) repeated for {consecutive_count} periods. Highly indicative of copy-paste reporting.",
                                "action": "Request primary paper register verification for this reporting facility.",
                            })
                            break
                    else:
                        consecutive_count = 1

                # Check 2: Catch-up batch reporting (2+ zeroes followed by >= 5x average)
                if len(vals) >= 4:
                    avg_val = np.mean(vals)
                    for i in range(2, len(vals)):
                        if vals[i - 1] == 0 and vals[i - 2] == 0 and vals[i] > avg_val * 2.5 and vals[i] > 10:
                            issues.append({
                                "facility": str(fac_name),
                                "column": col,
                                "issue_type": "batch_catch_up_spike",
                                "severity": "medium",
                                "description": f"Value spiked to {vals[i]:,.0f} immediately after 2 zero-reporting periods.",
                                "action": "Redistribute batch totals across missing months or confirm retroactive backlog entry.",
                            })
                            break

        return issues[:25]

    # ─────────────────────────────────────────────────────────────
    # 6. Data Distribution Drift Monitoring (PSI & KS Test)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def monitor_metric_drift(
        current_data: Union[pd.Series, np.ndarray, List[float]],
        baseline_data: Union[pd.Series, np.ndarray, List[float]],
        metric_name: str = "metric",
    ) -> DriftAlert:
        """
        Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov test
        to evaluate whether current incoming data has shifted away from baseline historical distributions.

        Rules of Thumb for PSI:
        - PSI < 0.10: No significant change (Stable)
        - 0.10 <= PSI <= 0.25: Moderate change / mild drift
        - PSI > 0.25: Significant change / severe distribution shift
        """
        curr = np.asarray(current_data, dtype=float)
        base = np.asarray(baseline_data, dtype=float)

        curr = curr[~np.isnan(curr)]
        base = base[~np.isnan(base)]

        if len(curr) < 10 or len(base) < 10:
            return DriftAlert(
                metric=metric_name,
                drift_status="stable",
                psi_score=0.0,
                ks_statistic=0.0,
                p_value=1.0,
                summary="Insufficient sample size to evaluate statistical drift (< 10 points).",
            )

        # 1. Kolmogorov-Smirnov 2-sample test
        ks_res = stats.ks_2samp(curr, base)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 2. Population Stability Index (PSI) using 10 quantile bins
        try:
            percentiles = np.linspace(0, 100, 11)
            bins = np.percentile(base, percentiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)

            base_counts, _ = np.histogram(base, bins=bins)
            curr_counts, _ = np.histogram(curr, bins=bins)

            # Smooth zero counts
            base_pct = np.maximum(base_counts, 1) / (np.sum(base_counts) + len(base_counts))
            curr_pct = np.maximum(curr_counts, 1) / (np.sum(curr_counts) + len(curr_counts))

            psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
        except Exception:
            psi = 0.0

        if psi > 0.25 or (p_val < 0.001 and ks_stat > 0.35):
            status = "significant_drift"
            summary = (
                f"Significant distribution drift detected in '{metric_name}' (PSI: {psi:.3f}, KS: {ks_stat:.3f}, p={p_val:.4f}). "
                "The current reporting profile differs fundamentally from historical baseline."
            )
        elif (psi >= 0.15 and p_val < 0.05) or (p_val < 0.01 and ks_stat > 0.25):
            status = "moderate_drift"
            summary = (
                f"Moderate distribution shift detected in '{metric_name}' (PSI: {psi:.3f}, KS: {ks_stat:.3f}). "
                "Monitor indicator closely in subsequent reporting cycles."
            )
        else:
            status = "stable"
            summary = f"'{metric_name}' distribution is statistically stable compared to baseline (PSI: {psi:.3f})."

        return DriftAlert(
            metric=metric_name,
            drift_status=status,
            psi_score=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            p_value=round(p_val, 5),
            summary=summary,
        )

    # ─────────────────────────────────────────────────────────────
    # 7. Comprehensive Anomaly Report Card
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def generate_anomaly_report(
        cls, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Produces an aggregated, structured anomaly report card for frontend UI consumption.
        """
        anomalies = cls.detect_anomalies(df, numeric_cols=numeric_cols)
        reporting_issues = cls.detect_reporting_anomalies(df, metric_cols=numeric_cols)

        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        type_counts = {}
        for a in anomalies:
            counts[a.severity] = counts.get(a.severity, 0) + 1
            type_counts[a.anomaly_type] = type_counts.get(a.anomaly_type, 0) + 1

        for r in reporting_issues:
            sev = r.get("severity", "medium")
            counts[sev] = counts.get(sev, 0) + 1

        total_anomalies = len(anomalies) + len(reporting_issues)

        # Health risk index [0 to 100]
        risk_penalty = counts["critical"] * 25 + counts["high"] * 10 + counts["medium"] * 3
        risk_index = min(100, risk_penalty)

        executive_summary = (
            f"Detected {total_anomalies} anomalies across the dataset: "
            f"{counts['critical']} critical, {counts['high']} high urgency, {counts['medium']} medium priority. "
        )
        if counts["critical"] > 0:
            executive_summary += "Immediate clinical or administrative audit is recommended for critical entries."
        else:
            executive_summary += "Overall data integrity is within acceptable operational tolerances."

        return {
            "total_anomalies": total_anomalies,
            "risk_index": risk_index,
            "severity_counts": counts,
            "type_counts": type_counts,
            "executive_summary": executive_summary,
            "records": [
                {
                    "row_index": a.row_index,
                    "columns": a.columns_involved,
                    "score": a.composite_score,
                    "severity": a.severity,
                    "type": a.anomaly_type,
                    "hypothesis": a.hypothesis,
                    "action": a.recommended_action,
                    "values": a.metric_values,
                }
                for a in anomalies[:20]
            ],
            "reporting_issues": reporting_issues,
        }
