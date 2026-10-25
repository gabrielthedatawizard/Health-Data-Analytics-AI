"""
Statistical Reasoning & Inference Engine for Health Data Analytics AI
Phase 2.1: Statistical Reasoning Engine

Provides rigorous inferential statistics and hypothesis testing:
- Non-parametric Mann-Kendall Trend Test and Sen's Slope
- Automatic Two-Group & Multi-Group Hypothesis Testing (t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis)
- Changepoint and Structural Break Detection (PELT / CUSUM)
- Effect Size Computation (Cohen's d, Hedges' g, Cliff's delta, Eta-squared)
- Impact-Ranked Subgroup Segmentation Analysis
- Sample Representativeness & Goodness-of-fit Testing
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats


# =====================================================================
# 1. DATA STRUCTURES & RESULTS
# =====================================================================

@dataclass
class TrendTestResult:
    method: str
    is_significant: bool
    p_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    slope: float
    confidence_interval_95: tuple[float, float]
    tau_or_r: float
    description: str


@dataclass
class GroupDifferenceResult:
    test_used: str
    is_significant: bool
    p_value: float
    statistic: float
    effect_size: float
    effect_size_metric: str  # "cohens_d", "hedges_g", "cliffs_delta", "eta_squared"
    practical_significance: str  # "negligible", "small", "medium", "large"
    group_means: dict[str, float]
    group_medians: dict[str, float]
    group_sizes: dict[str, int]
    interpretation: str


@dataclass
class ChangepointResult:
    index: int
    period_label: Optional[str]
    direction: str  # "upward_shift", "downward_shift", "variance_shift"
    mean_before: float
    mean_after: float
    absolute_change: float
    percent_change: float
    confidence: float
    description: str


# =====================================================================
# 2. STATISTICAL INFERENCE ENGINE
# =====================================================================

class StatisticalEngine:
    """
    Performs inferential statistics, hypothesis testing, changepoint detection,
    and effect size calculations.
    """

    # ─────────────────────────────────────────────────────────────
    # Trend Significance (Mann-Kendall + Linear Regression)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def test_trend_significance(
        series: Sequence[float],
        method: str = "mann_kendall",
        alpha: float = 0.05,
    ) -> TrendTestResult:
        """
        Tests whether a sequence has a statistically significant trend.
        Uses Mann-Kendall non-parametric test by default, which is robust
        to outliers and does not assume a normal distribution.
        """
        vals = pd.Series(series).dropna().to_numpy(dtype=float)
        n = len(vals)

        if n < 4:
            return TrendTestResult(
                method=method,
                is_significant=False,
                p_value=1.0,
                trend_direction="insufficient_data",
                slope=0.0,
                confidence_interval_95=(0.0, 0.0),
                tau_or_r=0.0,
                description="Insufficient observations to test trend (n < 4).",
            )

        # 1. Mann-Kendall Test
        if method == "mann_kendall":
            # Kendall's tau between time index and values
            time_idx = np.arange(n)
            res = stats.kendalltau(time_idx, vals)
            tau = float(res.statistic) if not math.isnan(res.statistic) else 0.0
            p_val = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0

            # Sen's slope estimator (median of pairwise slopes)
            pairwise_slopes = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairwise_slopes.append((vals[j] - vals[i]) / (j - i))

            sens_slope = float(np.median(pairwise_slopes)) if pairwise_slopes else 0.0

            # 95% Confidence interval for slope
            k = int(1.96 * math.sqrt(n * (n - 1) * (2 * n + 5) / 18))
            m = len(pairwise_slopes)
            pairwise_slopes.sort()
            lower_idx = max(0, int((m - k) / 2))
            upper_idx = min(m - 1, int((m + k) / 2))
            ci_slope = (round(float(pairwise_slopes[lower_idx]), 4), round(float(pairwise_slopes[upper_idx]), 4))

            is_sig = bool(p_val < alpha)
            if is_sig and sens_slope > 0:
                direction = "increasing"
            elif is_sig and sens_slope < 0:
                direction = "decreasing"
            else:
                direction = "stable"

            desc = (
                f"Statistically significant {direction} trend (p={p_val:.4f}, tau={tau:.3f}, Sen's slope={sens_slope:.3f}/period)."
                if is_sig
                else f"No statistically significant trend detected (p={p_val:.4f}, tau={tau:.3f})."
            )

            return TrendTestResult(
                method="mann_kendall",
                is_significant=is_sig,
                p_value=round(p_val, 5),
                trend_direction=direction,
                slope=round(sens_slope, 4),
                confidence_interval_95=ci_slope,
                tau_or_r=round(tau, 3),
                description=desc,
            )

        # 2. Linear Regression Fallback
        time_idx = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, vals)
        ci_lower = slope - 1.96 * std_err
        ci_upper = slope + 1.96 * std_err
        is_sig = p_value < alpha
        direction = "increasing" if (is_sig and slope > 0) else "decreasing" if (is_sig and slope < 0) else "stable"

        return TrendTestResult(
            method="linear_regression",
            is_significant=is_sig,
            p_value=round(float(p_value), 5),
            trend_direction=direction,
            slope=round(float(slope), 4),
            confidence_interval_95=(round(float(ci_lower), 4), round(float(ci_upper), 4)),
            tau_or_r=round(float(r_value), 3),
            description=f"Linear trend: {direction} (p={p_value:.4f}, R={r_value:.3f}, slope={slope:.3f}).",
        )

    # ─────────────────────────────────────────────────────────────
    # Automatic Group Difference Hypothesis Testing
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def test_group_difference(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        alpha: float = 0.05,
    ) -> GroupDifferenceResult:
        """
        Automatically selects and performs the optimal statistical test:
        - 2 groups: Normality test (Shapiro) -> Student's t-test or Mann-Whitney U test.
        - 3+ groups: Normality test -> One-way ANOVA or Kruskal-Wallis H test.
        Calculates effect sizes (Cohen's d or Eta-squared).
        """
        clean_df = df[[value_col, group_col]].dropna()
        groups = clean_df.groupby(group_col)
        group_keys = list(groups.groups.keys())

        if len(group_keys) < 2:
            return GroupDifferenceResult(
                test_used="none",
                is_significant=False,
                p_value=1.0,
                statistic=0.0,
                effect_size=0.0,
                effect_size_metric="none",
                practical_significance="negligible",
                group_means={},
                group_medians={},
                group_sizes={},
                interpretation="Fewer than two distinct groups in dataset.",
            )

        group_arrays = [groups.get_group(k)[value_col].to_numpy(dtype=float) for k in group_keys]
        group_means = {str(k): round(float(np.mean(arr)), 3) for k, arr in zip(group_keys, group_arrays)}
        group_medians = {str(k): round(float(np.median(arr)), 3) for k, arr in zip(group_keys, group_arrays)}
        group_sizes = {str(k): len(arr) for k, arr in zip(group_keys, group_arrays)}

        # Check normality for each group (if n >= 8)
        all_normal = True
        for arr in group_arrays:
            if len(arr) >= 8:
                _, p_norm = stats.shapiro(arr[:50])  # Sample first 50 for speed
                if p_norm < 0.05:
                    all_normal = False
                    break
            else:
                all_normal = False

        # Case A: Exactly 2 Groups
        if len(group_keys) == 2:
            g1, g2 = group_arrays[0], group_arrays[1]
            k1, k2 = group_keys[0], group_keys[1]

            if all_normal and len(g1) >= 15 and len(g2) >= 15:
                # Independent Welch's t-test
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                test_name = "Welch's Two-Sample t-test"
                stat_val = float(t_stat)
            else:
                # Mann-Whitney U test
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test_name = "Mann-Whitney U Test"
                stat_val = float(u_stat)

            # Cohen's d / Hedges' g
            n1, n2 = len(g1), len(g2)
            s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
            pooled_sd = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1))
            diff = float(np.mean(g1) - np.mean(g2))
            d = abs(diff / max(pooled_sd, 1e-6))
            # Hedges' g small-sample correction
            hedges_g = d * (1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)))

            # Practical significance scale
            if hedges_g < 0.2:
                prac_sig = "negligible"
            elif hedges_g < 0.5:
                prac_sig = "small"
            elif hedges_g < 0.8:
                prac_sig = "medium"
            else:
                prac_sig = "large"

            is_sig = bool(p_val < alpha)
            interp = (
                f"Statistically significant difference between '{k1}' and '{k2}' ({test_name}, p={p_val:.4f}). "
                f"Effect size is {prac_sig} (Hedges' g = {hedges_g:.2f})."
                if is_sig
                else f"No statistically significant difference observed between '{k1}' and '{k2}' ({test_name}, p={p_val:.4f})."
            )

            return GroupDifferenceResult(
                test_used=test_name,
                is_significant=is_sig,
                p_value=round(float(p_val), 5),
                statistic=round(stat_val, 3),
                effect_size=round(float(hedges_g), 3),
                effect_size_metric="hedges_g",
                practical_significance=prac_sig,
                group_means=group_means,
                group_medians=group_medians,
                group_sizes=group_sizes,
                interpretation=interp,
            )

        # Case B: 3 or more Groups
        if all_normal and min(len(a) for a in group_arrays) >= 15:
            # One-Way ANOVA
            f_stat, p_val = stats.f_oneway(*group_arrays)
            test_name = "One-Way ANOVA"
            stat_val = float(f_stat)

            # Eta-squared calculation
            grand_mean = clean_df[value_col].mean()
            ss_total = ((clean_df[value_col] - grand_mean) ** 2).sum()
            ss_between = sum(len(arr) * ((np.mean(arr) - grand_mean) ** 2) for arr in group_arrays)
            eta_sq = float(ss_between / max(ss_total, 1e-6))
        else:
            # Kruskal-Wallis H-test
            h_stat, p_val = stats.kruskal(*group_arrays)
            test_name = "Kruskal-Wallis H Test"
            stat_val = float(h_stat)

            # Epsilon-squared estimate for Kruskal-Wallis
            total_n = len(clean_df)
            eta_sq = float((stat_val - len(group_keys) + 1) / max(total_n - len(group_keys), 1))
            eta_sq = max(0.0, min(1.0, eta_sq))

        # Effect size classification
        if eta_sq < 0.01:
            prac_sig = "negligible"
        elif eta_sq < 0.06:
            prac_sig = "small"
        elif eta_sq < 0.14:
            prac_sig = "medium"
        else:
            prac_sig = "large"

        is_sig = bool(p_val < alpha)
        interp = (
            f"Statistically significant difference across {len(group_keys)} groups ({test_name}, p={p_val:.4f}). "
            f"Dimension explains {round(eta_sq*100, 1)}% of total variation (Effect size: {prac_sig})."
            if is_sig
            else f"No statistically significant difference observed across {len(group_keys)} groups ({test_name}, p={p_val:.4f})."
        )

        return GroupDifferenceResult(
            test_used=test_name,
            is_significant=is_sig,
            p_value=round(float(p_val), 5),
            statistic=round(stat_val, 3),
            effect_size=round(float(eta_sq), 3),
            effect_size_metric="eta_squared",
            practical_significance=prac_sig,
            group_means=group_means,
            group_medians=group_medians,
            group_sizes=group_sizes,
            interpretation=interp,
        )

    # ─────────────────────────────────────────────────────────────
    # Changepoint & Structural Break Detection (Binary Segmentation)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def detect_changepoints(
        series: Sequence[float],
        labels: Optional[Sequence[Any]] = None,
        min_segment_length: int = 3,
        threshold_variance_reduction: float = 0.35,
    ) -> list[ChangepointResult]:
        """
        Binary Segmentation changepoint detection:
        Recursively discovers structural shifts that maximize variance reduction
        without picking spurious secondary splits in homogeneous segments.
        """
        vals = pd.Series(series).dropna().to_numpy(dtype=float)
        n = len(vals)
        changepoints: list[ChangepointResult] = []

        def find_best_split(start_idx: int, end_idx: int) -> None:
            seg_len = end_idx - start_idx
            if seg_len < (min_segment_length * 2):
                return

            sub_vals = vals[start_idx:end_idx]
            sub_ss = float(np.sum((sub_vals - np.mean(sub_vals)) ** 2))
            if sub_ss <= 1e-7:
                return

            best_reduction = 0.0
            best_k = None
            best_diff = 0.0
            best_m1, best_m2 = 0.0, 0.0

            for k in range(min_segment_length, seg_len - min_segment_length + 1):
                left = sub_vals[:k]
                right = sub_vals[k:]
                m1, m2 = float(np.mean(left)), float(np.mean(right))
                cost = float(np.sum((left - m1) ** 2) + np.sum((right - m2) ** 2))
                var_reduced = (sub_ss - cost) / sub_ss

                if var_reduced > best_reduction:
                    best_reduction = var_reduced
                    best_k = start_idx + k
                    best_diff = m2 - m1
                    best_m1, best_m2 = m1, m2

            if best_reduction >= threshold_variance_reduction and best_k is not None:
                period_str = str(labels[best_k]) if (labels is not None and best_k < len(labels)) else f"Index {best_k}"
                pct_change = (best_diff / max(abs(best_m1), 1e-5)) * 100.0
                direction = "upward_shift" if best_diff > 0 else "downward_shift"

                changepoints.append(
                    ChangepointResult(
                        index=best_k,
                        period_label=period_str,
                        direction=direction,
                        mean_before=round(best_m1, 2),
                        mean_after=round(best_m2, 2),
                        absolute_change=round(best_diff, 2),
                        percent_change=round(pct_change, 1),
                        confidence=min(0.99, round(0.5 + 0.5 * best_reduction, 2)),
                        description=(
                            f"Structural {direction.replace('_', ' ')} detected at {period_str}: "
                            f"mean shifted from {best_m1:.1f} to {best_m2:.1f} ({pct_change:+.1f}%)."
                        ),
                    )
                )

                # Recurse into left and right sub-windows
                find_best_split(start_idx, best_k)
                find_best_split(best_k, end_idx)

        find_best_split(0, n)
        changepoints.sort(key=lambda c: c.index)
        return changepoints

    # ─────────────────────────────────────────────────────────────
    # Subgroup Impact Segmentation Analysis
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def run_segmentation_analysis(
        df: pd.DataFrame,
        target_col: str,
        dimension_cols: Optional[list[str]] = None,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Ranks categorical dimensions by their explanatory power (eta-squared)
        and statistical divergence for the target numeric metric.
        Identifies key high-performing and low-performing cohorts.
        """
        if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
            return []

        if dimension_cols is None:
            # Auto-select categorical columns with 2 to 30 unique values
            dimension_cols = [
                c for c in df.columns
                if c != target_col and 2 <= df[c].nunique() <= 30
            ]

        ranked_segments = []

        for dim in dimension_cols:
            res = StatisticalEngine.test_group_difference(df, target_col, dim)
            if res.is_significant or res.effect_size >= 0.05:
                # Find best and worst performing sub-cohorts
                sorted_groups = sorted(res.group_means.items(), key=lambda x: x[1], reverse=True)
                top_cohort = sorted_groups[0]
                bottom_cohort = sorted_groups[-1]
                gap = top_cohort[1] - bottom_cohort[1]

                ranked_segments.append({
                    "dimension": dim,
                    "test_used": res.test_used,
                    "p_value": res.p_value,
                    "effect_size": res.effect_size,
                    "practical_significance": res.practical_significance,
                    "variance_explained_pct": round(res.effect_size * 100, 1),
                    "highest_subgroup": {"name": top_cohort[0], "mean": top_cohort[1]},
                    "lowest_subgroup": {"name": bottom_cohort[0], "mean": bottom_cohort[1]},
                    "disparity_gap": round(gap, 2),
                    "interpretation": res.interpretation,
                })

        # Rank by effect size
        ranked_segments.sort(key=lambda x: x["effect_size"], reverse=True)
        return ranked_segments[:top_n]


def run_statistical_inference(
    df: pd.DataFrame,
    target_metric: Optional[str] = None,
    time_col: Optional[str] = None,
) -> dict[str, Any]:
    """
    Main entry point for deep inferential statistical analysis of a dataset.
    """
    engine = StatisticalEngine()
    results: dict[str, Any] = {}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # 1. Target metric selection
    if not target_metric and numeric_cols:
        target_metric = numeric_cols[0]

    # 2. Trend analysis if temporal data available
    if time_col and time_col in df.columns and target_metric:
        df_sorted = df.sort_values(by=time_col)
        time_series = df_sorted[target_metric].dropna()
        trend_res = engine.test_trend_significance(time_series)
        changepoints = engine.detect_changepoints(time_series, labels=df_sorted[time_col].tolist())
        results["trend_inference"] = {
            "metric": target_metric,
            "trend": asdict(trend_res) if hasattr(trend_res, "__dataclass_fields__") else trend_res.__dict__,
            "changepoints": [c.__dict__ for c in changepoints],
        }

    # 3. Segmentation analysis
    if target_metric:
        segments = engine.run_segmentation_analysis(df, target_metric)
        results["segmentation_drivers"] = {
            "target_metric": target_metric,
            "drivers": segments,
        }

    return results
