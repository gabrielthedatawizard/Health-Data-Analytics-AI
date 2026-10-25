"""
Contextual Interpretation & Benchmark Engine for Health Data Analytics AI
Phase 2.2: Statistical Reasoning Engine

Translates inferential statistics, KPIs, and trends into domain-aware clinical and public health intelligence:
- International & National Health Benchmarks (WHO, UNICEF, CDC, UNAIDS, DHIS2)
- Polarity & Directionality Framework (higher_is_better, lower_is_better, target_range)
- Severity & Clinical Urgency Classification
- Statistical & Domain Narrative Synthesis
"""

from __future__ import annotations

import re
import math
from typing import Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


# =====================================================================
# 1. INTERNATIONAL HEALTH BENCHMARK DATABASE
# =====================================================================

@dataclass
class HealthBenchmark:
    indicator_id: str
    indicator_name: str
    target_value: float
    target_type: str  # "min_threshold", "max_threshold", "target_range"
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    unit: str = "%"
    polarity: str = "higher_is_better"  # "higher_is_better", "lower_is_better", "target_range"
    authority: str = "WHO"
    citation: str = ""
    clinical_rationale: str = ""


HEALTH_BENCHMARKS: dict[str, HealthBenchmark] = {
    # ── Maternal, Newborn & Child Health (MNCH) ──
    "vaccination_coverage": HealthBenchmark(
        indicator_id="vaccination_coverage",
        indicator_name="Routine Immunization Coverage (Penta3/Measles)",
        target_value=90.0,
        target_type="min_threshold",
        unit="%",
        polarity="higher_is_better",
        authority="WHO / UNICEF",
        citation="Global Vaccine Action Plan (GVAP)",
        clinical_rationale="Achieving >=90% national coverage establishes herd immunity and prevents vaccine-preventable outbreaks.",
    ),
    "anc_4_plus": HealthBenchmark(
        indicator_id="anc_4_plus",
        indicator_name="Antenatal Care 4+ Visits Coverage",
        target_value=60.0,
        target_type="min_threshold",
        unit="%",
        polarity="higher_is_better",
        authority="WHO",
        citation="WHO ANC Model Guidelines",
        clinical_rationale="At least 4 focused ANC contacts significantly decrease maternal and perinatal mortality.",
    ),
    "skilled_birth_delivery": HealthBenchmark(
        indicator_id="skilled_birth_delivery",
        indicator_name="Skilled Birth Attendance / Institutional Delivery Rate",
        target_value=85.0,
        target_type="min_threshold",
        unit="%",
        polarity="higher_is_better",
        authority="WHO / SDG 3.1",
        citation="Sustainable Development Goals Framework",
        clinical_rationale="Deliveries attended by skilled health personnel ensure prompt management of obstetric emergencies.",
    ),
    "c_section_rate": HealthBenchmark(
        indicator_id="c_section_rate",
        indicator_name="Cesarean Section Delivery Rate",
        target_value=12.5,
        target_type="target_range",
        range_min=10.0,
        range_max=15.0,
        unit="%",
        polarity="target_range",
        authority="WHO",
        citation="WHO Statement on Caesarean Section Rates",
        clinical_rationale="Rates <10% suggest inadequate surgical access; rates >15% indicate unnecessary surgical interventions.",
    ),

    # ── Disease Control & Surveillance ──
    "malaria_positivity_rate": HealthBenchmark(
        indicator_id="malaria_positivity_rate",
        indicator_name="Malaria Test Positivity Rate (TPR)",
        target_value=5.0,
        target_type="max_threshold",
        unit="%",
        polarity="lower_is_better",
        authority="WHO Global Malaria Programme",
        citation="WHO Malaria Surveillance Guidelines",
        clinical_rationale="TPR <=5% reflects low transmission phase; sustained rates above 20% indicate epidemic transmission.",
    ),
    "hiv_viral_suppression": HealthBenchmark(
        indicator_id="hiv_viral_suppression",
        indicator_name="HIV Viral Load Suppression Rate (<1000 copies/mL)",
        target_value=95.0,
        target_type="min_threshold",
        unit="%",
        polarity="higher_is_better",
        authority="UNAIDS",
        citation="UNAIDS 95-95-95 Targets",
        clinical_rationale="Viral suppression above 95% halts sexual transmission (U=U) and improves patient longevity.",
    ),
    "tb_treatment_success": HealthBenchmark(
        indicator_id="tb_treatment_success",
        indicator_name="Tuberculosis Treatment Success Rate",
        target_value=90.0,
        target_type="min_threshold",
        unit="%",
        polarity="higher_is_better",
        authority="WHO End TB Strategy",
        citation="WHO Global Tuberculosis Report",
        clinical_rationale="High treatment completion prevents acquisition of multidrug-resistant tuberculosis (MDR-TB).",
    ),

    # ── Hospital Systems & Operations ──
    "bed_occupancy_rate": HealthBenchmark(
        indicator_id="bed_occupancy_rate",
        indicator_name="Inpatient Bed Occupancy Rate (BOR)",
        target_value=82.5,
        target_type="target_range",
        range_min=80.0,
        range_max=85.0,
        unit="%",
        polarity="target_range",
        authority="WHO / NHS",
        citation="WHO Health Systems Performance Standards",
        clinical_rationale="80-85% is optimal hospital capacity. Below 70% is underutilized; above 85% increases hospital-acquired infections and delays admissions.",
    ),
    "stock_out_days": HealthBenchmark(
        indicator_id="stock_out_days",
        indicator_name="Stock-out Days of Essential Medicines",
        target_value=0.0,
        target_type="max_threshold",
        unit="days",
        polarity="lower_is_better",
        authority="WHO",
        citation="Essential Medicines and Health Products Standards",
        clinical_rationale="Zero stock-outs ensures uncompromised continuum of life-saving medical care.",
    ),
    "data_reporting_completeness": HealthBenchmark(
        indicator_id="data_reporting_completeness",
        indicator_name="HMIS Monthly Reporting Completeness Rate",
        target_value=90.0,
        target_type="min_threshold",
        unit="%",
        polarity="higher_is_better",
        authority="DHIS2 / WHO",
        citation="WHO Data Quality Review Toolkit",
        clinical_rationale="Reporting completeness below 80% severely distorts disease burden estimates and resource allocation.",
    ),
}


# =====================================================================
# 2. CONTEXTUAL INTERPRETATION ENGINE
# =====================================================================

class InterpretationEngine:
    """
    Evaluates observed statistics and indicators against international clinical
    norms, assesses trend severity, and produces grounded executive narratives.
    """

    def __init__(self):
        self.benchmarks = HEALTH_BENCHMARKS

    def match_benchmark(self, col_name: str, semantic_type: Optional[str] = None) -> Optional[HealthBenchmark]:
        """Matches a dataset column name or semantic type to a health benchmark."""
        col_clean = col_name.lower()

        # Match by semantic type first
        if semantic_type:
            mapping = {
                "coverage_rate": "vaccination_coverage",
                "bed_occupancy_rate": "bed_occupancy_rate",
                "stock_out_days": "stock_out_days",
                "hiv_viral_load": "hiv_viral_suppression",
                "malaria_rdt_result": "malaria_positivity_rate",
                "data_completeness_rate": "data_reporting_completeness",
            }
            if semantic_type in mapping and mapping[semantic_type] in self.benchmarks:
                return self.benchmarks[mapping[semantic_type]]

        # Match by column name keywords
        for b_id, bench in self.benchmarks.items():
            if b_id in col_clean:
                return bench

        if "vaccin" in col_clean or "immun" in col_clean or "penta" in col_clean:
            return self.benchmarks["vaccination_coverage"]
        if "anc" in col_clean:
            return self.benchmarks["anc_4_plus"]
        if "occupancy" in col_clean or "bor" in col_clean:
            return self.benchmarks["bed_occupancy_rate"]
        if "stock" in col_clean and ("out" in col_clean or "day" in col_clean):
            return self.benchmarks["stock_out_days"]
        if "malaria" in col_clean and ("positiv" in col_clean or "rate" in col_clean or "tpr" in col_clean):
            return self.benchmarks["malaria_positivity_rate"]
        if "delivery" in col_clean or "birth" in col_clean:
            return self.benchmarks["skilled_birth_delivery"]

        return None

    def interpret_with_benchmark(
        self,
        col_name: str,
        observed_value: float,
        semantic_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Evaluates a single value against standard health benchmarks.
        """
        bench = self.match_benchmark(col_name, semantic_type)
        if not bench:
            return {
                "has_benchmark": False,
                "observed_value": observed_value,
                "status": "unbenchmarked",
                "narrative": f"Metric '{col_name}' evaluated at {observed_value} (no international threshold defined).",
            }

        # 1. Target Range check
        if bench.target_type == "target_range" and bench.range_min is not None and bench.range_max is not None:
            if bench.range_min <= observed_value <= bench.range_max:
                status = "optimal"
                gap = 0.0
                narrative = (
                    f"'{bench.indicator_name}' is optimal at {observed_value}{bench.unit}, "
                    f"within the {bench.authority} recommended target range of {bench.range_min}-{bench.range_max}{bench.unit}."
                )
            elif observed_value < bench.range_min:
                gap = observed_value - bench.range_min
                status = "suboptimal"
                narrative = (
                    f"'{bench.indicator_name}' is below the optimal threshold at {observed_value}{bench.unit} "
                    f"({abs(gap):.1f}{bench.unit} below the {bench.range_min}{bench.unit} target; {bench.authority}). {bench.clinical_rationale}"
                )
            else:
                gap = observed_value - bench.range_max
                status = "concerning" if gap > 10 else "warning"
                narrative = (
                    f"'{bench.indicator_name}' exceeds the safety ceiling at {observed_value}{bench.unit} "
                    f"({gap:.1f}{bench.unit} above {bench.range_max}{bench.unit}; {bench.authority}). {bench.clinical_rationale}"
                )

        # 2. Minimum Threshold check (higher is better)
        elif bench.polarity == "higher_is_better":
            gap = observed_value - bench.target_value
            if gap >= 0:
                status = "exceeds_target" if gap > 5 else "on_target"
                narrative = (
                    f"'{bench.indicator_name}' achieves the {bench.authority} target at {observed_value}{bench.unit} "
                    f"(target: >={bench.target_value}{bench.unit})."
                )
            elif gap >= -10:
                status = "suboptimal"
                narrative = (
                    f"'{bench.indicator_name}' is slightly below target at {observed_value}{bench.unit} "
                    f"(gap of {abs(gap):.1f}{bench.unit} from {bench.target_value}{bench.unit}; {bench.authority})."
                )
            else:
                status = "critical_alert"
                narrative = (
                    f"Critical shortfall in '{bench.indicator_name}' at {observed_value}{bench.unit} "
                    f"({abs(gap):.1f}{bench.unit} deficit below {bench.authority} standard of {bench.target_value}{bench.unit}). {bench.clinical_rationale}"
                )

        # 3. Maximum Threshold check (lower is better)
        else:
            gap = observed_value - bench.target_value
            if gap <= 0:
                status = "on_target"
                narrative = (
                    f"'{bench.indicator_name}' meets target at {observed_value}{bench.unit} "
                    f"(target: <={bench.target_value}{bench.unit}; {bench.authority})."
                )
            elif gap <= 5:
                status = "warning"
                narrative = (
                    f"'{bench.indicator_name}' is elevated at {observed_value}{bench.unit} "
                    f"(exceeds target by {gap:.1f}{bench.unit}; {bench.authority})."
                )
            else:
                status = "critical_alert"
                narrative = (
                    f"Severe alert: '{bench.indicator_name}' at {observed_value}{bench.unit} "
                    f"exceeds the {bench.authority} threshold ({bench.target_value}{bench.unit}) by {gap:.1f}{bench.unit}. {bench.clinical_rationale}"
                )

        return {
            "has_benchmark": True,
            "indicator_id": bench.indicator_id,
            "indicator_name": bench.indicator_name,
            "observed_value": observed_value,
            "target_value": bench.target_value,
            "unit": bench.unit,
            "gap": round(gap, 2),
            "status": status,
            "authority": bench.authority,
            "citation": bench.citation,
            "clinical_rationale": bench.clinical_rationale,
            "narrative": narrative,
        }

    def assess_trend_severity(
        self,
        col_name: str,
        trend_direction: str,
        is_significant: bool,
        percent_change: float = 0.0,
        semantic_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Determines domain-aware clinical severity based on directionality:
        - Increasing vaccination coverage -> "favorable / improving"
        - Increasing malaria cases / stockouts -> "critical_alert / worsening"
        """
        bench = self.match_benchmark(col_name, semantic_type)
        if bench:
            polarity = bench.polarity
        else:
            col_lower = col_name.lower()
            if any(k in col_lower for k in ["case", "death", "died", "mortality", "malaria", "tb", "stock", "gap", "infection"]):
                polarity = "lower_is_better"
            elif any(k in col_lower for k in ["occupancy", "bor", "bmi", "cesarean", "c_section"]):
                polarity = "target_range"
            else:
                polarity = "higher_is_better"

        if not is_significant or trend_direction == "stable":
            return {
                "severity": "stable",
                "polarity": polarity,
                "urgency": "routine",
                "summary": f"'{col_name}' has remained statistically stable over time.",
            }

        # Polarity matching
        if polarity == "higher_is_better":
            if trend_direction == "increasing":
                severity = "improving"
                urgency = "low"
                summary = f"Positive progress: '{col_name}' increased significantly (+{abs(percent_change):.1f}%)."
            else:
                severity = "critical" if abs(percent_change) > 15 else "concerning"
                urgency = "high"
                summary = f"Action required: Significant decline in '{col_name}' (-{abs(percent_change):.1f}%)."
        elif polarity == "lower_is_better":
            if trend_direction == "decreasing":
                severity = "improving"
                urgency = "low"
                summary = f"Positive progress: Significant reduction in '{col_name}' (-{abs(percent_change):.1f}%)."
            else:
                severity = "critical" if abs(percent_change) > 15 else "concerning"
                urgency = "emergency" if abs(percent_change) > 30 else "high"
                summary = f"Epidemic/Outbreak warning: '{col_name}' increased significantly (+{abs(percent_change):.1f}%)."
        else:
            severity = "warning"
            urgency = "medium"
            summary = f"Shift in '{col_name}' ({trend_direction}, {percent_change:+.1f}%)."

        return {
            "severity": severity,
            "polarity": polarity,
            "urgency": urgency,
            "summary": summary,
        }

    def generate_statistical_narrative(
        self,
        metric: str,
        trend_res: Optional[dict[str, Any]] = None,
        group_diff: Optional[dict[str, Any]] = None,
        changepoints: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """
        Compiles statistical test outputs into an executive-level, publication-ready narrative.
        """
        paragraphs: list[str] = []

        # 1. Trend narrative
        if trend_res:
            direction = trend_res.get("trend_direction", "stable")
            p_val = trend_res.get("p_value", 1.0)
            is_sig = trend_res.get("is_significant", False)
            slope = trend_res.get("slope", 0.0)

            if is_sig:
                paragraphs.append(
                    f"Analysis of **{metric}** reveals a statistically significant {direction} trajectory "
                    f"(Mann-Kendall p = {p_val:.4f}, Sen's slope = {slope:+.2f} per period). "
                )
            else:
                paragraphs.append(
                    f"**{metric}** demonstrated no statistically significant directional trend (p = {p_val:.4f}), "
                    f"indicating baseline temporal equilibrium."
                )

        # 2. Changepoint narrative
        if changepoints:
            cp = changepoints[0]
            paragraphs.append(
                f"A structural inflection occurred at **{cp.get('period_label')}**, where the mean shifted "
                f"from {cp.get('mean_before')} to {cp.get('mean_after')} ({cp.get('percent_change'):+.1f}% shift, confidence {int(cp.get('confidence', 0)*100)}%)."
            )

        # 3. Group disparity narrative
        if group_diff and group_diff.get("is_significant"):
            test_name = group_diff.get("test_used")
            p_val = group_diff.get("p_value")
            eff = group_diff.get("effect_size")
            prac = group_diff.get("practical_significance")
            paragraphs.append(
                f"Disparity analysis identifies significant variance across subgroups ({test_name}, p = {p_val:.4f}) "
                f"with a {prac} effect size ({group_diff.get('effect_size_metric')} = {eff:.2f})."
            )

        return " ".join(paragraphs) if paragraphs else f"No significant inferential patterns detected for {metric}."


def interpret_analytics(
    metric: str,
    observed_value: Optional[float] = None,
    trend_result: Optional[dict[str, Any]] = None,
    group_result: Optional[dict[str, Any]] = None,
    changepoints: Optional[list[dict[str, Any]]] = None,
    semantic_type: Optional[str] = None,
) -> dict[str, Any]:
    """
    Main API for contextual interpretation.
    """
    engine = InterpretationEngine()
    benchmark_res = engine.interpret_with_benchmark(metric, observed_value, semantic_type) if observed_value is not None else None
    trend_severity = None
    if trend_result:
        trend_severity = engine.assess_trend_severity(
            metric,
            trend_result.get("trend_direction", "stable"),
            trend_result.get("is_significant", False),
            semantic_type=semantic_type,
        )

    narrative = engine.generate_statistical_narrative(
        metric=metric,
        trend_res=trend_result,
        group_diff=group_result,
        changepoints=changepoints,
    )

    return {
        "metric": metric,
        "benchmark_evaluation": benchmark_res,
        "trend_severity": trend_severity,
        "executive_narrative": narrative,
    }
