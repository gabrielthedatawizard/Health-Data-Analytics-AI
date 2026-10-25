"""
Proactive Intelligence System for Health Data Analytics AI
Phase 6.2: Autonomous Analytics Agent

Monitors health programs proactively without waiting for user prompts:
- Evaluates indicators against national targets and WHO benchmarks
- Detects early-warning deteriorations before acute crises
- Identifies cross-dataset join and analysis opportunities
- Generates C-suite executive intelligence briefings
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class ProactiveIntelligence:
    """
    Continuous proactive surveillance and analytics engine.
    """

    def __init__(self):
        # Default global benchmarks (e.g. WHO standard targets)
        self.default_benchmarks: Dict[str, Dict[str, Any]] = {
            "anc_1_coverage": {"target": 95.0, "direction": "higher_is_better", "label": "ANC 1 Coverage"},
            "anc_4_coverage": {"target": 80.0, "direction": "higher_is_better", "label": "ANC 4+ Coverage"},
            "dpt3_penta3_coverage": {"target": 90.0, "direction": "higher_is_better", "label": "Pentavalent 3 Coverage"},
            "measles_coverage": {"target": 95.0, "direction": "higher_is_better", "label": "Measles 1st Dose Coverage"},
            "facility_delivery_rate": {"target": 90.0, "direction": "higher_is_better", "label": "Skilled Birth Attendance"},
        }

    # ─────────────────────────────────────────────────────────────
    # 1. Proactive Dataset Scanning & Target Tracking
    # ─────────────────────────────────────────────────────────────

    def scan_dataset(
        self,
        df: pd.DataFrame,
        facts: Dict[str, Any],
        custom_targets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a 360-degree proactive scan covering targets, early warnings,
        anomalies, and data quality.
        """
        findings = []
        metrics = facts.get("metrics", {}).get("kpis", {})
        trends = facts.get("trends", {})
        anom_report = facts.get("anomaly_report", {})

        targets = dict(self.default_benchmarks)
        if custom_targets:
            for k, v in custom_targets.items():
                targets[k.lower()] = {"target": float(v), "direction": "higher_is_better", "label": k.replace("_", " ").title()}

        # 1. Target Breaches
        for metric_key, metric_data in metrics.items():
            k_low = metric_key.lower()
            matching_target = None
            for t_k, t_info in targets.items():
                if t_k in k_low or k_low in t_k:
                    matching_target = t_info
                    break

            if matching_target:
                curr_val = metric_data.get("mean") or metric_data.get("total", 0.0)
                tgt = matching_target["target"]
                direction = matching_target.get("direction", "higher_is_better")

                is_breach = (curr_val < tgt) if direction == "higher_is_better" else (curr_val > tgt)
                if is_breach:
                    gap = abs(curr_val - tgt)
                    findings.append({
                        "category": "target_breach",
                        "severity": "critical" if gap > 15 else "warning",
                        "metric": metric_key,
                        "title": f"Target Breach in {matching_target['label']}",
                        "detail": f"Observed value ({curr_val:.1f}) is trailing national target ({tgt:.1f}) by a gap of {gap:.1f} percentage points.",
                        "action": f"Prioritize outreach campaigns to close the {gap:.1f} pt performance gap.",
                    })

        # 2. Early-Warning Negative Trends
        for t_k, t_data in trends.items():
            if "significance_test" in t_data:
                sig = t_data["significance_test"]
                direction = sig.get("trend_direction")
                is_sig = sig.get("is_significant", False)
                m_name = t_data.get("metric", t_k)

                if is_sig and direction in ("decreasing", "downward"):
                    findings.append({
                        "category": "early_warning",
                        "severity": "high",
                        "metric": m_name,
                        "title": f"Early Warning: Declining Trajectory in {m_name.replace('_', ' ').title()}",
                        "detail": f"Statistically significant downward trajectory detected (p={sig.get('p_value'):.4f}, slope={sig.get('slope'):.2f}).",
                        "action": "Initiate root-cause investigation before service volumes reach critical shortage thresholds.",
                    })

        # 3. Critical Data Anomalies
        if anom_report and anom_report.get("records"):
            crit_anoms = [a for a in anom_report["records"] if a.get("severity") == "critical"]
            if crit_anoms:
                top_crit = crit_anoms[0]
                findings.append({
                    "category": "data_integrity",
                    "severity": "critical",
                    "metric": ", ".join(top_crit.get("columns", [])),
                    "title": "Critical Data Entry Integrity Alert",
                    "detail": top_crit.get("hypothesis", ""),
                    "action": top_crit.get("action", "Validate ledger at source facility."),
                })

        # Sort findings by urgency
        sev_rank = {"critical": 3, "high": 2, "warning": 1, "info": 0}
        findings.sort(key=lambda x: sev_rank.get(x["severity"], 0), reverse=True)

        return {
            "dataset_id": facts.get("dataset_id", "current"),
            "total_findings": len(findings),
            "findings": findings,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }

    # ─────────────────────────────────────────────────────────────
    # 2. Cross-Dataset Analysis Opportunity Detection
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def suggest_cross_dataset_analyses(
        datasets_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Discovers opportunities to merge or cross-analyze disparate health datasets
        (e.g. Immunization Service Records + Cold Chain Logistics Inventory).
        """
        if len(datasets_metadata) < 2:
            return []

        suggestions = []
        n = len(datasets_metadata)

        for i in range(n):
            for j in range(i + 1, n):
                ds_a = datasets_metadata[i]
                ds_b = datasets_metadata[j]

                cols_a = {str(c).lower() for c in ds_a.get("columns", [])}
                cols_b = {str(c).lower() for c in ds_b.get("columns", [])}

                # Find candidate join keys
                common_keys = cols_a & cols_b
                joinable_keys = [k for k in common_keys if any(term in k for term in ["id", "facility", "district", "date", "month", "period", "code"])]

                if joinable_keys:
                    name_a = ds_a.get("name", f"Dataset_{i+1}")
                    name_b = ds_b.get("name", f"Dataset_{j+1}")
                    key_str = ", ".join(joinable_keys[:2])

                    suggestions.append({
                        "dataset_pair": [name_a, name_b],
                        "suggested_join_keys": joinable_keys[:2],
                        "rationale": f"Both datasets share '{key_str}' enabling unified facility-level cross-impact analytics.",
                        "suggested_question": f"How do {name_b} operational constraints directly impact {name_a} patient health outcomes?",
                        "impact_score": "High" if len(joinable_keys) >= 2 else "Medium",
                    })

        return suggestions

    # ─────────────────────────────────────────────────────────────
    # 3. Executive Intelligence Briefing
    # ─────────────────────────────────────────────────────────────

    def generate_executive_briefing(
        self,
        scan_results: Dict[str, Any],
        facts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constructs a concise, actionable C-suite briefing.
        """
        findings = scan_results.get("findings", [])
        total_crit = sum(1 for f in findings if f.get("severity") == "critical")
        total_warn = sum(1 for f in findings if f.get("severity") in ("high", "warning"))

        # Health score calculation [0 - 100]
        penalty = total_crit * 25 + total_warn * 10
        program_health_score = max(20, min(100, 100 - penalty))

        if program_health_score >= 85:
            posture = "Optimal Operational Performance"
        elif program_health_score >= 65:
            posture = "Moderate Operational Risk / Target Deficit"
        else:
            posture = "High Urgent Intervention Required"

        recommendations = [f.get("action") for f in findings if f.get("action")][:3]
        if not recommendations:
            recommendations = ["Maintain routine monthly epidemiological surveillance."]

        briefing_text = (
            f"**Executive Health Program Briefing**: Program Health Score is **{program_health_score}/100** ({posture}). "
            f"Identified {total_crit} critical alerts and {total_warn} high-priority early warnings across monitored indicators."
        )

        return {
            "health_score": program_health_score,
            "posture": posture,
            "briefing": briefing_text,
            "priority_recommendations": recommendations,
            "critical_alerts_count": total_crit,
            "findings": findings[:5],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
