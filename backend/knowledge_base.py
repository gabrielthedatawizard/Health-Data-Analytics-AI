"""
Analytics Knowledge Base for Health Data Analytics AI
Phase 4.1: Adaptive Learning System

Persistent, cross-session and cross-dataset intelligence store that:
1. Learns and standardizes indicator definitions across health programs
2. Learns dataset schema archetypes and suggests historical best-practice analyses
3. Remembers personalized user preferences (chart types, detail levels)
4. Tracks recurring facility quality pathologies across successive uploads
"""

from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class AnalyticsKnowledgeBase:
    """
    Persistent cross-session knowledge engine.
    Supports atomic file persistence with clean schema evolution.
    """

    def __init__(self, storage_dir: str = "data_store/knowledge_base"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.indicators_path = self.storage_dir / "indicators.json"
        self.patterns_path = self.storage_dir / "dataset_patterns.json"
        self.user_prefs_path = self.storage_dir / "user_preferences.json"
        self.recurring_issues_path = self.storage_dir / "recurring_issues.json"

        self._indicators: Dict[str, Dict[str, Any]] = self._load_json(self.indicators_path)
        self._patterns: List[Dict[str, Any]] = self._load_json(self.patterns_path, default=[])
        self._user_prefs: Dict[str, Dict[str, Any]] = self._load_json(self.user_prefs_path)
        self._recurring_issues: Dict[str, List[Dict[str, Any]]] = self._load_json(self.recurring_issues_path)

        # Seed standard WHO/UNICEF/HMIS indicator catalog if empty
        if not self._indicators:
            self._seed_default_indicators()

    def _load_json(self, path: Path, default: Any = None) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return default if default is not None else {}

    def _save_json(self, path: Path, data: Any) -> None:
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(path)

    # ─────────────────────────────────────────────────────────────
    # 1. Indicator Definition Learning
    # ─────────────────────────────────────────────────────────────

    def _seed_default_indicators(self) -> None:
        defaults = {
            "anc_1_coverage": {
                "name": "ANC 1 Coverage",
                "formula": "(ANC1_Visits / Estimated_Pregnant_Women) * 100",
                "unit": "percentage",
                "target": 95.0,
                "direction": "higher_is_better",
                "source": "WHO_HMIS",
                "tags": ["maternal_health", "antenatal", "who_core"],
                "learned_at": datetime.now(timezone.utc).isoformat(),
            },
            "anc_4_coverage": {
                "name": "ANC 4+ Coverage",
                "formula": "(ANC4_Visits / Estimated_Pregnant_Women) * 100",
                "unit": "percentage",
                "target": 80.0,
                "direction": "higher_is_better",
                "source": "WHO_HMIS",
                "tags": ["maternal_health", "antenatal", "who_core"],
                "learned_at": datetime.now(timezone.utc).isoformat(),
            },
            "facility_delivery_rate": {
                "name": "Skilled Birth Attendance / Facility Delivery Rate",
                "formula": "(Facility_Deliveries / Total_Expected_Deliveries) * 100",
                "unit": "percentage",
                "target": 90.0,
                "direction": "higher_is_better",
                "source": "WHO_HMIS",
                "tags": ["maternal_health", "delivery", "vital_statistics"],
                "learned_at": datetime.now(timezone.utc).isoformat(),
            },
            "dpt3_penta3_coverage": {
                "name": "DPT3 / Pentavalent 3 Coverage",
                "formula": "(Children_Receiving_Penta3 / Surviving_Infants) * 100",
                "unit": "percentage",
                "target": 90.0,
                "direction": "higher_is_better",
                "source": "UNICEF_EPI",
                "tags": ["immunization", "epi", "child_health"],
                "learned_at": datetime.now(timezone.utc).isoformat(),
            },
            "measles_coverage": {
                "name": "Measles 1st Dose Coverage",
                "formula": "(Children_Receiving_MCV1 / Surviving_Infants) * 100",
                "unit": "percentage",
                "target": 95.0,
                "direction": "higher_is_better",
                "source": "WHO_EPI",
                "tags": ["immunization", "epi", "measles"],
                "learned_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        self._indicators.update(defaults)
        self._save_json(self.indicators_path, self._indicators)

    def learn_indicator_definition(
        self,
        name: str,
        formula: str,
        unit: str = "number",
        target: Optional[float] = None,
        direction: str = "higher_is_better",
        source: str = "user_analyst",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Stores an indicator definition learned from an analyst session or data dictionary.
        """
        clean_key = re.sub(r"[^a-z0-9_]+", "_", name.lower().strip()).strip("_")
        record = {
            "name": name,
            "formula": formula,
            "unit": unit,
            "target": target,
            "direction": direction,
            "source": source,
            "tags": tags or [],
            "learned_at": datetime.now(timezone.utc).isoformat(),
        }
        self._indicators[clean_key] = record
        self._save_json(self.indicators_path, self._indicators)
        return record

    def get_indicator_definition(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves matching indicator definition by exact or fuzzy keyword match.
        """
        q_clean = re.sub(r"[^a-z0-9_]+", "_", query.lower().strip()).strip("_")
        if q_clean in self._indicators:
            return self._indicators[q_clean]

        # Substring / keyword match
        tokens = set(q_clean.split("_"))
        best_match = None
        best_overlap = 0

        for key, record in self._indicators.items():
            key_tokens = set(key.split("_"))
            overlap = len(tokens & key_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = record

        return best_match if best_overlap >= 1 else None

    def list_indicators(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        all_inds = list(self._indicators.values())
        if tag:
            return [ind for ind in all_inds if tag.lower() in [t.lower() for t in ind.get("tags", [])]]
        return all_inds

    # ─────────────────────────────────────────────────────────────
    # 2. Dataset Archetype Learning & Best-Practice Suggestions
    # ─────────────────────────────────────────────────────────────

    def learn_dataset_pattern(
        self,
        archetype_name: str,
        detected_categories: List[str],
        key_columns: List[str],
        recommended_questions: List[str],
        typical_dashboard_views: List[str],
    ) -> None:
        """
        Registers an observed dataset archetype to cross-pollinate analytical best practices.
        """
        pattern = {
            "archetype_name": archetype_name,
            "categories": sorted(list(set(detected_categories))),
            "key_columns": sorted(list(set(key_columns))),
            "recommended_questions": recommended_questions,
            "typical_dashboard_views": typical_dashboard_views,
            "learned_at": datetime.now(timezone.utc).isoformat(),
        }
        # Avoid duplicate archetypes
        self._patterns = [p for p in self._patterns if p.get("archetype_name") != archetype_name]
        self._patterns.append(pattern)
        self._save_json(self.patterns_path, self._patterns)

    def suggest_analyses_for_dataset(
        self,
        column_names: List[str],
        detected_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Matches incoming dataset schema against learned archetypes to suggest
        expert analytical workflows and priority questions.
        """
        col_set = {str(c).lower() for c in column_names}
        cat_set = set(detected_categories or [])

        best_score = -1
        matched_archetype = None

        for p in self._patterns:
            score = 0
            # Match categories
            p_cats = set(p.get("categories", []))
            cat_overlap = len(cat_set & p_cats)
            score += cat_overlap * 3

            # Match columns
            p_cols = {c.lower() for c in p.get("key_columns", [])}
            col_overlap = len(col_set & p_cols)
            score += col_overlap * 2

            if score > best_score:
                best_score = score
                matched_archetype = p

        # Default fallback suggestions if no pattern matches strongly
        if not matched_archetype or best_score < 2:
            return {
                "matched_archetype": "general_health_monitoring",
                "confidence": 0.5,
                "suggested_questions": [
                    "What are the top 5 facilities or districts by patient volume?",
                    "Are there noticeable seasonal changes across reporting periods?",
                    "Which indicators exhibit unexpected data entry spikes or drops?",
                ],
                "recommended_views": ["KPI Summary Cards", "Monthly Trends", "District Comparison Bar"],
            }

        return {
            "matched_archetype": matched_archetype["archetype_name"],
            "confidence": min(0.95, round(0.5 + 0.1 * best_score, 2)),
            "suggested_questions": matched_archetype.get("recommended_questions", []),
            "recommended_views": matched_archetype.get("typical_dashboard_views", []),
        }

    # ─────────────────────────────────────────────────────────────
    # 3. User & Analyst Preference Learning
    # ─────────────────────────────────────────────────────────────

    def learn_user_preference(
        self, user_id: str, preference_key: str, value: Any
    ) -> None:
        """
        Stores user-specific analysis habits (e.g., preferred charts, dark mode, default time grains).
        """
        if user_id not in self._user_prefs:
            self._user_prefs[user_id] = {}
        self._user_prefs[user_id][preference_key] = value
        self._user_prefs[user_id]["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save_json(self.user_prefs_path, self._user_prefs)

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        return self._user_prefs.get(user_id, {})

    # ─────────────────────────────────────────────────────────────
    # 4. Cross-Dataset Recurring Issue Tracking
    # ─────────────────────────────────────────────────────────────

    def record_issue(
        self,
        entity_name: str,
        issue_type: str,
        dataset_id: str,
        details: str,
    ) -> None:
        """
        Tracks data quality issues (e.g. repeated frozen facility reporting, chronic missingness).
        """
        key = entity_name.strip()
        if key not in self._recurring_issues:
            self._recurring_issues[key] = []

        self._recurring_issues[key].append({
            "issue_type": issue_type,
            "dataset_id": dataset_id,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Cap history to 50 records per entity
        if len(self._recurring_issues[key]) > 50:
            self._recurring_issues[key] = self._recurring_issues[key][-50:]

        self._save_json(self.recurring_issues_path, self._recurring_issues)

    def get_recurring_issues(self, entity_name: str) -> List[Dict[str, Any]]:
        return self._recurring_issues.get(entity_name.strip(), [])

    def detect_chronic_offenders(self, min_incidents: int = 2) -> List[Dict[str, Any]]:
        """
        Surfaces entities (facilities/indicators) with repeated quality pathologies.
        """
        chronic = []
        for entity, issues in self._recurring_issues.items():
            if len(issues) >= min_incidents:
                chronic.append({
                    "entity": entity,
                    "total_incidents": len(issues),
                    "unique_issue_types": list({i["issue_type"] for i in issues}),
                    "latest_incident": issues[-1],
                })
        chronic.sort(key=lambda x: x["total_incidents"], reverse=True)
        return chronic
