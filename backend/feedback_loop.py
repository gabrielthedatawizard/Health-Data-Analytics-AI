"""
Feedback-Driven Improvement Engine for Health Data Analytics AI
Phase 4.2: Adaptive Learning System

Captures analyst corrections, chart overrides, and satisfaction ratings
to iteratively adapt and improve analytics generation over time.
"""

from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class FeedbackLoop:
    """
    Analyst feedback and continuous learning engine.
    Stores and applies human-in-the-loop corrections to AI analytics.
    """

    def __init__(self, storage_dir: str = "data_store/feedback"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_path = self.storage_dir / "analyst_corrections.json"
        self.chart_overrides_path = self.storage_dir / "chart_overrides.json"
        self.ratings_path = self.storage_dir / "ratings.json"

        self._corrections: List[Dict[str, Any]] = self._load_json(self.corrections_path, default=[])
        self._chart_overrides: Dict[str, Dict[str, Any]] = self._load_json(self.chart_overrides_path)
        self._ratings: List[Dict[str, Any]] = self._load_json(self.ratings_path, default=[])

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
    # 1. Analyst Corrections (Human-in-the-Loop)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower())
        stopwords = {"what", "when", "where", "which", "with", "from", "that", "this", "these", "have", "more", "most", "about"}
        return {w for w in words if w not in stopwords}

    def capture_correction(
        self,
        question: str,
        original_answer: str,
        user_correction: str,
        intent: str = "general",
        user_id: str = "analyst",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Records an analyst correction to an AI-generated answer.
        """
        record = {
            "id": f"corr_{len(self._corrections) + 1:04d}",
            "question": question,
            "tokens": sorted(list(self._tokenize(question))),
            "original_answer": original_answer,
            "user_correction": user_correction,
            "intent": intent,
            "user_id": user_id,
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._corrections.append(record)
        # Cap stored corrections to 500
        if len(self._corrections) > 500:
            self._corrections = self._corrections[-500:]
        self._save_json(self.corrections_path, self._corrections)
        return record

    def get_applicable_corrections(
        self,
        question: str,
        min_jaccard: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Finds previous corrections relevant to the current user question based on token overlap.
        """
        q_tokens = self._tokenize(question)
        if not q_tokens:
            return []

        matched = []
        for corr in self._corrections:
            c_tokens = set(corr.get("tokens", []))
            if not c_tokens:
                c_tokens = self._tokenize(corr.get("question", ""))
            if not c_tokens:
                continue

            intersection = q_tokens & c_tokens
            union = q_tokens | c_tokens
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard >= min_jaccard or len(intersection) >= 3:
                matched.append({
                    "correction": corr,
                    "jaccard_score": round(jaccard, 3),
                    "overlapping_tokens": sorted(list(intersection)),
                })

        # Rank by relevance
        matched.sort(key=lambda m: m["jaccard_score"], reverse=True)
        return [m["correction"] for m in matched[:3]]

    def apply_learned_corrections(
        self,
        draft_answer: str,
        applicable_corrections: List[Dict[str, Any]],
    ) -> str:
        """
        Enriches or refines the draft AI answer using historical analyst guidance.
        """
        if not applicable_corrections:
            return draft_answer

        top_corr = applicable_corrections[0]
        correction_text = top_corr.get("user_correction", "").strip()

        # If high-fidelity explicit correction exists, append institutional analyst note
        analyst_note = f"\n\n💡 *Analyst Knowledge Guidance*: An analyst previously noted: \"{correction_text}\""
        if analyst_note not in draft_answer:
            return draft_answer + analyst_note
        return draft_answer

    # ─────────────────────────────────────────────────────────────
    # 2. Chart Override & Preference Adaptation
    # ─────────────────────────────────────────────────────────────

    def capture_chart_override(
        self,
        metric: str,
        default_recommended: str,
        user_chosen: str,
    ) -> None:
        """
        Learns when an analyst overrides a default chart recommendation for a given metric.
        """
        m_key = metric.lower().strip()
        if m_key not in self._chart_overrides:
            self._chart_overrides[m_key] = {"overrides": {}, "total_overrides": 0}

        overrides = self._chart_overrides[m_key]["overrides"]
        overrides[user_chosen] = overrides.get(user_chosen, 0) + 1
        self._chart_overrides[m_key]["total_overrides"] += 1
        self._chart_overrides[m_key]["last_override"] = datetime.now(timezone.utc).isoformat()
        self._save_json(self.chart_overrides_path, self._chart_overrides)

    def get_adjusted_chart_recommendation(
        self, metric: str, default_recommended: str, min_votes: int = 2
    ) -> str:
        """
        Returns the user's preferred chart type if a strong preference pattern exists.
        """
        m_key = metric.lower().strip()
        if m_key not in self._chart_overrides:
            return default_recommended

        overrides = self._chart_overrides[m_key].get("overrides", {})
        if not overrides:
            return default_recommended

        top_choice, count = max(overrides.items(), key=lambda x: x[1])
        if count >= min_votes:
            return top_choice
        return default_recommended

    # ─────────────────────────────────────────────────────────────
    # 3. Satisfaction Ratings & Quality Metrics
    # ─────────────────────────────────────────────────────────────

    def capture_rating(
        self,
        surface: str,
        target_id: str,
        rating: str,  # "positive" | "negative"
        feedback_text: Optional[str] = None,
        user_id: str = "analyst",
    ) -> Dict[str, Any]:
        """
        Logs user satisfaction for a specific output (Ask Your Data, Report, Insights).
        """
        rec = {
            "surface": surface,
            "target_id": target_id,
            "rating": rating.lower(),
            "feedback_text": feedback_text or "",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._ratings.append(rec)
        if len(self._ratings) > 1000:
            self._ratings = self._ratings[-1000:]
        self._save_json(self.ratings_path, self._ratings)
        return rec

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """
        Returns performance and quality tracking metrics.
        """
        total_ratings = len(self._ratings)
        positives = sum(1 for r in self._ratings if r.get("rating") == "positive")
        negatives = sum(1 for r in self._ratings if r.get("rating") == "negative")

        satisfaction_rate = (positives / total_ratings * 100) if total_ratings > 0 else 100.0

        return {
            "total_ratings": total_ratings,
            "positive_count": positives,
            "negative_count": negatives,
            "satisfaction_rate": round(satisfaction_rate, 1),
            "total_corrections_captured": len(self._corrections),
            "total_chart_overrides": sum(
                v.get("total_overrides", 0) for v in self._chart_overrides.values()
            ),
        }
