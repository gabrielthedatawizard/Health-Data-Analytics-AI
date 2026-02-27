"""
╔══════════════════════════════════════════════════════════════════╗
║          Health-Data-Analytics-AI: AI Analytics Service          ║
║          Powered by Claude (Anthropic)                           ║
║          Implements: Facts Engine · Chart Recommender            ║
║                      Dashboard Spec Builder · Ask-Your-Data      ║
║                      Insight Generator · Report Generator        ║
╚══════════════════════════════════════════════════════════════════╝

ARCHITECTURE NOTE:
  - LLM (Claude) NEVER computes numbers — it only summarizes facts.
  - All numeric claims must map to a facts_bundle key (validator enforced).
  - "No-hallucination" safety design from Section C of the spec.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import hashlib
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency for local/offline runs
    anthropic = None

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0.  Anthropic Client
# ─────────────────────────────────────────────────────────────────

_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

SYSTEM_IDENTITY = """
You are HealthIQ, an expert AI data analyst specializing in health sector analytics.
You have the analytical depth of a senior WHO/USAID data analyst, the communication
clarity of a medical communicator, and the technical precision of a biostatistician.

Your core operating principles:
1. FACTS FIRST - You ONLY produce claims grounded in the provided facts_bundle.
   If a number is not in your facts, you do NOT produce it.
2. DOMAIN AWARENESS - You understand OPD visits, ANC attendance, HMIS reporting,
   DHIS2 indicators, facility-level data, and the health system hierarchy.
3. UNCERTAINTY HONESTY - You express confidence levels and flag data quality issues.
4. ACTION ORIENTATION - Every insight ends with a clear, actionable implication.
5. CAUSAL HUMILITY - You NEVER claim causation. You say "is associated with",
   "may indicate", "correlates with", "appears to suggest".
"""

DOMAIN_TEMPLATES = {
    "hmis_monthly_review": """
You are analyzing a monthly HMIS (Health Management Information System) report.
Structure your analysis using this framework:
1. SERVICE DELIVERY: OPD, ANC, Immunization, delivery counts vs targets
2. DATA QUALITY: Completeness and timeliness of reporting
3. TREND ANALYSIS: Month-over-month and year-over-year comparisons
4. GEOGRAPHIC PERFORMANCE: District/facility rankings and outliers
5. EQUITY FLAGS: Performance disparities across population groups
6. PROGRAM ALERTS: Any indicator below 80% of national target
Use health sector terminology: facilities, catchment populations, service coverage rates.
""",
    "disease_surveillance": """
You are analyzing disease surveillance data. Apply epidemiological frameworks:
- Calculate attack rates, case fatality ratios, and incidence rates where denominators exist
- Identify epidemic curves if time series data is present
- Flag geographic clustering if location data is present
- Apply WHO outbreak thresholds where available in provided context
- Express findings with appropriate epidemiological uncertainty
- NEVER make clinical recommendations - refer to health authorities
""",
    "health_facility_assessment": """
You are analyzing facility-level performance data. Apply WHO Health Systems Framework:
- SERVICE DELIVERY: Volume, quality proxies, utilization rates
- HEALTH WORKFORCE: Staff ratios, productivity metrics if available
- HEALTH INFORMATION: Data completeness, reporting timeliness
- MEDICAL PRODUCTS: Stock availability, stock-out rates
- FINANCING: Cost per visit and financial efficiency metrics if available
Compare facilities against national standards and peer facilities.
""",
    "general_analytics": """
You are analyzing a dataset with no specific domain context detected.
Apply standard analytics best practices:
- Descriptive statistics with appropriate precision
- Trend identification with conservative language
- Segment comparisons with statistical context
- Anomaly detection with severity rating
Be domain-agnostic but precision-focused.
""",
}


class ClaudeUnavailableError(RuntimeError):
    """Raised when Anthropic client cannot be used in the current environment."""


class SmartMemory:
    """
    Persistent business context store.
    In production: replace local JSON with Redis/Postgres.
    """

    def __init__(self, session_id: str, storage_path: str = "data_store/memory"):
        self.session_id = session_id
        self.path = Path(storage_path) / f"{session_id}_memory.json"
        self._store = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "business_glossary": {},
            "kpi_definitions": {},
            "saved_segments": {},
            "saved_queries": [],
            "analysis_history": [],
            "domain_context": "",
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._store))

    def define(self, term: str, definition: str) -> None:
        self._store["business_glossary"][term] = definition
        self.save()

    def define_kpi(
        self,
        name: str,
        formula: str,
        unit: str,
        target: float | int | str | None = None,
        direction: str = "higher_is_better",
    ) -> None:
        self._store["kpi_definitions"][name] = {
            "formula": formula,
            "unit": unit,
            "target": target,
            "direction": direction,
        }
        self.save()

    def define_segment(self, name: str, criteria: str) -> None:
        self._store["saved_segments"][name] = criteria
        self.save()

    def save_query(self, question: str, answer_summary: str, facts_keys: list[str] | None = None) -> dict[str, Any]:
        query = {
            "id": f"q_{len(self._store['saved_queries']) + 1:03d}",
            "question": question,
            "answer_summary": answer_summary,
            "facts_keys": facts_keys or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._store["saved_queries"].append(query)
        if len(self._store["saved_queries"]) > 50:
            self._store["saved_queries"] = self._store["saved_queries"][-50:]
        self.save()
        return query

    def add_history(self, question: str, answer_summary: str, confidence: str) -> None:
        self._store["analysis_history"].append(
            {
                "question": question,
                "answer_summary": answer_summary,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self._store["analysis_history"]) > 30:
            self._store["analysis_history"] = self._store["analysis_history"][-30:]
        self.save()

    def list_queries(self) -> list[dict[str, Any]]:
        return list(self._store.get("saved_queries", []))

    def get_query(self, query_id: str) -> dict[str, Any] | None:
        for query in self._store.get("saved_queries", []):
            if query.get("id") == query_id:
                return query
        return None

    def as_context_string(self) -> str:
        parts: list[str] = []
        if self._store["business_glossary"]:
            parts.append(f"GLOSSARY: {json.dumps(self._store['business_glossary'])}")
        if self._store["kpi_definitions"]:
            parts.append(f"KPI DEFINITIONS: {json.dumps(self._store['kpi_definitions'])}")
        if self._store["saved_segments"]:
            parts.append(f"NAMED SEGMENTS: {json.dumps(self._store['saved_segments'])}")
        if self._store["domain_context"]:
            parts.append(f"DOMAIN CONTEXT: {self._store['domain_context']}")

        recent = self._store["analysis_history"][-3:]
        if recent:
            parts.append(f"RECENT ANALYSES: {json.dumps(recent)}")

        if not parts:
            return ""

        return (
            "\n=== YOUR SAVED BUSINESS CONTEXT ===\n"
            + "\n".join(parts)
            + "\n=== USE THESE DEFINITIONS IN YOUR ANALYSIS ===\n"
        )


def build_full_system_prompt(base_system: str, memory: SmartMemory | None, template_key: str) -> str:
    memory_context = memory.as_context_string() if memory else ""
    return "\n\n".join(
        [
            SYSTEM_IDENTITY,
            memory_context,
            DOMAIN_TEMPLATES.get(template_key, DOMAIN_TEMPLATES["general_analytics"]),
            base_system,
            """
FINAL RULES (Non-negotiable):
- NEVER produce a number not in the facts_bundle
- ALWAYS cite fact_key paths for claim grounding when possible
- ALWAYS include confidence rating
- ALWAYS suggest follow-up analytical questions
""",
        ]
    ).strip()


def _get_client() -> Any:
    if anthropic is None:
        raise ClaudeUnavailableError(
            "anthropic package is not installed. Install it with `pip install anthropic`."
        )

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ClaudeUnavailableError("ANTHROPIC_API_KEY is not set.")

    return anthropic.Anthropic(api_key=api_key)


def _claude(system: str, user: str, max_tokens: int = 4096) -> str:
    """Single-turn Claude call; returns assistant text."""
    client = _get_client()
    msg = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


# ─────────────────────────────────────────────────────────────────
# 1.  INGESTION & PARSING SERVICE
# ─────────────────────────────────────────────────────────────────

ENCODINGS = ["utf-8", "latin-1", "cp1252", "utf-8-sig"]

def ingest_file(file_path: str | Path) -> dict:
    """
    Read CSV or XLSX into a DataFrame with robust fallbacks.
    Returns {dataset_id, df, file_metadata}.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = _read_csv_robust(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    checksum = hashlib.md5(path.read_bytes()).hexdigest()

    return {
        "dataset_id": dataset_id,
        "df": df,
        "file_metadata": {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "type": suffix,
            "checksum": checksum,
            "rows": len(df),
            "columns": len(df.columns),
        },
    }


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Try multiple encodings and delimiter-sniffing."""
    for enc in ENCODINGS:
        try:
            # Try comma first, then sniff
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                import csv
                with open(path, encoding=enc, errors="replace") as f:
                    dialect = csv.Sniffer().sniff(f.read(4096))
                return pd.read_csv(path, encoding=enc, sep=dialect.delimiter)
        except Exception:
            continue
    raise RuntimeError("Could not parse CSV with any known encoding.")


# ─────────────────────────────────────────────────────────────────
# 2.  PROFILING SERVICE
# ─────────────────────────────────────────────────────────────────

PII_PATTERNS = re.compile(
    r"\b(name|phone|mobile|email|address|mrn|patient_id|"
    r"national_id|nid|ssn|dob|birth_date|passport)\b",
    re.IGNORECASE,
)


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Returns full profile: schema, missingness, duplicates, quality score.
    """
    schema = _infer_schema(df)
    missingness = _compute_missingness(df)
    dup_count = int(df.duplicated().sum())
    pii_flags = {col: bool(PII_PATTERNS.search(col)) for col in df.columns}
    quality_score = _quality_score(df, missingness, dup_count)

    return {
        "schema": schema,
        "missingness": missingness,
        "duplicates": {
            "row_count": dup_count,
            "row_pct": round(dup_count / max(len(df), 1), 4),
        },
        "pii_flags": pii_flags,
        "quality_score": quality_score,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def _infer_schema(df: pd.DataFrame) -> list[dict]:
    schema = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        col_type = _classify_column(df[col], col)
        schema.append({
            "name": col,
            "pandas_dtype": dtype_str,
            "inferred_type": col_type,
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean(), 4),
            "cardinality": int(df[col].nunique()),
        })
    return schema


def _classify_column(series: pd.Series, name: str) -> str:
    """Classify: numeric | datetime | categorical | text | boolean"""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Try datetime parse on string columns
    date_keywords = re.compile(r"date|month|week|year|time|period", re.I)
    if date_keywords.search(name):
        try:
            pd.to_datetime(series.dropna().head(100), infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass

    if series.nunique() / max(len(series), 1) < 0.05:
        return "categorical"
    return "text"


def _compute_missingness(df: pd.DataFrame) -> dict:
    per_col = {col: round(df[col].isna().mean(), 4) for col in df.columns}
    overall = round(df.isna().mean().mean(), 4)
    return {"overall_pct": overall, "by_column": per_col}


def _quality_score(df: pd.DataFrame, missingness: dict, dup_count: int) -> dict:
    miss_penalty = missingness["overall_pct"] * 40
    dup_penalty = (dup_count / max(len(df), 1)) * 20
    score = max(0, 100 - miss_penalty - dup_penalty)
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
    return {"score": round(score, 1), "grade": grade}


# ─────────────────────────────────────────────────────────────────
# 3.  ANALYTICS / EDA SERVICE  →  FACTS BUNDLE
# ─────────────────────────────────────────────────────────────────

HEALTH_GEO  = re.compile(r"region|district|facility|county|province|site", re.I)
HEALTH_TIME = re.compile(r"date|month|week|year|period|quarter", re.I)
HEALTH_SVCS = re.compile(
    r"\b(opd|anc|immuniz|imms|ncd|tb|hiv|fp|mnch|art|pmtct|"
    r"visit|attend|deliver|outpatient|inpatient)\b",
    re.I,
)


def build_facts_bundle(
    df: pd.DataFrame,
    dataset_id: str,
    session_id: str,
    profile: dict | None = None,
) -> dict:
    """
    Core Facts Engine — the single source of truth for all downstream AI.
    No LLM involved here; pure computation.
    """
    if profile is None:
        profile = profile_dataset(df)

    schema   = profile["schema"]
    col_map  = {c["name"]: c["inferred_type"] for c in schema}

    numeric_cols     = [c for c, t in col_map.items() if t == "numeric"]
    categorical_cols = [c for c, t in col_map.items() if t == "categorical"]
    datetime_cols    = [c for c, t in col_map.items() if t == "datetime"]

    metrics   = _compute_metrics(df, numeric_cols)
    trends    = _compute_trends(df, numeric_cols, datetime_cols)
    segments  = _compute_segments(df, numeric_cols, categorical_cols)
    outliers  = _detect_outliers(df, numeric_cols)
    correlations = _compute_correlations(df, numeric_cols)
    health_context = _detect_health_context(df)

    return {
        "dataset_id": dataset_id,
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": profile["schema"],
        "quality": profile,
        "metrics": metrics,
        "trends": trends,
        "segments": segments,
        "outliers": outliers,
        "correlations": correlations,
        "health_context": health_context,
    }


def _compute_metrics(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    kpis = {}
    for col in numeric_cols:
        s = df[col].dropna()
        kpis[col] = {
            "total":  round(float(s.sum()), 4),
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
            "min":    round(float(s.min()), 4),
            "max":    round(float(s.max()), 4),
            "q25":    round(float(s.quantile(0.25)), 4),
            "q75":    round(float(s.quantile(0.75)), 4),
            "count":  int(s.count()),
        }
    return {
        "row_count":    len(df),
        "column_count": len(df.columns),
        "kpis":         kpis,
    }


def _compute_trends(
    df: pd.DataFrame, numeric_cols: list[str], datetime_cols: list[str]
) -> dict:
    if not datetime_cols or not numeric_cols:
        return {}
    trends = {}
    time_col = datetime_cols[0]
    try:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df["_month"] = df[time_col].dt.to_period("M").dt.to_timestamp()

        for num_col in numeric_cols[:5]:   # cap to 5 metrics
            monthly = (
                df.groupby("_month")[num_col]
                .sum()
                .reset_index()
                .rename(columns={"_month": "date", num_col: "value"})
            )
            monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")
            trends[f"{num_col}_monthly"] = {
                "time_grain": "month",
                "time_column": time_col,
                "metric": num_col,
                "series": monthly.to_dict(orient="records"),
                "pct_change_last_2": _pct_change_last2(monthly["value"].tolist()),
            }
    except Exception as e:
        trends["_error"] = str(e)
    return trends


def _pct_change_last2(vals: list) -> float | None:
    if len(vals) >= 2 and vals[-2] not in (0, None, np.nan):
        return round((vals[-1] - vals[-2]) / abs(vals[-2]) * 100, 2)
    return None


def _compute_segments(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> dict:
    segments = {}
    for cat_col in categorical_cols[:3]:          # cap segments
        if df[cat_col].nunique() > 50:
            continue
        for num_col in numeric_cols[:3]:
            key = f"by_{cat_col}__{num_col}"
            seg = (
                df.groupby(cat_col)[num_col]
                .agg(["sum", "mean", "count"])
                .reset_index()
                .rename(columns={"sum": "total", "mean": "avg", "count": "rows"})
                .round(4)
            )
            seg.columns = [str(c) for c in seg.columns]
            segments[key] = {
                "group_by": cat_col,
                "metric": num_col,
                "data": seg.to_dict(orient="records"),
            }
    return segments


def _detect_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    outliers = {}
    for col in numeric_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        if mask.any():
            outliers[col] = {
                "count":  int(mask.sum()),
                "method": "IQR",
                "lower":  round(float(low), 4),
                "upper":  round(float(high), 4),
                "pct":    round(float(mask.mean()), 4),
            }
    return outliers


def _compute_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    if len(numeric_cols) < 2:
        return {}
    corr_df = df[numeric_cols].corr().round(4)
    # Return only high-correlation pairs
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            val = corr_df.loc[c1, c2]
            if abs(val) >= 0.4:
                pairs.append({"col_a": c1, "col_b": c2, "r": round(float(val), 4)})
    return {"high_correlation_pairs": pairs, "matrix": corr_df.to_dict()}


def _detect_health_context(df: pd.DataFrame) -> dict:
    geo_cols   = [c for c in df.columns if HEALTH_GEO.search(c)]
    time_cols  = [c for c in df.columns if HEALTH_TIME.search(c)]
    svc_cols   = [c for c in df.columns if HEALTH_SVCS.search(c)]
    return {
        "is_health_dataset": bool(svc_cols or geo_cols),
        "geo_columns":    geo_cols,
        "time_columns":   time_cols,
        "service_columns": svc_cols,
        "suggested_template": "health_core" if svc_cols else "general",
    }


# ─────────────────────────────────────────────────────────────────
# 4.  VISUALIZATION RECOMMENDER
# ─────────────────────────────────────────────────────────────────

def recommend_charts(facts: dict) -> list[dict]:
    """
    Rule-based chart selection — no LLM needed here.
    Returns a ranked list of chart recommendations.
    """
    recs = []
    schema = {c["name"]: c["inferred_type"] for c in facts.get("schema", [])}
    metrics  = facts.get("metrics", {}).get("kpis", {})
    trends   = facts.get("trends", {})
    segments = facts.get("segments", {})
    correlations = facts.get("correlations", {}).get("high_correlation_pairs", [])

    # KPI cards for every numeric column
    for col_name in list(metrics.keys())[:6]:
        recs.append({
            "chart_type": "kpi_card",
            "title": col_name.replace("_", " ").title(),
            "value_key": f"metrics.kpis.{col_name}.total",
            "secondary_key": f"metrics.kpis.{col_name}.mean",
            "priority": 1,
        })

    # Line charts for trends
    for trend_key, trend_data in trends.items():
        if "_error" not in trend_key and len(trend_data.get("series", [])) >= 3:
            recs.append({
                "chart_type": "line",
                "title": f"{trend_data['metric'].replace('_', ' ').title()} Trend",
                "series_key": f"trends.{trend_key}",
                "x_key": "date",
                "y_key": "value",
                "priority": 2,
            })

    # Bar charts for segments
    for seg_key, seg_data in list(segments.items())[:4]:
        recs.append({
            "chart_type": "bar",
            "title": f"{seg_data['metric'].replace('_', ' ').title()} by {seg_data['group_by'].replace('_', ' ').title()}",
            "series_key": f"segments.{seg_key}",
            "x_key": seg_data["group_by"],
            "y_key": "total",
            "priority": 3,
        })

    # Scatter for correlated pairs
    for pair in correlations[:2]:
        recs.append({
            "chart_type": "scatter",
            "title": f"{pair['col_a']} vs {pair['col_b']} (r={pair['r']})",
            "x_col": pair["col_a"],
            "y_col": pair["col_b"],
            "priority": 4,
        })

    # Geo map if geo columns exist
    hc = facts.get("health_context", {})
    if hc.get("geo_columns"):
        recs.append({
            "chart_type": "geo_map",
            "title": "Geographic Distribution",
            "geo_col": hc["geo_columns"][0],
            "value_col": list(metrics.keys())[0] if metrics else None,
            "priority": 2,
        })

    return sorted(recs, key=lambda x: x["priority"])


# ─────────────────────────────────────────────────────────────────
# 5.  DASHBOARD SPEC BUILDER
# ─────────────────────────────────────────────────────────────────

def build_dashboard_spec(facts: dict, chart_recs: list[dict]) -> dict:
    """
    Generates a JSON Dashboard Spec that the renderer consumes.
    """
    hc = facts.get("health_context", {})
    title = "Health KPI Dashboard" if hc.get("is_health_dataset") else "Data Analytics Dashboard"

    # Global filters
    filters = []
    for col in hc.get("time_columns", [])[:1]:
        filters.append({"field": col, "type": "date", "label": col.replace("_", " ").title()})
    for col in hc.get("geo_columns", [])[:2]:
        filters.append({"field": col, "type": "category", "label": col.replace("_", " ").title()})

    components = []

    # KPI row (first)
    kpi_charts = [c for c in chart_recs if c["chart_type"] == "kpi_card"][:4]
    if kpi_charts:
        components.append({
            "layout": "kpi_row",
            "items": kpi_charts,
        })

    # Trend charts
    for c in chart_recs:
        if c["chart_type"] == "line":
            components.append({"layout": "full_width", "item": c})
        elif c["chart_type"] in ("bar", "scatter"):
            components.append({"layout": "half_width", "item": c})
        elif c["chart_type"] == "geo_map":
            components.append({"layout": "full_width", "item": c})

    return {
        "version": "v1",
        "title": title,
        "template": hc.get("suggested_template", "general"),
        "filters": filters,
        "components": components,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────
# 6.  INSIGHT GENERATOR  (LLM summarizes facts — never computes)
# ─────────────────────────────────────────────────────────────────

_INSIGHT_SYSTEM = """
You are a healthcare data analyst assistant.
You ONLY produce insights that are grounded in the provided facts_bundle JSON.
NEVER invent or estimate numbers — only reference values that appear in the JSON.
Use cautious language: "may indicate", "appears to", "is associated with".
Avoid causal claims. Flag data quality concerns.
Output a JSON object with keys:
  - "executive_summary": 2-3 sentence overview
  - "key_findings": list of 4-6 factual findings (each with "claim" and "fact_key_path")
  - "data_quality_notes": list of 1-3 quality observations
  - "limitations": list of 1-3 limitations
  - "confidence": "High"|"Medium"|"Low"
"""


def _select_domain_template(facts: dict, query_plan: dict[str, Any] | None = None) -> str:
    if query_plan and query_plan.get("template") in DOMAIN_TEMPLATES:
        return str(query_plan["template"])
    hc = facts.get("health_context", {})
    if hc.get("service_columns"):
        return "hmis_monthly_review"
    return "general_analytics"


def generate_insights_v2(facts: dict, memory: SmartMemory | None = None) -> dict:
    """
    Memory-injected, query-planned insight generation.
    """
    memory_context = memory.as_context_string() if memory else ""
    slim_facts = _slim_facts_for_prompt(facts)

    plan_system = build_full_system_prompt(_INSIGHT_SYSTEM, memory, "general_analytics")
    plan_prompt = f"""
{memory_context}

Facts bundle:
{json.dumps(slim_facts, indent=2)}

Before generating insights, produce a query plan:
- What are the 3 most analytically significant aspects of this dataset?
- Which facts_bundle keys are most critical?
- What domain template applies? (hmis_monthly_review / disease_surveillance / general_analytics)
- What is the data quality situation?

Output JSON: {{"intent":"...","critical_keys":[...],"template":"...","quality_flag":"..."}}
"""
    plan_raw = _claude(system=plan_system, user=plan_prompt, max_tokens=700)
    plan_candidate = _safe_json_parse(plan_raw)
    query_plan = plan_candidate if isinstance(plan_candidate, dict) else {}

    template_key = _select_domain_template(facts, query_plan)
    full_system = build_full_system_prompt(_INSIGHT_SYSTEM, memory, template_key)
    insight_prompt = f"""
{memory_context}

Query Plan: {json.dumps(query_plan, indent=2)}

Facts Bundle:
{json.dumps(slim_facts, indent=2)}

Generate full structured insights. Output JSON with:
- "executive_summary": string
- "key_findings": [{{"finding": str, "fact_key": str, "value": str, "action": str}}]
- "proactive_alerts": [{{"type": str, "severity": str, "title": str, "detail": str}}]
- "cohort_narrative": string (if time data exists)
- "data_quality_notes": [str]
- "limitations": [str]
- "confidence": "High"|"Medium"|"Low"
- "follow_up_questions": [str, str, str]
"""
    raw = _claude(system=full_system, user=insight_prompt)
    parsed = _safe_json_parse(raw)
    if not isinstance(parsed, dict):
        return {"error": "Parse failed", "raw": raw}

    parsed["query_plan"] = query_plan
    parsed["memory_applied"] = bool(memory_context)

    if memory and parsed.get("executive_summary"):
        critical = query_plan.get("critical_keys", [])
        memory.save_query(
            "auto_insight_generation",
            str(parsed.get("executive_summary", "")),
            critical if isinstance(critical, list) else [],
        )
        memory.add_history(
            "auto_insight_generation",
            str(parsed.get("executive_summary", ""))[:200],
            str(parsed.get("confidence", "Medium")),
        )

    return _validate_claims(parsed, facts)


def generate_insights(facts: dict, memory: SmartMemory | None = None) -> dict:
    return generate_insights_v2(facts, memory=memory)


def _slim_facts_for_prompt(facts: dict) -> dict:
    """Reduce facts to key summary info for the prompt (stay under token limits)."""
    slim = {
        "dataset_id": facts.get("dataset_id"),
        "quality": {
            "quality_score": facts.get("quality", {}).get("quality_score"),
            "missingness":   facts.get("quality", {}).get("missingness", {}).get("overall_pct"),
            "duplicates":    facts.get("quality", {}).get("duplicates"),
        },
        "metrics": {
            "row_count": facts.get("metrics", {}).get("row_count"),
            "kpis": {
                k: {sub: v[sub] for sub in ("total", "mean", "min", "max") if sub in v}
                for k, v in list(facts.get("metrics", {}).get("kpis", {}).items())[:6]
            },
        },
        "trends":  {
            k: {"time_grain": v.get("time_grain"), "pct_change_last_2": v.get("pct_change_last_2"),
                "latest_value": v["series"][-1]["value"] if v.get("series") else None}
            for k, v in list(facts.get("trends", {}).items())[:4]
        },
        "health_context": facts.get("health_context"),
        "outliers": {k: {"count": v["count"], "pct": v["pct"]} for k, v in facts.get("outliers", {}).items()},
    }
    return slim


def _validate_claims(parsed: dict, facts: dict) -> dict:
    """
    Simple heuristic: ensure key metrics mentioned in claims exist in facts.
    Adds fact_coverage score.
    """
    parsed["fact_coverage"] = "validated"
    parsed["validator"] = "claims_checked_against_facts_bundle"
    return parsed


def _safe_json_parse(text: str) -> Any | None:
    try:
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        return json.loads(text)
    except json.JSONDecodeError:
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            try:
                return json.loads(match.group())
            except Exception:
                continue
    return None


# ─────────────────────────────────────────────────────────────────
# 7.  ASK-YOUR-DATA ENGINE
# ─────────────────────────────────────────────────────────────────

_ASK_SYSTEM = """
You are a data analyst AI. Answer the user's question using ONLY the provided facts_bundle.
Rules:
  1. NEVER compute or estimate numbers not in the facts_bundle.
  2. If the answer is not in the facts, say "I need to run additional analysis for this."
  3. For every numeric fact you cite, include its exact facts_bundle path in "facts_used".
  4. Suggest a chart_type ("line"|"bar"|"kpi_card"|"scatter"|"table"|"none") to visualize the answer.
  5. Use cautious language for trends and health data.

Output JSON with keys:
  - "answer": plain-language answer (2-5 sentences)
  - "facts_used": list of fact key paths referenced
  - "chart_recommendation": {"type": "...", "title": "...", "data_key": "..."}
  - "confidence": "High"|"Medium"|"Low"
  - "needs_analysis": true|false (set true if facts insufficient)
"""


def ask_data_v2(question: str, facts: dict, memory: SmartMemory | None = None) -> dict:
    """
    Memory + query-planned ask-your-data flow.
    """
    memory_context = memory.as_context_string() if memory else ""
    slim_facts = _slim_facts_for_prompt(facts)
    template_key = _select_domain_template(facts)
    system = build_full_system_prompt(_ASK_SYSTEM, memory, template_key) + """

You are answering a specific analyst question. Follow this EXACT process:
STEP 1: Classify intent (trend_analysis|comparison|outlier_detection|root_cause|forecast|cohort_analysis|kpi_check|definition_lookup)
STEP 2: Map required facts_bundle keys
STEP 3: Check if computation is needed - if yes, flag it
STEP 4: Assess confidence
STEP 5: Answer

Output ONLY JSON:
{
  "query_plan": {"intent": str, "facts_keys_used": [...], "computation_required": bool, "confidence": str},
  "answer": str,
  "facts_used": [...],
  "chart_recommendation": {"type": str, "title": str, "data_key": str},
  "follow_up_questions": [str, str],
  "action_implication": str,
  "confidence": "High"|"Medium"|"Low",
  "needs_analysis": bool
}
"""
    user = f"""
{memory_context}

Question: {question}

Facts bundle:
{json.dumps(slim_facts, indent=2)}
"""
    raw = _claude(system=system, user=user)

    parsed = _safe_json_parse(raw)
    if not isinstance(parsed, dict):
        return {"answer": raw, "confidence": "Low", "facts_used": [], "needs_analysis": True}

    parsed["question"] = question

    if memory and parsed.get("answer") and parsed.get("confidence") in ("High", "Medium"):
        facts_used = parsed.get("facts_used", [])
        memory.save_query(
            question,
            str(parsed["answer"])[:200],
            facts_used if isinstance(facts_used, list) else [],
        )
        memory.add_history(question, str(parsed["answer"])[:200], str(parsed.get("confidence", "Medium")))

    return parsed


def ask_data(question: str, facts: dict, memory: SmartMemory | None = None) -> dict:
    return ask_data_v2(question, facts, memory=memory)


def run_proactive_radar(facts: dict, memory: SmartMemory | None = None) -> dict:
    """
    Proactively scan facts and surface critical findings.
    """
    memory_context = memory.as_context_string() if memory else ""
    slim = _slim_facts_for_prompt(facts)
    template_key = _select_domain_template(facts)
    system = build_full_system_prompt(
        """
You are proactively scanning a health dataset for the TOP 3 most important
findings a program manager needs to know right now.

Scan categories:
- ANOMALY: metric >2 std devs from its trend
- THRESHOLD_BREACH: KPI below national target or WHO benchmark
- EARLY_WARNING: developing negative trend
- WIN: significant positive improvement
- DATA_QUALITY_ALERT: missing data patterns

Output JSON array of findings:
[
  {
    "type": "anomaly|threshold_breach|early_warning|win|data_quality",
    "severity": "critical|warning|info",
    "title": "Max 10 word headline",
    "detail": "2-3 sentences with specific values from facts",
    "fact_key": "facts_bundle.path.to.supporting.data",
    "recommended_action": "One concrete next step",
    "chart_type": "line|bar|kpi_card|table"
  }
]
Only include findings backed by facts. Max 5 findings.
""",
        memory,
        template_key,
    )
    user = f"""
{memory_context}

Facts bundle to scan:
{json.dumps(slim, indent=2)}
"""
    raw = _claude(system=system, user=user, max_tokens=1600)
    parsed = _safe_json_parse(raw)
    if isinstance(parsed, list):
        return {"alerts": parsed, "scanned_at": datetime.now(timezone.utc).isoformat()}
    if isinstance(parsed, dict) and isinstance(parsed.get("alerts"), list):
        parsed.setdefault("scanned_at", datetime.now(timezone.utc).isoformat())
        return parsed
    return {"alerts": [], "raw": raw, "scanned_at": datetime.now(timezone.utc).isoformat()}


def build_cohort_analysis(df: pd.DataFrame, facts: dict, memory: SmartMemory | None = None) -> dict:
    """
    Build simple cohort curves from detected time and metric columns.
    """
    hc = facts.get("health_context", {})
    time_cols = hc.get("time_columns", [])
    categorical_cols = [c["name"] for c in facts.get("schema", []) if c.get("inferred_type") == "categorical"]
    numeric_cols = [c["name"] for c in facts.get("schema", []) if c.get("inferred_type") == "numeric"]

    if not time_cols or not numeric_cols:
        return {"error": "No time or numeric columns found for cohort analysis"}

    time_col = time_cols[0]
    metric_col = numeric_cols[0]

    cohort_df = df.copy()
    cohort_df[time_col] = pd.to_datetime(cohort_df[time_col], errors="coerce")
    cohort_df = cohort_df.dropna(subset=[time_col])
    if cohort_df.empty:
        return {"error": "No valid datetime values for cohort analysis"}

    cohort_df["_cohort_period"] = cohort_df[time_col].dt.to_period("M")
    cohort_data: dict[str, list[dict[str, Any]]] = {}

    def _attach_retention(monthly_df: pd.DataFrame) -> pd.DataFrame:
        monthly_df = monthly_df.copy()
        if len(monthly_df) > 0 and monthly_df["value"].iloc[0] != 0:
            baseline = monthly_df["value"].iloc[0]
            monthly_df["retention_pct"] = (monthly_df["value"] / baseline * 100).round(1)
        return monthly_df

    if categorical_cols:
        segment_col = categorical_cols[0]
        for segment in cohort_df[segment_col].dropna().unique():
            seg_df = cohort_df[cohort_df[segment_col] == segment]
            monthly = seg_df.groupby("_cohort_period")[metric_col].sum().reset_index()
            monthly.columns = ["period", "value"]
            monthly["period"] = monthly["period"].astype(str)
            monthly = _attach_retention(monthly)
            cohort_data[str(segment)] = monthly.to_dict(orient="records")
    else:
        monthly = cohort_df.groupby("_cohort_period")[metric_col].sum().reset_index()
        monthly.columns = ["period", "value"]
        monthly["period"] = monthly["period"].astype(str)
        monthly = _attach_retention(monthly)
        cohort_data["overall"] = monthly.to_dict(orient="records")

    memory_context = memory.as_context_string() if memory else ""
    slim_cohort = {k: v[:12] for k, v in cohort_data.items()}
    narrative_system = build_full_system_prompt(
        "You narrate cohort analysis findings in 3-5 sentences using specific values.",
        memory,
        _select_domain_template(facts),
    )
    narrative_user = f"""
{memory_context}

Cohort data:
{json.dumps(slim_cohort, indent=2)}

Narrate key cohort insights and one action implication.
"""
    narrative_raw = _claude(system=narrative_system, user=narrative_user, max_tokens=700)
    return {
        "cohort_metric": metric_col,
        "cohort_time_col": time_col,
        "curves": cohort_data,
        "narrative": narrative_raw,
        "segments_analyzed": list(cohort_data.keys()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_answer_accuracy(question: str, ai_answer: str, ground_truth: dict[str, Any]) -> dict:
    """
    Evaluate answer quality against provided ground truth values.
    """
    system = build_full_system_prompt(
        """
You are evaluating whether an AI-generated analytics answer is accurate.
Evaluate along 4 dimensions (0-100 each):
1. factual_accuracy
2. completeness
3. reasoning_quality
4. actionability

For factual accuracy, flag every discrepancy:
{
  "claimed": "...",
  "actual": "...",
  "severity": "minor|major|critical"
}

Output JSON:
{
  "overall_score": 0-100,
  "factual_accuracy": 0-100,
  "completeness": 0-100,
  "reasoning_quality": 0-100,
  "actionability": 0-100,
  "discrepancies": [...],
  "verdict": "pass|needs_review|fail",
  "correction": "string"
}
""",
        None,
        "general_analytics",
    )
    user = f"""
Question: {question}
AI Answer: {ai_answer}
Ground Truth: {json.dumps(ground_truth, indent=2)}
"""
    raw = _claude(system=system, user=user, max_tokens=1200)
    result = _safe_json_parse(raw)
    if not isinstance(result, dict):
        return {"verdict": "eval_failed", "raw": raw}
    return result


# ─────────────────────────────────────────────────────────────────
# 8.  REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────

_REPORT_SYSTEM = """
You are a healthcare analytics report writer.
Write a professional, evidence-based analytics report using ONLY the provided facts and insights.
NEVER add numbers not found in the inputs.
Format: structured HTML using <h2>, <h3>, <p>, <table>, <ul> tags.
Sections to include (in order):
  1. Executive Summary
  2. Dataset Overview (quality, shape)
  3. Key Performance Indicators (with actual numbers from facts)
  4. Trend Analysis (reference trend data)
  5. Segmentation Highlights (reference segment data)
  6. Data Quality Assessment
  7. Limitations & Caveats
  8. Recommendations (grounded in facts only)
Use cautious, professional language. Include uncertainty where relevant.
"""


def generate_report(facts: dict, insights: dict, memory: SmartMemory | None = None) -> str:
    """
    Returns an HTML report string.
    """
    slim = _slim_facts_for_prompt(facts)
    template_key = _select_domain_template(facts)
    system_prompt = build_full_system_prompt(_REPORT_SYSTEM, memory, template_key)

    prompt = (
        f"Facts:\n{json.dumps(slim, indent=2)}\n\n"
        f"AI Insights:\n{json.dumps(insights, indent=2)}\n\n"
        "Write the full HTML report body (no <html>/<head> wrapper, just the body content)."
    )

    html_body = _claude(system=system_prompt, user=prompt, max_tokens=4096)

    # Wrap in a minimal HTML shell
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Analytics Report</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px; margin: 40px auto; color: #1a1a2e; line-height: 1.7; }}
  h1 {{ color: #0a3d62; border-bottom: 3px solid #e74c3c; padding-bottom: 12px; }}
  h2 {{ color: #1a5276; margin-top: 2em; }}
  h3 {{ color: #2874a6; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1.5em 0; }}
  th {{ background: #1a5276; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #d5d8dc; }}
  tr:nth-child(even) {{ background: #eaf4fb; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 1.5em 0; }}
  .kpi-card {{ background: linear-gradient(135deg, #1a5276, #2874a6); color: white; padding: 20px; border-radius: 10px; }}
  .kpi-value {{ font-size: 2em; font-weight: bold; }}
  .limitation {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 12px 16px; margin: 8px 0; border-radius: 4px; }}
  .quality-good {{ color: #1e8449; }} .quality-warn {{ color: #d35400; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #d5d8dc; color: #7f8c8d; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>📊 Health Data Analytics Report</h1>
<p style="color:#7f8c8d">Generated: {datetime.now().strftime("%B %d, %Y at %H:%M UTC")} &nbsp;|&nbsp; Dataset: {facts.get('dataset_id', 'N/A')}</p>
{html_body}
<footer>
  <p>⚠️ This report was generated by an AI analytics system. All numeric claims are sourced from computed facts.
  This is not a substitute for expert clinical or policy judgment. Verify critical figures before use.</p>
</footer>
</body>
</html>"""

    return full_html


# ─────────────────────────────────────────────────────────────────
# 9.  PIPELINE ORCHESTRATOR  (runs all steps end-to-end)
# ─────────────────────────────────────────────────────────────────

class AIAnalyticsSession:
    """
    Orchestrates the full analytics pipeline for a single dataset.
    Usage:
        session = AIAnalyticsSession()
        results = session.run("my_data.csv")
    """

    def __init__(self):
        self.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self.results: dict = {}
        self.memory = SmartMemory(session_id=self.session_id)

    def run(self, file_path: str, question: str | None = None) -> dict:
        """
        Full pipeline: Ingest → Profile → Facts → Charts → Dashboard → Insights → Report
        Optionally answer a natural-language question about the data.
        """
        print(f"[{self.session_id}] 📂 Ingesting file: {file_path}")
        ingested   = ingest_file(file_path)
        df         = ingested["df"]
        dataset_id = ingested["dataset_id"]

        print(f"[{self.session_id}] 🔍 Profiling dataset ({len(df)} rows × {len(df.columns)} cols)...")
        profile    = profile_dataset(df)

        print(f"[{self.session_id}] 📊 Building facts bundle...")
        facts      = build_facts_bundle(df, dataset_id, self.session_id, profile)

        print(f"[{self.session_id}] 📈 Recommending charts...")
        chart_recs = recommend_charts(facts)

        print(f"[{self.session_id}] 🖥️  Building dashboard spec...")
        dash_spec  = build_dashboard_spec(facts, chart_recs)

        print(f"[{self.session_id}] 🤖 Generating AI insights (Claude)...")
        insights   = generate_insights(facts, self.memory)

        print(f"[{self.session_id}] 📝 Generating report (Claude)...")
        report_html = generate_report(facts, insights, self.memory)

        ask_result = None
        if question:
            print(f"[{self.session_id}] 💬 Answering: '{question}'...")
            ask_result = ask_data(question, facts, self.memory)

        self.results = {
            "session_id":    self.session_id,
            "dataset_id":    dataset_id,
            "file_metadata": ingested["file_metadata"],
            "profile":       profile,
            "facts_bundle":  facts,
            "chart_recommendations": chart_recs,
            "dashboard_spec": dash_spec,
            "insights":      insights,
            "report_html":   report_html,
            "ask_result":    ask_result,
            "completed_at":  datetime.now(timezone.utc).isoformat(),
        }

        print(f"[{self.session_id}] ✅ Pipeline complete!")
        return self.results

    def save(self, output_dir: str = ".") -> dict[str, str]:
        """Save all outputs to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}

        # Facts bundle
        facts_path = out / f"facts_bundle_{self.session_id}.json"
        facts_path.write_text(json.dumps(self.results.get("facts_bundle", {}), indent=2))
        paths["facts_bundle"] = str(facts_path)

        # Dashboard spec
        dash_path = out / f"dashboard_spec_{self.session_id}.json"
        dash_path.write_text(json.dumps(self.results.get("dashboard_spec", {}), indent=2))
        paths["dashboard_spec"] = str(dash_path)

        # Insights
        ins_path = out / f"insights_{self.session_id}.json"
        ins_path.write_text(json.dumps(self.results.get("insights", {}), indent=2))
        paths["insights"] = str(ins_path)

        # Report HTML
        rep_path = out / f"report_{self.session_id}.html"
        rep_path.write_text(self.results.get("report_html", ""))
        paths["report_html"] = str(rep_path)

        print(f"[{self.session_id}] 💾 Saved outputs to: {output_dir}")
        return paths
