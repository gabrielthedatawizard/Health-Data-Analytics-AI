"""
FastAPI REST API for the AI analytics service from files (1).zip.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.ai_analytics_service import (
    AIAnalyticsSession,
    ClaudeUnavailableError,
    SmartMemory,
    ask_data,
    build_cohort_analysis,
    build_dashboard_spec,
    build_facts_bundle,
    evaluate_answer_accuracy,
    generate_insights,
    generate_report,
    ingest_file,
    profile_dataset,
    recommend_charts,
    run_proactive_radar,
)

router = APIRouter(tags=["ai-analytics-v1"])

# In-memory session store (replace with Redis/Postgres in production)
_SESSIONS: dict[str, dict[str, Any]] = {}
_MEMORY_PATH = os.getenv("AI_MEMORY_PATH", "data_store/memory")


class SessionCreate(BaseModel):
    name: str = "Untitled Dataset"
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    mode: str = "safe"


class DashboardSpecRequest(BaseModel):
    template: str = "auto"
    include: list[str] = Field(default_factory=lambda: ["kpi", "trend", "bar"])


class ReportRequest(BaseModel):
    template: str = "health_report"
    sections: list[str] = Field(default_factory=lambda: ["quality", "kpis", "trends", "limitations"])


class MemoryDefineRequest(BaseModel):
    term: str
    definition: str


class MemoryDefineKPIRequest(BaseModel):
    name: str
    formula: str
    unit: str
    target: float | str | None = None
    direction: str = "higher_is_better"


class EvalRequest(BaseModel):
    question: str
    ai_answer: str
    ground_truth: dict[str, Any] = Field(default_factory=dict)


@router.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "ai-analytics", "version": "1.0.0"}


@router.post("/api/v1/sessions")
def create_session(body: SessionCreate) -> dict[str, str]:
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    memory = SmartMemory(session_id=session_id, storage_path=_MEMORY_PATH)
    _SESSIONS[session_id] = {
        "session_id": session_id,
        "dataset_id": dataset_id,
        "name": body.name,
        "description": body.description,
        "tags": body.tags,
        "state": "created",
        "df": None,
        "profile": None,
        "facts": None,
        "chart_recs": None,
        "dashboard_spec": None,
        "insights": None,
        "report_html": None,
        "memory": memory,
    }
    return {"session_id": session_id, "dataset_id": dataset_id}


@router.post("/api/v1/sessions/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    _assert_session(session_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV/XLSX/XLS files are supported.")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            ingested = ingest_file(tmp_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {exc}") from exc
        sess = _SESSIONS[session_id]
        sess["df"] = ingested["df"]
        sess["dataset_id"] = ingested["dataset_id"]
        sess["file_metadata"] = ingested["file_metadata"]
        sess["state"] = "uploaded"
        return {
            "status": "uploaded",
            "session_id": session_id,
            "dataset_id": ingested["dataset_id"],
            "file_metadata": ingested["file_metadata"],
        }
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.get("/api/v1/sessions/{session_id}/preview")
def preview_dataset(session_id: str, rows: int = 10) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    df = sess["df"]
    return {
        "session_id": session_id,
        "columns": list(df.columns),
        "preview": df.head(rows).to_dict(orient="records"),
        "shape": {"rows": len(df), "columns": len(df.columns)},
    }


@router.get("/api/v1/sessions/{session_id}/profile")
def get_profile(session_id: str) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    if not sess.get("profile"):
        sess["profile"] = profile_dataset(sess["df"])
        sess["state"] = "profiled"
    return {"session_id": session_id, "profile": sess["profile"]}


@router.get("/api/v1/sessions/{session_id}/facts")
def get_facts(session_id: str) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    if not sess.get("profile"):
        sess["profile"] = profile_dataset(sess["df"])

    if not sess.get("facts"):
        sess["facts"] = build_facts_bundle(
            sess["df"], sess["dataset_id"], session_id, sess["profile"]
        )
        sess["state"] = "facts_built"

    return {"session_id": session_id, "facts_bundle": _json_safe(sess["facts"])}


@router.post("/api/v1/sessions/{session_id}/dashboard-spec")
def create_dashboard_spec(session_id: str, body: DashboardSpecRequest) -> dict[str, Any]:
    _ = body.template, body.include
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    chart_recs = recommend_charts(sess["facts"])
    sess["chart_recs"] = chart_recs
    sess["dashboard_spec"] = build_dashboard_spec(sess["facts"], chart_recs)
    sess["state"] = "dashboard_spec_built"
    return {
        "session_id": session_id,
        "dashboard_spec": sess["dashboard_spec"],
        "chart_recommendations": chart_recs,
    }


@router.get("/api/v1/sessions/{session_id}/dashboard")
def get_dashboard(session_id: str) -> dict[str, Any]:
    sess = _assert_session(session_id)
    if not sess.get("dashboard_spec"):
        raise HTTPException(status_code=400, detail="Run POST /api/v1/sessions/{id}/dashboard-spec first.")
    return {"session_id": session_id, "dashboard_spec": sess["dashboard_spec"]}


@router.post("/api/v1/sessions/{session_id}/insights")
def create_insights(session_id: str) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    memory = _get_memory(sess, session_id)
    insights = _run_llm(generate_insights, sess["facts"], memory)
    sess["insights"] = insights
    sess["state"] = "insights_generated"
    return {"session_id": session_id, "insights": insights}


@router.post("/api/v1/sessions/{session_id}/ask")
def ask_question(session_id: str, body: AskRequest) -> dict[str, Any]:
    _ = body.mode
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    memory = _get_memory(sess, session_id)
    result = _run_llm(ask_data, body.question, sess["facts"], memory)
    return {"session_id": session_id, "question": body.question, **result}


@router.post("/api/v1/sessions/{session_id}/report")
def create_report(session_id: str, body: ReportRequest) -> dict[str, Any]:
    _ = body.template, body.sections
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    memory = _get_memory(sess, session_id)
    if not sess.get("insights"):
        sess["insights"] = _run_llm(generate_insights, sess["facts"], memory)
    report_html = _run_llm(generate_report, sess["facts"], sess["insights"], memory)
    sess["report_html"] = report_html
    sess["state"] = "report_generated"
    return {"session_id": session_id, "status": "generated", "report_html": report_html}


@router.get("/api/v1/sessions/{session_id}/report", response_class=HTMLResponse)
def view_report(session_id: str) -> HTMLResponse:
    sess = _assert_session(session_id)
    if not sess.get("report_html"):
        raise HTTPException(status_code=400, detail="Run POST /api/v1/sessions/{id}/report first.")
    return HTMLResponse(content=sess["report_html"])


@router.get("/api/v1/sessions/{session_id}/memory")
def get_memory(session_id: str) -> dict[str, Any]:
    sess = _assert_session(session_id)
    memory = _get_memory(sess, session_id)
    return {"session_id": session_id, "memory": memory.to_dict()}


@router.post("/api/v1/sessions/{session_id}/memory/define")
def define_memory_term(session_id: str, body: MemoryDefineRequest) -> dict[str, Any]:
    sess = _assert_session(session_id)
    memory = _get_memory(sess, session_id)
    memory.define(body.term, body.definition)
    return {"session_id": session_id, "status": "saved", "memory": memory.to_dict()}


@router.post("/api/v1/sessions/{session_id}/memory/define-kpi")
def define_memory_kpi(session_id: str, body: MemoryDefineKPIRequest) -> dict[str, Any]:
    sess = _assert_session(session_id)
    memory = _get_memory(sess, session_id)
    memory.define_kpi(
        name=body.name,
        formula=body.formula,
        unit=body.unit,
        target=body.target,
        direction=body.direction,
    )
    return {"session_id": session_id, "status": "saved", "memory": memory.to_dict()}


@router.get("/api/v1/sessions/{session_id}/radar")
def proactive_radar(session_id: str) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    memory = _get_memory(sess, session_id)
    radar = _run_llm(run_proactive_radar, sess["facts"], memory)
    return {"session_id": session_id, **radar}


@router.get("/api/v1/sessions/{session_id}/cohorts")
def cohort_analysis(session_id: str) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    memory = _get_memory(sess, session_id)
    cohort = _run_llm(build_cohort_analysis, sess["df"], sess["facts"], memory)
    return {"session_id": session_id, "cohort_analysis": cohort}


@router.post("/api/v1/sessions/{session_id}/eval")
def eval_answer(session_id: str, body: EvalRequest) -> dict[str, Any]:
    _assert_session(session_id)
    result = _run_llm(evaluate_answer_accuracy, body.question, body.ai_answer, body.ground_truth)
    return {"session_id": session_id, "evaluation": result}


@router.get("/api/v1/sessions/{session_id}/queries")
def list_saved_queries(session_id: str) -> dict[str, Any]:
    sess = _assert_session(session_id)
    memory = _get_memory(sess, session_id)
    return {"session_id": session_id, "saved_queries": memory.list_queries()}


@router.post("/api/v1/sessions/{session_id}/queries/{query_id}/rerun")
def rerun_saved_query(session_id: str, query_id: str) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    memory = _get_memory(sess, session_id)
    query = memory.get_query(query_id)
    if not query:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")
    question = str(query.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Saved query has no question text.")
    result = _run_llm(ask_data, question, sess["facts"], memory)
    return {"session_id": session_id, "query_id": query_id, "question": question, "result": result}


@router.post("/api/v1/sessions/{session_id}/run-full-pipeline")
def run_full_pipeline(session_id: str, question: str | None = None) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    df = sess["df"]
    dataset_id = sess["dataset_id"]
    memory = _get_memory(sess, session_id)

    profile = profile_dataset(df)
    sess["profile"] = profile

    facts = build_facts_bundle(df, dataset_id, session_id, profile)
    sess["facts"] = facts

    chart_recs = recommend_charts(facts)
    sess["chart_recs"] = chart_recs

    dash_spec = build_dashboard_spec(facts, chart_recs)
    sess["dashboard_spec"] = dash_spec

    insights = _run_llm(generate_insights, facts, memory)
    sess["insights"] = insights

    report_html = _run_llm(generate_report, facts, insights, memory)
    sess["report_html"] = report_html

    ask_result = _run_llm(ask_data, question, facts, memory) if question else None
    sess["state"] = "complete"

    return {
        "session_id": session_id,
        "dataset_id": dataset_id,
        "state": "complete",
        "profile_summary": {
            "rows": profile["row_count"],
            "columns": profile["column_count"],
            "quality_score": profile["quality_score"],
        },
        "insights_summary": {
            "executive_summary": insights.get("executive_summary", ""),
            "confidence": insights.get("confidence", ""),
            "key_findings_count": len(insights.get("key_findings", [])),
        },
        "dashboard_spec": dash_spec,
        "chart_recommendations": chart_recs,
        "ask_result": ask_result,
        "report_available": True,
    }


@router.post("/api/v1/pipeline/run")
def run_pipeline_direct(file_path: str, question: str | None = None) -> dict[str, Any]:
    """
    Optional convenience endpoint for local integration tests.
    Executes the class orchestrator directly.
    """
    try:
        session = AIAnalyticsSession()
        return session.run(file_path, question=question)
    except ClaudeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ───────────────────────────────────────────────────────────────
# Phase 1 — Deep Data Understanding Endpoints
# ───────────────────────────────────────────────────────────────

@router.get("/api/v1/sessions/{session_id}/schema-intelligence")
def schema_intelligence(session_id: str) -> dict[str, Any]:
    """Deep column analysis: semantic types, units, hierarchies, temporal patterns."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    facts = sess["facts"]
    return {
        "session_id": session_id,
        "data_dictionary": facts.get("data_dictionary"),
        "star_schema": facts.get("star_schema"),
    }


@router.get("/api/v1/sessions/{session_id}/data-relationships")
def data_relationships(session_id: str) -> dict[str, Any]:
    """Entity-relationship graph: topology, hierarchies, functional dependencies."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    facts = sess["facts"]
    return {
        "session_id": session_id,
        "topology": facts.get("topology"),
        "star_schema": facts.get("star_schema"),
    }


@router.get("/api/v1/sessions/{session_id}/quality-diagnosis")
def quality_diagnosis(session_id: str) -> dict[str, Any]:
    """Root-cause quality analysis: missingness, data entry errors, imputation strategies."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    facts = sess["facts"]
    return {
        "session_id": session_id,
        "quality_report_card": facts.get("quality_report_card"),
        "quality": facts.get("quality", {}),
    }


# ───────────────────────────────────────────────────────────────
# Phase 2 — Statistical Reasoning Endpoints
# ───────────────────────────────────────────────────────────────

@router.get("/api/v1/sessions/{session_id}/statistical-tests")
def statistical_tests(session_id: str) -> dict[str, Any]:
    """Significance tests on key metrics: trend tests, changepoints, effect sizes."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    facts = sess["facts"]
    # Extract significance tests from trends
    stat_results = {}
    for key, trend in facts.get("trends", {}).items():
        if "significance_test" in trend:
            stat_results[key] = {
                "significance_test": trend["significance_test"],
                "changepoints": trend.get("changepoints", []),
            }
    return {
        "session_id": session_id,
        "statistical_results": stat_results,
        "executive_narrative": facts.get("executive_narrative", ""),
    }


class CompareRequest(BaseModel):
    metric: str
    segment_column: str


@router.post("/api/v1/sessions/{session_id}/compare")
def statistical_comparison(session_id: str, body: CompareRequest) -> dict[str, Any]:
    """Statistical comparison between segments for a specific metric."""
    sess = _assert_session_with_data(session_id)
    df = sess["df"]
    try:
        from backend.statistical_engine import StatisticalEngine
        engine = StatisticalEngine()
        if body.segment_column not in df.columns or body.metric not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names.")
        groups = {
            name: group[body.metric].dropna().values
            for name, group in df.groupby(body.segment_column)
        }
        result = engine.test_group_difference(list(groups.values()))
        return {"session_id": session_id, "metric": body.metric, "segment_column": body.segment_column, "result": result}
    except ImportError:
        raise HTTPException(status_code=501, detail="Statistical engine not available.")


# ───────────────────────────────────────────────────────────────
# Phase 3 — Predictive Intelligence Endpoints
# ───────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    metric: str
    horizon: int = 6


@router.post("/api/v1/sessions/{session_id}/forecast")
def forecast_metric(session_id: str, body: ForecastRequest) -> dict[str, Any]:
    """Time series forecasting for a specific metric."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    # Check if forecast already exists in trends
    trends = sess["facts"].get("trends", {})
    for key, trend in trends.items():
        if trend.get("metric") == body.metric and "forecast" in trend:
            return {"session_id": session_id, "metric": body.metric, "forecast": trend["forecast"]}
    # Generate fresh forecast
    try:
        from backend.forecasting_engine import ForecastingEngine
        engine = ForecastingEngine()
        df = sess["df"]
        series = df[body.metric].dropna().values.tolist() if body.metric in df.columns else []
        if len(series) < 4:
            raise HTTPException(status_code=400, detail=f"Not enough data points for '{body.metric}' (need >= 4).")
        result = engine.auto_forecast(series, horizon=body.horizon)
        return {"session_id": session_id, "metric": body.metric, "forecast": result}
    except ImportError:
        raise HTTPException(status_code=501, detail="Forecasting engine not available.")


@router.get("/api/v1/sessions/{session_id}/anomalies")
def anomaly_detection(session_id: str) -> dict[str, Any]:
    """Anomaly detection results: statistical outliers, clinical impossibilities, drift."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    return {
        "session_id": session_id,
        "anomaly_report": sess["facts"].get("anomaly_report", {}),
    }


class ScenarioRequest(BaseModel):
    metric: str
    change_pct: float = 10.0
    scenario_type: str = "optimistic"


@router.post("/api/v1/sessions/{session_id}/scenario")
def scenario_analysis(session_id: str, body: ScenarioRequest) -> dict[str, Any]:
    """What-if scenario analysis for a metric."""
    sess = _assert_session_with_data(session_id)
    try:
        from backend.forecasting_engine import ForecastingEngine
        engine = ForecastingEngine()
        df = sess["df"]
        series = df[body.metric].dropna().values.tolist() if body.metric in df.columns else []
        if len(series) < 4:
            raise HTTPException(status_code=400, detail=f"Not enough data for '{body.metric}'.")
        base = engine.auto_forecast(series, horizon=6)
        scenarios = engine.scenario_analysis(series, change_pct=body.change_pct)
        return {"session_id": session_id, "metric": body.metric, "baseline": base, "scenarios": scenarios}
    except ImportError:
        raise HTTPException(status_code=501, detail="Forecasting engine not available.")


# ───────────────────────────────────────────────────────────────
# Phase 4 — Feedback & Learning Endpoints
# ───────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    surface: str = "ask_data"
    target_id: str = ""
    rating: str = "positive"
    correction: str = ""
    question: str = ""
    ai_answer: str = ""


@router.post("/api/v1/sessions/{session_id}/feedback")
def capture_feedback(session_id: str, body: FeedbackRequest) -> dict[str, Any]:
    """Capture analyst corrections and ratings for feedback-driven improvement."""
    _assert_session(session_id)
    try:
        from backend.feedback_loop import FeedbackLoop
        fb = FeedbackLoop()
        if body.correction and body.question:
            fb.capture_correction(body.question, body.ai_answer, body.correction)
        fb.capture_rating(body.surface, body.target_id, body.rating)
        return {"session_id": session_id, "status": "captured", "surface": body.surface, "rating": body.rating}
    except ImportError:
        raise HTTPException(status_code=501, detail="Feedback loop not available.")


@router.get("/api/v1/sessions/{session_id}/suggestions")
def ai_suggestions(session_id: str) -> dict[str, Any]:
    """AI-suggested analyses based on dataset profile and knowledge base."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    try:
        from backend.knowledge_base import AnalyticsKnowledgeBase
        kb = AnalyticsKnowledgeBase()
        suggestions = kb.suggest_analysis(sess["facts"])
        return {"session_id": session_id, "suggestions": suggestions}
    except (ImportError, Exception):
        # Fallback: return basic suggestions from facts
        facts = sess["facts"]
        numeric_metrics = list(facts.get("metrics", {}).get("kpis", {}).keys())[:5]
        return {
            "session_id": session_id,
            "suggestions": [f"Analyze trend in {m}" for m in numeric_metrics],
        }


# ───────────────────────────────────────────────────────────────
# Phase 6 — Autonomous Analytics Endpoints
# ───────────────────────────────────────────────────────────────

class InvestigateRequest(BaseModel):
    question: str


@router.post("/api/v1/sessions/{session_id}/investigate")
def launch_investigation(session_id: str, body: InvestigateRequest) -> dict[str, Any]:
    """Launch an autonomous multi-step investigation."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    try:
        from backend.investigation_agent import InvestigationAgent
        agent = InvestigationAgent()
        report = agent.investigate(body.question, sess["df"], facts=sess["facts"])
        return {
            "session_id": session_id,
            "investigation_id": report.investigation_id,
            "question": report.question,
            "target_metric": report.target_metric,
            "executive_summary": report.executive_summary,
            "root_cause_diagnosis": report.root_cause_diagnosis,
            "evidence_chain": report.evidence_chain,
            "recommended_playbook": report.recommended_playbook,
            "steps_executed": report.steps_executed,
        }
    except ImportError:
        raise HTTPException(status_code=501, detail="Investigation agent not available.")


@router.get("/api/v1/sessions/{session_id}/proactive-scan")
def proactive_scan(session_id: str) -> dict[str, Any]:
    """Proactive intelligence scan: emerging issues, target breaches, early warnings."""
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    try:
        from backend.proactive_intelligence import ProactiveIntelligence
        pi = ProactiveIntelligence()
        scan = pi.scan_dataset(sess["df"], facts=sess["facts"])
        briefing = pi.generate_executive_briefing([scan])
        return {
            "session_id": session_id,
            "scan_results": scan,
            "executive_briefing": briefing,
        }
    except ImportError:
        raise HTTPException(status_code=501, detail="Proactive intelligence not available.")


def _assert_session(session_id: str) -> dict[str, Any]:
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _SESSIONS[session_id]


def _assert_session_with_data(session_id: str) -> dict[str, Any]:
    sess = _assert_session(session_id)
    if sess.get("df") is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet. POST to /upload first.")
    return sess


def _get_memory(sess: dict[str, Any], session_id: str) -> SmartMemory:
    memory = sess.get("memory")
    if isinstance(memory, SmartMemory):
        return memory
    memory = SmartMemory(session_id=session_id, storage_path=_MEMORY_PATH)
    sess["memory"] = memory
    return memory


def _ensure_facts(sess: dict[str, Any], session_id: str) -> None:
    if not sess.get("profile"):
        sess["profile"] = profile_dataset(sess["df"])
    if not sess.get("facts"):
        sess["facts"] = build_facts_bundle(
            sess["df"], sess["dataset_id"], session_id, sess["profile"]
        )


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _run_llm(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ClaudeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def build_standalone_app() -> FastAPI:
    api_app = FastAPI(
        title="Health-Data-Analytics-AI — AI Service",
        description="AI analytics engine: facts, charts, dashboards, ask-data, reports.",
        version="1.0.0",
    )
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_app.include_router(router)
    return api_app


app = build_standalone_app()
