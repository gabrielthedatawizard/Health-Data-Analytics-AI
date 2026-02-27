"""
FastAPI REST API for the AI analytics service from files (1).zip.
"""
from __future__ import annotations

import json
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
    ask_data,
    build_dashboard_spec,
    build_facts_bundle,
    generate_insights,
    generate_report,
    ingest_file,
    profile_dataset,
    recommend_charts,
)

router = APIRouter(tags=["ai-analytics-v1"])

# In-memory session store (replace with Redis/Postgres in production)
_SESSIONS: dict[str, dict[str, Any]] = {}


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


@router.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "ai-analytics", "version": "1.0.0"}


@router.post("/api/v1/sessions")
def create_session(body: SessionCreate) -> dict[str, str]:
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
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

        ingested = ingest_file(tmp_path)
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
    insights = _run_llm(generate_insights, sess["facts"])
    sess["insights"] = insights
    sess["state"] = "insights_generated"
    return {"session_id": session_id, "insights": insights}


@router.post("/api/v1/sessions/{session_id}/ask")
def ask_question(session_id: str, body: AskRequest) -> dict[str, Any]:
    _ = body.mode
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    result = _run_llm(ask_data, body.question, sess["facts"])
    return {"session_id": session_id, "question": body.question, **result}


@router.post("/api/v1/sessions/{session_id}/report")
def create_report(session_id: str, body: ReportRequest) -> dict[str, Any]:
    _ = body.template, body.sections
    sess = _assert_session_with_data(session_id)
    _ensure_facts(sess, session_id)
    if not sess.get("insights"):
        sess["insights"] = _run_llm(generate_insights, sess["facts"])
    report_html = _run_llm(generate_report, sess["facts"], sess["insights"])
    sess["report_html"] = report_html
    sess["state"] = "report_generated"
    return {"session_id": session_id, "status": "generated", "report_html": report_html}


@router.get("/api/v1/sessions/{session_id}/report", response_class=HTMLResponse)
def view_report(session_id: str) -> HTMLResponse:
    sess = _assert_session(session_id)
    if not sess.get("report_html"):
        raise HTTPException(status_code=400, detail="Run POST /api/v1/sessions/{id}/report first.")
    return HTMLResponse(content=sess["report_html"])


@router.post("/api/v1/sessions/{session_id}/run-full-pipeline")
def run_full_pipeline(session_id: str, question: str | None = None) -> dict[str, Any]:
    sess = _assert_session_with_data(session_id)
    df = sess["df"]
    dataset_id = sess["dataset_id"]

    profile = profile_dataset(df)
    sess["profile"] = profile

    facts = build_facts_bundle(df, dataset_id, session_id, profile)
    sess["facts"] = facts

    chart_recs = recommend_charts(facts)
    sess["chart_recs"] = chart_recs

    dash_spec = build_dashboard_spec(facts, chart_recs)
    sess["dashboard_spec"] = dash_spec

    insights = _run_llm(generate_insights, facts)
    sess["insights"] = insights

    report_html = _run_llm(generate_report, facts, insights)
    sess["report_html"] = report_html

    ask_result = _run_llm(ask_data, question, facts) if question else None
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


def _assert_session(session_id: str) -> dict[str, Any]:
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _SESSIONS[session_id]


def _assert_session_with_data(session_id: str) -> dict[str, Any]:
    sess = _assert_session(session_id)
    if sess.get("df") is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet. POST to /upload first.")
    return sess


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

