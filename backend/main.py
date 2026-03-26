from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Response as FastAPIResponse, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

try:
    from pypdf import PdfReader
except Exception:  # noqa: PERF203
    PdfReader = None

from backend.api import router as ai_analytics_router
from backend.cache import CacheManager
from backend.jobs import create_job, get_job, list_jobs, update_job
from backend.llm_client import LLMClient, create_llm_client_from_env
from backend.llm_gate import FactsGroundingError, SchemaValidationError, validate_facts_references, validate_schema
from backend.llm_schemas import ASK_NARRATIVE_SCHEMA, DASHBOARD_SPEC_SCHEMA, FACTS_BUNDLE_SCHEMA, QUERY_PLAN_SCHEMA
from backend.semantic_layer import (
    QueryPlanValidationError,
    build_semantic_layer,
    semantic_prompt_context,
    validate_and_resolve_query_plan,
)

DATA_DIR = Path("data_store")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
PII_KEYWORDS = (
    "name",
    "phone",
    "email",
    "address",
    "mrn",
    "patient",
    "id",
    "national",
    "nin",
    "passport",
)
ID_KEYWORDS = ("id", "mrn", "patient_id", "record_id", "encounter_id")
HMIS_TEMPLATE_FIELDS = ("facility", "district", "month", "diagnosis", "age_group", "sex")
TIME_KEYWORDS = ("date", "visit_date", "month", "week", "year", "period")
GEO_KEYWORDS = ("region", "district", "facility", "county", "subcounty", "province")
SERVICE_KEYWORDS = ("opd", "anc", "immun", "ncd", "tb", "hiv", "malaria")

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"^\+?[\d\-\s\(\)]{7,}$")
ID_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9\-_]{6,}$")
DATASET_ID_PATTERN = re.compile(r"^[a-f0-9\-]{36}$")

SMALL_ROW_MAX = 50_000
MID_ROW_MAX = 1_000_000
MID_COL_MAX = 200
SAMPLE_MAX_ROWS = 250_000
CACHE_VERSION = "v1"
DOCUMENT_LIBRARY_DIR = DATA_DIR / "document_library"
DOCUMENT_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".json", ".pdf"}
MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
DOCUMENT_CHUNK_SIZE = 900
DOCUMENT_CHUNK_OVERLAP = 120
DOCUMENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
VALID_USER_ROLES = {"viewer", "analyst", "reviewer", "admin"}
ROLE_PERMISSIONS: dict[str, list[str]] = {
    "viewer": ["sessions:read_own", "sessions:export_masked_own", "docs:read_own"],
    "analyst": [
        "sessions:create",
        "sessions:read_own",
        "sessions:write_own",
        "sessions:compute_own",
        "sessions:export_own",
        "sensitive_export:request_own",
        "workflow:create_own",
        "workflow:execute_own",
        "docs:create",
        "docs:read_own",
    ],
    "reviewer": [
        "sessions:create",
        "sessions:read_own",
        "sessions:write_own",
        "sessions:compute_own",
        "sessions:export_own",
        "sessions:read_all",
        "sessions:export_all",
        "sensitive_export:request_own",
        "sensitive_export:review",
        "ml:promote",
        "workflow:create_own",
        "workflow:review",
        "workflow:execute_all",
        "docs:create",
        "docs:read_own",
        "docs:read_all",
    ],
    "admin": [
        "sessions:create",
        "sessions:read_all",
        "sessions:write_all",
        "sessions:compute_all",
        "sessions:export_all",
        "sensitive_export:review",
        "ml:promote",
        "workflow:create_all",
        "workflow:review",
        "workflow:execute_all",
        "docs:create",
        "docs:read_all",
        "admin:all",
    ],
}
WORKFLOW_ACTION_TYPES = {"draft_email", "create_ticket", "action_plan", "schedule_report"}
WORKFLOW_STATUS_PENDING = "pending_approval"
WORKFLOW_STATUS_APPROVED = "approved"
WORKFLOW_STATUS_REJECTED = "rejected"
WORKFLOW_STATUS_EXECUTED = "executed"
FEEDBACK_SURFACES = {"ask_data", "document_qa", "dashboard_summary", "chart_explanation"}
FEEDBACK_RATINGS = {"positive", "negative"}
COHORT_OPERATOR_LABELS = {
    "eq": "equals",
    "neq": "does not equal",
    "contains": "contains",
    "starts_with": "starts with",
    "gt": "is greater than",
    "gte": "is at least",
    "lt": "is less than",
    "lte": "is at most",
    "between": "is between",
    "in": "is one of",
    "not_in": "is not one of",
    "is_null": "is blank",
    "not_null": "is present",
}
COHORT_OPERATORS_BY_TYPE = {
    "string": {"eq", "neq", "contains", "starts_with", "in", "not_in", "is_null", "not_null"},
    "number": {"eq", "neq", "gt", "gte", "lt", "lte", "between", "in", "not_in", "is_null", "not_null"},
    "datetime": {"eq", "neq", "gt", "gte", "lt", "lte", "between", "in", "not_in", "is_null", "not_null"},
}


app = FastAPI(title="AI Analytics Backend", version="0.1.0")
cache = CacheManager()
_llm_client: LLMClient | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ai_analytics_router)


class CreateSessionRequest(BaseModel):
    created_by: str = "anonymous"
    display_name: str | None = None
    description: str | None = None


class CreateSessionResponse(BaseModel):
    dataset_id: str
    created_at: str


class SessionSummaryResponse(BaseModel):
    dataset_id: str
    display_name: str
    description: str
    status: str
    created_at: str
    updated_at: str
    created_by: str
    pii_masking_enabled: bool
    allow_sensitive_export: bool
    file_name: str | None = None
    file_type: str | None = None
    size_bytes: int = 0
    row_count: int = 0
    column_count: int = 0
    quality_score: float = 0.0
    quality_issues: list[str] = Field(default_factory=list)
    sensitive_export_approval: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class UpdateSessionRequest(BaseModel):
    display_name: str | None = None
    description: str | None = None


class SessionMetaResponse(BaseModel):
    dataset_id: str
    status: str
    created_at: str
    updated_at: str
    created_by: str
    user_id: str | None = None
    pii_masking_enabled: bool
    allow_sensitive_export: bool = False
    sensitive_export_approval: dict[str, Any] = Field(default_factory=dict)
    file: dict[str, Any] | None = None
    file_hash: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class ProfileResponse(BaseModel):
    dataset_id: str
    profile: dict[str, Any]


class FactsResponse(BaseModel):
    dataset_id: str
    facts_bundle: dict[str, Any]


class DashboardSpecResponse(BaseModel):
    dataset_id: str
    dashboard_spec: dict[str, Any]


class AnomalyFinding(BaseModel):
    anomaly_id: str
    kind: str
    severity: str
    title: str
    summary: str
    metric: str | None = None
    dimension: str | None = None
    segment: str | None = None
    period: str | None = None
    evidence: list[str] = Field(default_factory=list)
    root_cause_hints: list[str] = Field(default_factory=list)
    recommended_question: str | None = None


class AnomalyAnalysisPayload(BaseModel):
    generated_at: str
    anomaly_count: int
    summary: str
    anomalies: list[AnomalyFinding] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)


class AnomalyAnalysisResponse(BaseModel):
    dataset_id: str
    analysis: AnomalyAnalysisPayload


class CohortCriterion(BaseModel):
    field: str = Field(min_length=1, max_length=200)
    operator: str = Field(min_length=2, max_length=32)
    value: Any | None = None


class CohortBuildRequest(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    criteria: list[CohortCriterion] = Field(default_factory=list)
    limit: int = Field(default=25, ge=1, le=100)


class CohortAnalysisPayload(BaseModel):
    generated_at: str
    name: str
    description: str | None = None
    row_count: int
    population_row_count: int
    criteria_count: int
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)
    summary: str
    suggested_questions: list[str] = Field(default_factory=list)


class CohortAnalysisResponse(BaseModel):
    dataset_id: str
    cohort: CohortAnalysisPayload


class WorkflowDraftRequest(BaseModel):
    action_type: str = Field(min_length=3, max_length=64)
    title: str | None = Field(default=None, max_length=180)
    target: str | None = Field(default=None, max_length=180)
    objective: str | None = Field(default=None, max_length=1000)


class WorkflowDecisionRequest(BaseModel):
    approved: bool
    note: str | None = Field(default=None, max_length=1000)


class WorkflowExecutionRequest(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


class WorkflowActionRecord(BaseModel):
    action_id: str
    action_type: str
    status: str
    title: str
    target: str | None = None
    objective: str | None = None
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    generated_at: str
    updated_at: str
    created_by: str
    review_note: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    executed_by: str | None = None
    executed_at: str | None = None
    execution_note: str | None = None
    requires_approval: bool = True


class WorkflowActionsResponse(BaseModel):
    dataset_id: str
    actions: list[WorkflowActionRecord] = Field(default_factory=list)


class InvestigationSaveRequest(BaseModel):
    title: str | None = Field(default=None, max_length=180)
    question: str = Field(min_length=3, max_length=2000)
    context_type: str = Field(default="ask", max_length=64)
    note: str | None = Field(default=None, max_length=1000)
    result: dict[str, Any] = Field(default_factory=dict)


class SavedInvestigationRecord(BaseModel):
    investigation_id: str
    title: str
    question: str
    context_type: str
    note: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    created_by: str


class SavedInvestigationsResponse(BaseModel):
    dataset_id: str
    investigations: list[SavedInvestigationRecord] = Field(default_factory=list)


class PlaybookSaveRequest(BaseModel):
    name: str = Field(min_length=3, max_length=180)
    question_template: str = Field(min_length=3, max_length=2000)
    description: str | None = Field(default=None, max_length=1000)
    context_type: str = Field(default="ask", max_length=64)


class SavedPlaybookRecord(BaseModel):
    playbook_id: str
    name: str
    question_template: str
    description: str | None = None
    context_type: str
    created_at: str
    created_by: str


class SavedPlaybooksResponse(BaseModel):
    dataset_id: str
    playbooks: list[SavedPlaybookRecord] = Field(default_factory=list)


class ReportScheduleRecord(BaseModel):
    schedule_id: str
    title: str
    frequency: str
    report_template: str
    sections: list[str] = Field(default_factory=list)
    audience: str | None = None
    objective: str | None = None
    delivery_note: str | None = None
    status: str = "active"
    source_action_id: str | None = None
    created_at: str
    updated_at: str
    created_by: str
    last_run_at: str | None = None
    last_job_id: str | None = None
    last_run_status: str | None = None


class ReportSchedulesResponse(BaseModel):
    dataset_id: str
    schedules: list[ReportScheduleRecord] = Field(default_factory=list)


class ReportScheduleRunResponse(BaseModel):
    dataset_id: str
    schedule_id: str
    job_id: str
    status: str


class ForecastTrainRequest(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    time_field: str | None = Field(default=None, max_length=200)
    metric_field: str | None = Field(default=None, max_length=200)
    horizon: int = Field(default=3, ge=1, le=12)
    aggregation: str = Field(default="sum", max_length=16)


class ForecastSeriesPoint(BaseModel):
    period: str
    value: float


class ForecastCandidateMetrics(BaseModel):
    model_name: str
    mae: float
    rmse: float
    mape: float | None = None
    holdout_points: int


class ForecastRunPayload(BaseModel):
    generated_at: str
    name: str
    time_field: str
    metric_field: str
    aggregation: str
    periods_used: int
    holdout_points: int
    horizon: int
    champion_model: str
    training_data_hash: str | None = None
    baseline_mean: float | None = None
    baseline_std: float | None = None
    latest_actual: float | None = None
    summary: str
    warnings: list[str] = Field(default_factory=list)
    candidate_models: list[ForecastCandidateMetrics] = Field(default_factory=list)
    historical: list[ForecastSeriesPoint] = Field(default_factory=list)
    forecast: list[ForecastSeriesPoint] = Field(default_factory=list)


class ForecastRunRecord(BaseModel):
    run_id: str
    model_kind: str
    status: str
    created_at: str
    created_by: str
    payload: ForecastRunPayload


class ForecastRunsResponse(BaseModel):
    dataset_id: str
    runs: list[ForecastRunRecord] = Field(default_factory=list)


class ForecastDriftSignal(BaseModel):
    code: str
    severity: str
    message: str


class ForecastDriftPayload(BaseModel):
    generated_at: str
    run_id: str
    run_name: str
    champion_model: str
    time_field: str
    metric_field: str
    aggregation: str
    training_data_hash: str | None = None
    current_data_hash: str | None = None
    stale_model: bool = False
    drift_score: float
    periods_analyzed: int
    baseline_mean: float
    recent_mean: float
    baseline_std: float
    recent_std: float
    summary: str
    signals: list[ForecastDriftSignal] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class ForecastDriftResponse(BaseModel):
    dataset_id: str
    drift: ForecastDriftPayload


class ModelPromotionRequest(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


class ModelRegistryEntry(BaseModel):
    registry_id: str
    run_id: str
    model_kind: str
    status: str
    promoted_at: str
    promoted_by: str
    note: str | None = None
    name: str
    champion_model: str
    metric_field: str
    time_field: str
    aggregation: str
    source_data_hash: str | None = None


class ModelRegistryResponse(BaseModel):
    dataset_id: str
    entries: list[ModelRegistryEntry] = Field(default_factory=list)


class ModelEvaluationRunSummary(BaseModel):
    run_id: str
    name: str
    champion_model: str
    metric_field: str
    mae: float
    rmse: float
    mape: float | None = None
    drift_score: float
    stale_model: bool
    source: str


class ModelEvaluationPayload(BaseModel):
    generated_at: str
    active_run: ModelEvaluationRunSummary
    challenger_run: ModelEvaluationRunSummary
    recommendation: str
    winner: str
    rationale: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)


class ModelEvaluationResponse(BaseModel):
    dataset_id: str
    evaluation: ModelEvaluationPayload


class ReportResponse(BaseModel):
    dataset_id: str
    report_html_path: str
    report_html: str


class MaskingRequest(BaseModel):
    enabled: bool


class SensitiveExportRequest(BaseModel):
    enabled: bool


class SensitiveExportApprovalRequest(BaseModel):
    justification: str = Field(min_length=8, max_length=1000)


class SensitiveExportApprovalDecision(BaseModel):
    approved: bool
    note: str | None = Field(default=None, max_length=1000)


class CleanRequest(BaseModel):
    actions: list[str] = Field(default_factory=list)
    pii_mask: bool = False


class AskRequest(BaseModel):
    question: str
    mode: str = "safe"


class AskResponse(BaseModel):
    dataset_id: str
    answer: str
    facts_used: list[str]
    confidence: str
    fact_coverage: float
    data_coverage: str
    query_plan: dict[str, Any] | None = None
    result_rows: list[dict[str, Any]] = Field(default_factory=list)
    chart: dict[str, Any] | None = None
    governance: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    surface: str = Field(min_length=3, max_length=64)
    target_id: str = Field(min_length=1, max_length=200)
    rating: str = Field(min_length=3, max_length=16)
    question: str | None = Field(default=None, max_length=1000)
    title: str | None = Field(default=None, max_length=240)
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackRecord(BaseModel):
    feedback_id: str
    surface: str
    target_id: str
    rating: str
    question: str | None = None
    title: str | None = None
    comment: str | None = None
    created_at: str
    created_by: str


class FeedbackListResponse(BaseModel):
    dataset_id: str
    feedback: list[FeedbackRecord] = Field(default_factory=list)


class SemanticLayerResponse(BaseModel):
    dataset_id: str
    semantic_layer: dict[str, Any]


class ReportRequest(BaseModel):
    template: str | None = "health_report"
    sections: list[str] = Field(default_factory=lambda: ["quality", "kpis", "trends", "limitations"])


class DashboardSpecRequest(BaseModel):
    template: str = "health_core"


class AuthContextResponse(BaseModel):
    actor: str
    role: str
    permissions: list[str] = Field(default_factory=list)


class SystemStatusCounts(BaseModel):
    visible_sessions: int = 0
    visible_documents: int = 0
    active_jobs: int = 0
    queued_jobs: int = 0
    failed_jobs: int = 0
    active_models: int = 0
    stale_models: int = 0
    pending_sensitive_exports: int = 0
    pending_workflow_reviews: int = 0
    superseded_documents: int = 0
    recent_audit_events_24h: int = 0


class SystemStatusAlert(BaseModel):
    level: str
    message: str


class SystemStatusResponse(BaseModel):
    status: str
    timestamp: str
    actor: str
    role: str
    counts: SystemStatusCounts
    alerts: list[SystemStatusAlert] = Field(default_factory=list)


class ReviewQueueItem(BaseModel):
    item_id: str
    dataset_id: str
    dataset_label: str
    category: str
    severity: str
    status: str
    title: str
    summary: str
    created_at: str
    updated_at: str
    action_hint: str | None = None


class ReviewQueueResponse(BaseModel):
    status: str
    timestamp: str
    actor: str
    role: str
    total_items: int = 0
    items: list[ReviewQueueItem] = Field(default_factory=list)


class DocumentSummaryResponse(BaseModel):
    document_id: str
    title: str
    source_name: str
    status: str
    created_at: str
    updated_at: str
    created_by: str
    file_name: str
    file_type: str
    chunk_count: int
    char_count: int
    version_label: str | None = None
    effective_date: str | None = None
    supersedes_document_id: str | None = None
    superseded_by_document_id: str | None = None
    is_current: bool = True
    freshness: str = "current"
    freshness_note: str = ""


class DocumentMetaResponse(DocumentSummaryResponse):
    snippet_preview: str = ""


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    document_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=10)


class DocumentSearchHit(BaseModel):
    citation_key: str
    document_id: str
    title: str
    source_name: str
    snippet: str
    chunk_index: int
    score: float
    version_label: str | None = None
    effective_date: str | None = None
    freshness: str = "current"


class DocumentSearchResponse(BaseModel):
    query: str
    results: list[DocumentSearchHit] = Field(default_factory=list)


class DocumentAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=4, ge=1, le=8)


class DocumentCitation(BaseModel):
    citation_key: str
    document_id: str
    title: str
    source_name: str
    snippet: str
    chunk_index: int
    version_label: str | None = None
    effective_date: str | None = None
    freshness: str = "current"


class DocumentAskResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: str
    citations: list[DocumentCitation] = Field(default_factory=list)
    freshness_summary: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_dataset_id(dataset_id: str) -> None:
    if not DATASET_ID_PATTERN.match(dataset_id):
        raise HTTPException(status_code=400, detail="Invalid dataset_id format.")


def _dataset_path(dataset_id: str) -> Path:
    _validate_dataset_id(dataset_id)
    return DATA_DIR / dataset_id


def _meta_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "meta.json"


def _document_path(document_id: str) -> Path:
    _validate_dataset_id(document_id)
    return DOCUMENT_LIBRARY_DIR / document_id


def _document_meta_path(document_id: str) -> Path:
    return _document_path(document_id) / "meta.json"


def _document_content_path(document_id: str) -> Path:
    return _document_path(document_id) / "content.txt"


def _document_chunks_path(document_id: str) -> Path:
    return _document_path(document_id) / "chunks.json"


def _profile_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "profile.json"


def _facts_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "facts.json"


def _anomaly_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "anomaly_analysis.json"


def _cohort_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "cohort_analysis.json"


def _feedback_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "feedback.json"


def _report_schedules_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "report_schedules.json"


def _workflow_actions_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "workflow_actions.json"


def _saved_investigations_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "saved_investigations.json"


def _saved_playbooks_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "saved_playbooks.json"


def _ml_runs_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "ml_runs.json"


def _ml_drift_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "ml_drift.json"


def _ml_registry_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "ml_registry.json"


def _ml_evaluation_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "ml_evaluation.json"


def _dashboard_spec_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "dashboard_spec.json"


def _report_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "report.html"


def _cleaned_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "cleaned.csv"


def _report_pdf_path(dataset_id: str) -> Path:
    return _report_path(dataset_id).with_suffix(".pdf")


def _audit_log_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "audit_log.jsonl"


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_meta(dataset_id: str) -> dict[str, Any]:
    path = _meta_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    return _load_json(path)


def _save_meta(dataset_id: str, meta: dict[str, Any]) -> None:
    meta["updated_at"] = _utc_now_iso()
    _save_json(_meta_path(dataset_id), meta)


def _iter_accessible_sessions(actor: str, permissions: list[str]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for child in sorted(DATA_DIR.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = _load_json(meta_path)
            created_by = str(meta.get("user_id") or meta.get("created_by") or "anonymous")
            if "sessions:read_all" not in permissions and "admin:all" not in permissions and created_by != actor:
                continue
            dataset_id = str(meta.get("dataset_id") or child.name)
            items.append((dataset_id, meta))
        except Exception:
            continue
    return items


def _load_document_meta(document_id: str) -> dict[str, Any]:
    path = _document_meta_path(document_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    return _load_json(path)


def _save_document_meta(document_id: str, meta: dict[str, Any]) -> None:
    meta["updated_at"] = _utc_now_iso()
    _save_json(_document_meta_path(document_id), meta)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    try:
        return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None


def _count_recent_audit_events(dataset_id: str, since: datetime) -> int:
    path = _audit_log_path(dataset_id)
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            happened_at = _parse_iso_datetime(payload.get("timestamp"))
            if happened_at and happened_at >= since:
                count += 1
    return count


def _normalize_feedback_request(payload: FeedbackRequest) -> FeedbackRequest:
    payload.surface = payload.surface.strip().lower()
    payload.rating = payload.rating.strip().lower()
    payload.target_id = payload.target_id.strip()
    payload.question = payload.question.strip() if payload.question else None
    payload.title = payload.title.strip() if payload.title else None
    payload.comment = payload.comment.strip() if payload.comment else None
    if payload.surface not in FEEDBACK_SURFACES:
        raise HTTPException(status_code=400, detail=f"Unsupported feedback surface '{payload.surface}'.")
    if payload.rating not in FEEDBACK_RATINGS:
        raise HTTPException(status_code=400, detail=f"Unsupported feedback rating '{payload.rating}'.")
    if not payload.target_id:
        raise HTTPException(status_code=400, detail="target_id is required.")
    return payload


def _playbook_record(dataset_id: str, playbook_id: str) -> dict[str, Any]:
    for item in _load_saved_playbooks(dataset_id):
        if str(item.get("playbook_id") or "") == playbook_id:
            return item
    raise HTTPException(status_code=404, detail="Playbook not found.")


def _report_schedule_record(dataset_id: str, schedule_id: str) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    schedules = _load_report_schedules(dataset_id)
    for index, item in enumerate(schedules):
        if str(item.get("schedule_id") or "") == schedule_id:
            return index, item, schedules
    raise HTTPException(status_code=404, detail="Report schedule not found.")


def _load_workflow_actions(dataset_id: str) -> list[dict[str, Any]]:
    path = _workflow_actions_path(dataset_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    actions = payload.get("actions") if isinstance(payload, dict) else []
    return [item for item in actions if isinstance(item, dict)]


def _load_feedback(dataset_id: str) -> list[dict[str, Any]]:
    path = _feedback_path(dataset_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    items = payload.get("feedback") if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _load_report_schedules(dataset_id: str) -> list[dict[str, Any]]:
    path = _report_schedules_path(dataset_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    items = payload.get("schedules") if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _save_report_schedules(dataset_id: str, schedules: list[dict[str, Any]]) -> None:
    _save_json(
        _report_schedules_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "generated_at": _utc_now_iso(),
            "schedules": schedules,
        },
    )


def _save_feedback(dataset_id: str, items: list[dict[str, Any]]) -> None:
    _save_json(
        _feedback_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "generated_at": _utc_now_iso(),
            "feedback": items,
        },
    )


def _save_workflow_actions(dataset_id: str, actions: list[dict[str, Any]]) -> None:
    _save_json(
        _workflow_actions_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "generated_at": _utc_now_iso(),
            "actions": actions,
        },
    )


def _load_saved_investigations(dataset_id: str) -> list[dict[str, Any]]:
    path = _saved_investigations_path(dataset_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    items = payload.get("investigations") if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _save_saved_investigations(dataset_id: str, investigations: list[dict[str, Any]]) -> None:
    _save_json(
        _saved_investigations_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "generated_at": _utc_now_iso(),
            "investigations": investigations,
        },
    )


def _load_saved_playbooks(dataset_id: str) -> list[dict[str, Any]]:
    path = _saved_playbooks_path(dataset_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    items = payload.get("playbooks") if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _save_saved_playbooks(dataset_id: str, playbooks: list[dict[str, Any]]) -> None:
    _save_json(
        _saved_playbooks_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "generated_at": _utc_now_iso(),
            "playbooks": playbooks,
        },
    )


def _load_ml_runs(dataset_id: str) -> list[dict[str, Any]]:
    path = _ml_runs_path(dataset_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    items = payload.get("runs") if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _save_ml_runs(dataset_id: str, runs: list[dict[str, Any]]) -> None:
    _save_json(
        _ml_runs_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "generated_at": _utc_now_iso(),
            "runs": runs,
        },
    )


def _load_ml_drift(dataset_id: str) -> dict[str, Any] | None:
    path = _ml_drift_path(dataset_id)
    if not path.exists():
        return None
    return _load_json(path)


def _load_ml_registry(dataset_id: str) -> list[dict[str, Any]]:
    path = _ml_registry_path(dataset_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    items = payload.get("entries") if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _save_ml_registry(dataset_id: str, entries: list[dict[str, Any]]) -> None:
    _save_json(
        _ml_registry_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "generated_at": _utc_now_iso(),
            "entries": entries,
        },
    )


def _find_ml_run(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    return next((item for item in runs if str(item.get("run_id")) == run_id), None)


def _safe_directory_size(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _session_summary_from_meta(dataset_id: str, meta: dict[str, Any]) -> SessionSummaryResponse:
    file_meta = meta.get("file") or {}
    file_name = str(file_meta.get("filename") or "") or None
    file_type = str(file_meta.get("extension") or "").lstrip(".") or None
    display_name = str(meta.get("display_name") or file_name or dataset_id)
    description = str(meta.get("description") or "")

    row_count = 0
    column_count = 0
    quality_score = 0.0
    quality_issues: list[str] = []

    profile_path = _profile_path(dataset_id)
    if profile_path.exists():
        try:
            profile = _load_json(profile_path)
            shape = profile.get("shape") or {}
            row_count = int(shape.get("rows") or 0)
            column_count = int(shape.get("cols") or 0)
            quality_score = float(profile.get("quality_score") or 0.0)
            quality_issues = [
                str(issue.get("issue") or issue.get("code") or issue)
                for issue in (profile.get("quality_issues") or [])
                if issue is not None
            ]
            if not quality_issues:
                columns = profile.get("columns") or []
                for column in columns:
                    if not isinstance(column, dict):
                        continue
                    if float(column.get("missing_percent") or 0.0) >= 20:
                        quality_issues.append(f"{column.get('name', 'column')} has high missingness")
                quality_issues = quality_issues[:4]
        except Exception:
            row_count = 0
            column_count = 0
            quality_score = 0.0
            quality_issues = []

    size_bytes = int(file_meta.get("size_bytes") or 0)
    if size_bytes <= 0:
        size_bytes = _safe_directory_size(_dataset_path(dataset_id))

    return SessionSummaryResponse(
        dataset_id=dataset_id,
        display_name=display_name,
        description=description,
        status=str(meta.get("status", "unknown")),
        created_at=str(meta.get("created_at", "")),
        updated_at=str(meta.get("updated_at", "")),
        created_by=str(meta.get("created_by", "anonymous")),
        pii_masking_enabled=bool(meta.get("pii_masking_enabled", False)),
        allow_sensitive_export=bool(meta.get("allow_sensitive_export", False)),
        file_name=file_name,
        file_type=file_type,
        size_bytes=size_bytes,
        row_count=row_count,
        column_count=column_count,
        quality_score=round(quality_score, 2),
        quality_issues=quality_issues,
        sensitive_export_approval=_sensitive_export_approval(meta),
        artifacts=meta.get("artifacts", {}),
    )


def _append_audit(dataset_id: str, action: str, actor: str, details: dict[str, Any] | None = None) -> None:
    record = {
        "timestamp": _utc_now_iso(),
        "dataset_id": dataset_id,
        "action": action,
        "actor": actor or "anonymous",
        "details": details or {},
    }
    path = _audit_log_path(dataset_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _actor_from_header(x_api_key: str | None, fallback: str = "anonymous") -> str:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return fallback


def _normalize_role(role: str | None) -> str | None:
    if not role:
        return None
    normalized = role.strip().lower()
    if normalized in VALID_USER_ROLES:
        return normalized
    return None


def _role_from_identity(actor: str, explicit_role: str | None = None) -> str:
    normalized = _normalize_role(explicit_role)
    if normalized:
        return normalized

    identity = (actor or "").strip().lower()
    if not identity or identity == "anonymous":
        return "viewer"
    if identity == "admin" or identity.startswith("admin") or identity.endswith("_admin"):
        return "admin"
    if any(token in identity for token in ("manager", "reviewer", "approver", "auditor")):
        return "reviewer"
    if any(token in identity for token in ("viewer", "read_only", "readonly")):
        return "viewer"
    return "analyst"


def _permissions_for_role(role: str) -> list[str]:
    return list(ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["analyst"]))


def _session_owner(meta: dict[str, Any]) -> str:
    return str(meta.get("user_id") or meta.get("created_by") or "anonymous")


def _resolve_auth_context(
    x_api_key: str | None,
    x_user_role: str | None = None,
    *,
    fallback_actor: str = "anonymous",
    actor_query: str | None = None,
    role_query: str | None = None,
) -> tuple[str, str, list[str]]:
    actor = _actor_from_header(x_api_key, fallback=actor_query.strip() if actor_query and actor_query.strip() else fallback_actor)
    role = _role_from_identity(actor, explicit_role=(x_user_role or role_query))
    return actor, role, _permissions_for_role(role)


def _session_action_allowed(meta: dict[str, Any], actor: str, role: str, action: str) -> bool:
    if role == "admin":
        return True

    owner = actor == _session_owner(meta)
    if action == "read":
        return owner or role == "reviewer"
    if action in {"write", "compute"}:
        return owner and role in {"analyst", "reviewer"}
    if action == "draft_workflow":
        return owner and role in {"analyst", "reviewer"}
    if action == "review_workflow":
        return role == "reviewer"
    if action == "promote_ml":
        return role == "reviewer"
    if action == "execute_workflow":
        return role == "reviewer" or (owner and role in {"analyst", "reviewer"})
    if action == "export_masked":
        return owner or role == "reviewer"
    if action == "export_sensitive":
        return role == "reviewer" or (owner and role in {"analyst", "reviewer"})
    if action == "request_sensitive_export":
        return owner and role in {"analyst", "reviewer"}
    if action == "review_sensitive_export":
        return role == "reviewer"
    return False


def _require_session_action(meta: dict[str, Any], actor: str, role: str, action: str) -> None:
    if _session_action_allowed(meta, actor, role, action):
        return

    owner = _session_owner(meta)
    role_label = role or "viewer"
    raise HTTPException(
        status_code=403,
        detail=f"{role_label} is not allowed to {action.replace('_', ' ')} for session owned by {owner}.",
    )


def _authorized_session_context(
    dataset_id: str,
    *,
    action: str,
    x_api_key: str | None = None,
    x_user_role: str | None = None,
    fallback_actor: str = "anonymous",
    actor_query: str | None = None,
    role_query: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    actor, role, _ = _resolve_auth_context(
        x_api_key,
        x_user_role,
        fallback_actor=fallback_actor,
        actor_query=actor_query,
        role_query=role_query,
    )
    _require_session_action(meta, actor, role, action)
    return meta, actor, role


def _document_owner(meta: dict[str, Any]) -> str:
    return str(meta.get("created_by") or "anonymous")


def _document_action_allowed(meta: dict[str, Any], actor: str, role: str, action: str) -> bool:
    if role == "admin":
        return True

    owner = actor == _document_owner(meta)
    if action == "read":
        return owner or role == "reviewer"
    if action == "create":
        return role in {"analyst", "reviewer"}
    if action == "write":
        return owner or role == "reviewer"
    return False


def _require_document_action(meta: dict[str, Any], actor: str, role: str, action: str) -> None:
    if _document_action_allowed(meta, actor, role, action):
        return

    owner = _document_owner(meta)
    role_label = role or "viewer"
    raise HTTPException(
        status_code=403,
        detail=f"{role_label} is not allowed to {action.replace('_', ' ')} for document owned by {owner}.",
    )


def _authorized_document_context(
    document_id: str,
    *,
    action: str,
    x_api_key: str | None = None,
    x_user_role: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    meta = _load_document_meta(document_id)
    actor, role, _ = _resolve_auth_context(x_api_key, x_user_role)
    _require_document_action(meta, actor, role, action)
    return meta, actor, role


def _normalize_effective_date(value: str | None) -> str | None:
    candidate = (value or "").strip()
    if not candidate:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
            parsed = datetime.strptime(candidate, "%Y-%m-%d")
        else:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="effective_date must be a valid ISO date like YYYY-MM-DD.") from exc


def _effective_date_sort_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (0, "")
    normalized = _normalize_effective_date(value)
    return (1, normalized or "")


def _document_freshness(meta: dict[str, Any]) -> tuple[str, bool, str]:
    superseded_by = str(meta.get("superseded_by_document_id") or "").strip()
    effective_date = _normalize_effective_date(str(meta.get("effective_date") or "").strip() or None)
    today = datetime.now(timezone.utc).date().isoformat()

    if superseded_by:
        return ("superseded", False, f"Superseded by document {superseded_by}.")
    if effective_date and effective_date > today:
        return ("pending", False, f"Effective on {effective_date}.")
    status = str(meta.get("status") or "ready").strip().lower()
    if status not in {"", "ready"}:
        return (status, False, f"Document status is {status}.")
    if effective_date:
        return ("current", True, f"Effective as of {effective_date}.")
    return ("current", True, "Current trusted version.")


def _document_recency_bonus(meta: dict[str, Any]) -> float:
    freshness, _, _ = _document_freshness(meta)
    bonus = {"current": 1.25, "pending": 0.4, "superseded": -1.5}.get(freshness, -0.2)
    _, effective_date = _effective_date_sort_key(str(meta.get("effective_date") or "").strip() or None)
    if effective_date:
        age_days = (datetime.now(timezone.utc).date() - datetime.strptime(effective_date, "%Y-%m-%d").date()).days
        bonus += max(0.0, 0.5 - max(0, age_days) / 3650)
    return bonus


def _document_summary_from_meta(document_id: str, meta: dict[str, Any]) -> DocumentMetaResponse:
    content_path = _document_content_path(document_id)
    snippet_preview = ""
    if content_path.exists():
        try:
            snippet_preview = content_path.read_text(encoding="utf-8")[:240].strip()
        except OSError:
            snippet_preview = ""

    freshness, is_current, freshness_note = _document_freshness(meta)
    return DocumentMetaResponse(
        document_id=document_id,
        title=str(meta.get("title") or meta.get("file_name") or document_id),
        source_name=str(meta.get("source_name") or meta.get("title") or meta.get("file_name") or document_id),
        status=str(meta.get("status") or "ready"),
        created_at=str(meta.get("created_at") or ""),
        updated_at=str(meta.get("updated_at") or ""),
        created_by=str(meta.get("created_by") or "anonymous"),
        file_name=str(meta.get("file_name") or ""),
        file_type=str(meta.get("file_type") or ""),
        chunk_count=int(meta.get("chunk_count") or 0),
        char_count=int(meta.get("char_count") or 0),
        version_label=str(meta.get("version_label") or "") or None,
        effective_date=_normalize_effective_date(str(meta.get("effective_date") or "").strip() or None),
        supersedes_document_id=str(meta.get("supersedes_document_id") or "") or None,
        superseded_by_document_id=str(meta.get("superseded_by_document_id") or "") or None,
        is_current=is_current,
        freshness=freshness,
        freshness_note=freshness_note,
        snippet_preview=snippet_preview,
    )


def _read_document_text(content: bytes, extension: str) -> str:
    if extension in {".txt", ".md"}:
        for encoding in CSV_ENCODINGS:
            try:
                return content.decode(encoding)
            except Exception:  # noqa: PERF203
                continue
        raise ValueError("Could not decode text document with supported encodings.")

    if extension in {".html", ".htm"}:
        for encoding in CSV_ENCODINGS:
            try:
                decoded = content.decode(encoding)
                stripped = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", decoded, flags=re.IGNORECASE)
                stripped = re.sub(r"<[^>]+>", " ", stripped)
                return html.unescape(stripped)
            except Exception:  # noqa: PERF203
                continue
        raise ValueError("Could not decode HTML document with supported encodings.")

    if extension == ".json":
        try:
            parsed = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse JSON document: {exc}") from exc
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    if extension == ".pdf":
        if PdfReader is None:
            raise ValueError("PDF document support requires the pypdf package to be installed.")
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Could not parse PDF document: {exc}") from exc

        pages: list[str] = []
        for index, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
            except Exception as exc:
                raise ValueError(f"Could not extract text from PDF page {index + 1}: {exc}") from exc
            extracted = extracted.strip()
            if extracted:
                pages.append(f"[Page {index + 1}]\n{extracted}")

        if not pages:
            raise ValueError("PDF did not contain extractable text. Scanned PDFs without a text layer are not supported yet.")
        return "\n\n".join(pages)

    raise ValueError(f"Unsupported document type '{extension}'.")


def _normalize_document_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def _chunk_document_text(text: str) -> list[str]:
    normalized = _normalize_document_text(text)
    if not normalized:
        return []

    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", normalized) if segment.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= DOCUMENT_CHUNK_SIZE:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= DOCUMENT_CHUNK_SIZE:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + DOCUMENT_CHUNK_SIZE)
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(paragraph):
                break
            start = max(end - DOCUMENT_CHUNK_OVERLAP, start + 1)
        current = ""
    if current:
        chunks.append(current)
    return chunks[:100]


def _tokenize_document_query(text: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) >= 3]
    return [token for token in tokens if token not in DOCUMENT_STOPWORDS]


def _document_snippet(text: str, max_chars: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _score_document_chunk(query: str, query_tokens: list[str], chunk_text: str, title: str, source_name: str) -> float:
    haystack = f"{title} {source_name} {chunk_text}".lower()
    if not haystack.strip():
        return 0.0

    score = 0.0
    lowered_query = query.lower().strip()
    if lowered_query and lowered_query in haystack:
        score += 6.0

    chunk_tokens = set(_tokenize_document_query(haystack))
    for token in query_tokens:
        if token in chunk_tokens:
            score += 1.5
        if token in title.lower():
            score += 1.0
        if token in source_name.lower():
            score += 0.75

    return score


def _load_document_chunks(document_id: str) -> list[str]:
    path = _document_chunks_path(document_id)
    if not path.exists():
        return []
    payload = _load_json(path)
    chunks = payload.get("chunks") or []
    return [str(chunk) for chunk in chunks if isinstance(chunk, str)]


def _iter_accessible_documents(actor: str, role: str) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for child in sorted(DOCUMENT_LIBRARY_DIR.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = _load_json(meta_path)
            if not _document_action_allowed(meta, actor, role, "read"):
                continue
            items.append((str(meta.get("document_id") or child.name), meta))
        except Exception:
            continue
    return items


def _search_documents_internal(
    actor: str,
    role: str,
    query: str,
    *,
    document_ids: list[str] | None = None,
    limit: int = 5,
) -> list[DocumentSearchHit]:
    requested_ids = {document_id for document_id in (document_ids or []) if document_id}
    query_tokens = _tokenize_document_query(query)
    candidates: list[DocumentSearchHit] = []

    for document_id, meta in _iter_accessible_documents(actor, role):
        if requested_ids and document_id not in requested_ids:
            continue
        title = str(meta.get("title") or meta.get("file_name") or document_id)
        source_name = str(meta.get("source_name") or title)
        version_label = str(meta.get("version_label") or "").strip() or None
        effective_date = _normalize_effective_date(str(meta.get("effective_date") or "").strip() or None)
        freshness, _, _ = _document_freshness(meta)
        chunks = _load_document_chunks(document_id)
        for index, chunk in enumerate(chunks):
            score = _score_document_chunk(query, query_tokens, chunk, title, source_name) + _document_recency_bonus(meta)
            if score <= 0:
                continue
            candidates.append(
                DocumentSearchHit(
                    citation_key=f"doc:{document_id}:{index}",
                    document_id=document_id,
                    title=title,
                    source_name=source_name,
                    snippet=_document_snippet(chunk),
                    chunk_index=index,
                    score=round(score, 3),
                    version_label=version_label,
                    effective_date=effective_date,
                    freshness=freshness,
                )
            )

    candidates.sort(key=lambda item: (item.score, item.effective_date or "", item.document_id), reverse=True)
    return candidates[:limit]


def _answer_documents(question: str, hits: list[DocumentSearchHit]) -> DocumentAskResponse:
    if not hits:
        return DocumentAskResponse(
            answer="No grounded answer was found in the trusted document library for that question.",
            grounded=False,
            confidence="Low",
            citations=[],
            freshness_summary="No trusted document citations matched the question.",
        )

    lead = hits[0]
    supporting = hits[1:3]
    lead_label = lead.title
    if lead.version_label:
        lead_label = f"{lead_label} ({lead.version_label})"
    if lead.freshness != "current":
        lead_label = f"{lead_label} [{lead.freshness}]"
    answer_parts = [
        f"The most relevant trusted source is {lead_label}.",
        f"It states: \"{lead.snippet}\"",
    ]
    if supporting:
        answer_parts.append(
            "Supporting documents add: "
            + " ".join(
                f"{item.title}{f' ({item.version_label})' if item.version_label else ''}: \"{item.snippet}\""
                for item in supporting
            )
        )

    freshness_parts: list[str] = []
    if lead.effective_date:
        freshness_parts.append(f"Lead source effective date: {lead.effective_date}.")
    if lead.freshness != "current":
        freshness_parts.append(f"Lead source freshness: {lead.freshness}.")
    superseded_hits = [item for item in hits if item.freshness == "superseded"]
    if superseded_hits:
        freshness_parts.append("One or more supporting citations are superseded, so review the current policy version before acting.")

    citations = [
        DocumentCitation(
            citation_key=item.citation_key,
            document_id=item.document_id,
            title=item.title,
            source_name=item.source_name,
            snippet=item.snippet,
            chunk_index=item.chunk_index,
            version_label=item.version_label,
            effective_date=item.effective_date,
            freshness=item.freshness,
        )
        for item in hits
    ]
    confidence = "High" if len(hits) >= 2 and hits[0].score >= 3 else "Medium"
    return DocumentAskResponse(
        answer=" ".join(answer_parts),
        grounded=True,
        confidence=confidence,
        citations=citations,
        freshness_summary=" ".join(freshness_parts).strip(),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _etag_for(dataset_id: str, artifact: str, dataset_hash: str, schema_hash: str | None = None, params: str = "") -> str:
    token = "|".join([dataset_id, artifact, dataset_hash, schema_hash or "", params])
    return _sha256_text(token)[:32]


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = create_llm_client_from_env()
    return _llm_client


def _detect_csv_encoding(content: bytes) -> str:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            decoded = content.decode(encoding)
            pd.read_csv(io.StringIO(decoded), nrows=200)
            return encoding
        except Exception as exc:  # noqa: PERF203
            last_error = exc
    raise ValueError(f"Could not parse CSV with supported encodings. Last error: {last_error}")


def _validate_excel_sheet(content: bytes, selected_sheet: str | None) -> tuple[list[str], str]:
    with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as workbook:
        sheets = workbook.sheet_names
    if not sheets:
        raise ValueError("The Excel workbook does not contain sheets.")
    if selected_sheet and selected_sheet not in sheets:
        raise ValueError(f"Selected sheet '{selected_sheet}' not found. Available: {sheets}")
    return sheets, (selected_sheet or sheets[0])


def _to_json_compatible_rows(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    preview = df.head(limit).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].astype("string")
    payload = json.loads(preview.to_json(orient="records", date_format="iso"))
    return payload


def _read_uploaded_file(meta: dict[str, Any]) -> pd.DataFrame:
    file_meta = meta.get("file") or {}
    raw_path = file_meta.get("raw_path")
    extension = file_meta.get("extension")
    if not raw_path or not extension:
        raise ValueError("Session does not have an uploaded file.")

    path = Path(raw_path)
    if not path.exists():
        raise ValueError("Uploaded file cannot be found on disk.")

    if extension == ".csv":
        encoding = file_meta.get("encoding", "utf-8")
        return pd.read_csv(path, encoding=encoding, encoding_errors="replace")

    if extension == ".xlsx":
        sheet_name = file_meta.get("sheet_name")
        return pd.read_excel(path, sheet_name=sheet_name)

    raise ValueError("Unsupported file extension.")


def _compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_signature(meta: dict[str, Any]) -> str:
    file_meta = meta.get("file") or {}
    raw_path = file_meta.get("raw_path")
    if not raw_path:
        return ""
    return _compute_file_hash(Path(raw_path))


def _compute_schema_hash(df: pd.DataFrame) -> str:
    columns = [{"name": str(column), "dtype": str(df[column].dtype)} for column in df.columns]
    return _sha256_text(json.dumps(columns, sort_keys=True, ensure_ascii=True))


def _cache_key(dataset_id: str, artifact: str, file_hash: str, schema_hash: str = "", params: str = "default") -> str:
    signature = f"{file_hash}:{schema_hash}" if schema_hash else file_hash
    return f"cache:{CACHE_VERSION}:{dataset_id}:{artifact}:{signature}:{params}"


def _seed_from_dataset_hash(dataset_hash: str) -> int:
    if not dataset_hash:
        return 42
    return int(dataset_hash[:8], 16)


def _deterministic_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed, replace=False)


def _determine_sampling_strategy(df: pd.DataFrame, profile: dict[str, Any], seed: int, max_rows: int) -> tuple[pd.DataFrame, str]:
    if len(df) <= max_rows:
        return df, "uniform"

    datetime_cols = profile.get("datetime_cols", [])
    categorical_cols = profile.get("categorical_cols", [])
    geo_candidates = [col for col in categorical_cols if any(key in str(col).lower() for key in GEO_KEYWORDS)]

    if datetime_cols:
        time_col = datetime_cols[0]
        scoped = df.copy()
        scoped[time_col] = pd.to_datetime(scoped[time_col], errors="coerce")
        scoped = scoped.dropna(subset=[time_col]).sort_values(time_col)
        if not scoped.empty:
            head_n = max(1, max_rows // 5)
            tail_n = max(1, max_rows // 5)
            core_n = max(0, max_rows - head_n - tail_n)
            head_df = scoped.head(head_n)
            tail_df = scoped.tail(tail_n)
            middle = scoped.iloc[head_n:-tail_n] if len(scoped) > (head_n + tail_n) else scoped.iloc[0:0]
            middle_sample = _deterministic_sample(middle, core_n, seed)
            sampled = pd.concat([head_df, middle_sample, tail_df]).drop_duplicates()
            sampled = sampled.head(max_rows)
            return sampled, "time_stratified"

    if geo_candidates:
        key_col = geo_candidates[0]
        groups = []
        grouped = df.groupby(key_col, dropna=False)
        per_group = max(1, max_rows // max(1, grouped.ngroups))
        for _, chunk in grouped:
            groups.append(_deterministic_sample(chunk, per_group, seed))
        sampled = pd.concat(groups).drop_duplicates()
        if len(sampled) > max_rows:
            sampled = _deterministic_sample(sampled, max_rows, seed)
        return sampled, "stratified"

    if categorical_cols:
        key_col = categorical_cols[0]
        grouped = df.groupby(key_col, dropna=False)
        groups = []
        per_group = max(1, max_rows // max(1, grouped.ngroups))
        for _, chunk in grouped:
            groups.append(_deterministic_sample(chunk, per_group, seed))
        sampled = pd.concat(groups).drop_duplicates()
        if len(sampled) > max_rows:
            sampled = _deterministic_sample(sampled, max_rows, seed)
        return sampled, "stratified"

    return _deterministic_sample(df, max_rows, seed), "uniform"


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return "string"
        parsed = pd.to_datetime(non_null.astype(str), errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return "datetime"
        return "string"
    return "unknown"


def _outlier_count(series: pd.Series) -> int:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 4:
        return 0

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if not pd.notna(iqr) or iqr == 0:
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def _detect_pii_reasons(column: str, series: pd.Series) -> list[str]:
    reasons: list[str] = []
    lower_name = column.lower()
    if any(keyword in lower_name for keyword in PII_KEYWORDS):
        reasons.append("name_keyword")

    sampled = series.dropna().astype(str).head(300)
    if sampled.empty:
        return reasons

    email_ratio = sampled.str.match(EMAIL_PATTERN).mean()
    phone_ratio = sampled.str.match(PHONE_PATTERN).mean()
    id_ratio = sampled.str.match(ID_VALUE_PATTERN).mean()

    if email_ratio >= 0.2:
        reasons.append("email_pattern")
    if phone_ratio >= 0.2:
        reasons.append("phone_pattern")
    if id_ratio >= 0.7 and len(sampled) >= 20:
        reasons.append("identifier_pattern")

    return reasons


def _is_id_like_column(column: str, series: pd.Series, inferred_type: str) -> bool:
    lower_name = column.lower()
    if any(keyword in lower_name for keyword in ID_KEYWORDS):
        return True

    if inferred_type not in {"string", "number"}:
        return False
    non_null = series.dropna()
    if len(non_null) < 20:
        return False
    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    return unique_ratio >= 0.95


def _quality_score(
    missing_percent_map: dict[str, float], duplicate_percent: float, outlier_column_count: int, total_columns: int
) -> float:
    if total_columns == 0:
        return 0.0

    avg_missing = sum(missing_percent_map.values()) / total_columns
    high_missing_ratio = sum(1 for value in missing_percent_map.values() if value > 20) / total_columns

    penalty_missing = min(45.0, avg_missing * 0.8)
    penalty_duplicates = min(30.0, duplicate_percent * 1.2)
    penalty_high_missing_cols = min(15.0, high_missing_ratio * 20.0)
    penalty_outliers = min(10.0, (outlier_column_count / total_columns) * 15.0)
    score = 100.0 - (penalty_missing + penalty_duplicates + penalty_high_missing_cols + penalty_outliers)
    return round(max(0.0, min(100.0, score)), 1)


def _build_profile(df: pd.DataFrame) -> dict[str, Any]:
    row_count, column_count = df.shape
    duplicate_rows = int(df.duplicated().sum()) if row_count else 0
    duplicate_percent = round((duplicate_rows / row_count) * 100, 2) if row_count else 0.0

    missing_percent: dict[str, float] = {}
    dtypes: dict[str, str] = {}
    datetime_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    pii_candidates: list[str] = []
    id_like_columns: list[str] = []
    column_profiles: list[dict[str, Any]] = []
    outlier_column_count = 0

    for column in df.columns:
        series = df[column]
        missing_pct = round(float(series.isna().mean() * 100), 2) if row_count else 0.0
        missing_percent[column] = missing_pct
        dtypes[column] = str(series.dtype)

        inferred_type = _infer_column_type(series)
        if inferred_type == "datetime":
            datetime_cols.append(column)
        elif inferred_type == "number":
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)

        outliers = _outlier_count(series) if inferred_type == "number" else 0
        if outliers > 0:
            outlier_column_count += 1

        pii_reasons = _detect_pii_reasons(column, series)
        if pii_reasons:
            pii_candidates.append(column)

        id_like = _is_id_like_column(column, series, inferred_type)
        if id_like:
            id_like_columns.append(column)

        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        unique_percent = round((unique_count / max(len(non_null), 1)) * 100, 2) if len(non_null) else 0.0

        column_profiles.append(
            {
                "name": column,
                "dtype": dtypes[column],
                "inferred_type": inferred_type,
                "missing_percent": missing_pct,
                "unique_count": unique_count,
                "unique_percent": unique_percent,
                "outlier_count": outliers,
                "outlier_percent": round((outliers / row_count) * 100, 2) if row_count else 0.0,
                "is_id_like": id_like,
                "is_pii_candidate": bool(pii_reasons),
                "pii_reasons": pii_reasons,
                "sample_values": [str(value) for value in non_null.head(5).tolist()],
            }
        )

    score = _quality_score(missing_percent, duplicate_percent, outlier_column_count, column_count)
    missing_cells = int(df.isna().sum().sum()) if row_count and column_count else 0
    total_cells = row_count * column_count
    completeness_percent = round(100 - ((missing_cells / total_cells) * 100), 2) if total_cells else 0.0

    columns_lower = {str(column).lower() for column in df.columns}
    matched_template_fields = [field for field in HMIS_TEMPLATE_FIELDS if field in columns_lower]
    template_name = "hmis" if len(matched_template_fields) >= 4 else "generic"
    health_signals = {
        "time_columns": [col for col in df.columns if any(key in str(col).lower() for key in TIME_KEYWORDS)],
        "geography_columns": [col for col in df.columns if any(key in str(col).lower() for key in GEO_KEYWORDS)],
        "service_columns": [col for col in df.columns if any(key in str(col).lower() for key in SERVICE_KEYWORDS)],
    }

    return {
        "generated_at": _utc_now_iso(),
        "shape": {"rows": int(row_count), "cols": int(column_count)},
        "quality_score": score,
        "completeness_percent": completeness_percent,
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": duplicate_percent,
        "dtypes": dtypes,
        "missing_percent": missing_percent,
        "datetime_cols": datetime_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "pii_candidates": sorted(set(pii_candidates)),
        "id_like_columns": sorted(set(id_like_columns)),
        "columns": column_profiles,
        "health_template": {
            "name": template_name,
            "matched_fields": matched_template_fields,
        },
        "health_signals": health_signals,
    }


def _apply_profile_masking(profile: dict[str, Any], mask_enabled: bool) -> dict[str, Any]:
    if not mask_enabled:
        return profile

    pii_columns = profile.get("pii_candidates", [])
    aliases = {column: f"pii_field_{index + 1}" for index, column in enumerate(pii_columns)}
    if not aliases:
        masked = dict(profile)
        masked["masking"] = {"enabled": True, "masked_columns": []}
        return masked

    def mask_value(value: str) -> str:
        return aliases.get(value, value)

    masked = json.loads(json.dumps(profile))
    masked["dtypes"] = {mask_value(key): value for key, value in masked.get("dtypes", {}).items()}
    masked["missing_percent"] = {
        mask_value(key): value for key, value in masked.get("missing_percent", {}).items()
    }
    masked["datetime_cols"] = [mask_value(name) for name in masked.get("datetime_cols", [])]
    masked["numeric_cols"] = [mask_value(name) for name in masked.get("numeric_cols", [])]
    masked["categorical_cols"] = [mask_value(name) for name in masked.get("categorical_cols", [])]
    masked["pii_candidates"] = [mask_value(name) for name in masked.get("pii_candidates", [])]
    masked["id_like_columns"] = [mask_value(name) for name in masked.get("id_like_columns", [])]

    for column in masked.get("columns", []):
        column["name"] = mask_value(column.get("name", ""))

    masked["masking"] = {
        "enabled": True,
        "masked_columns": sorted(set(aliases.values())),
    }
    return masked


def _can_export_sensitive(meta: dict[str, Any]) -> bool:
    return bool(meta.get("allow_sensitive_export", False))


def _default_sensitive_export_approval() -> dict[str, Any]:
    return {
        "status": "not_requested",
        "requested_by": None,
        "requested_at": None,
        "justification": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "review_note": None,
    }


def _sensitive_export_approval(meta: dict[str, Any]) -> dict[str, Any]:
    approval = meta.get("sensitive_export_approval")
    if not isinstance(approval, dict):
        approval = _default_sensitive_export_approval()
        meta["sensitive_export_approval"] = dict(approval)
        return dict(approval)

    normalized = _default_sensitive_export_approval()
    normalized.update(
        {
            key: approval.get(key)
            for key in normalized
            if key in approval
        }
    )
    meta["sensitive_export_approval"] = normalized
    return dict(normalized)


def _load_profile_if_exists(dataset_id: str) -> dict[str, Any] | None:
    path = _profile_path(dataset_id)
    if path.exists():
        return _load_json(path)
    return None


def _column_profile_index(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    columns = profile.get("columns", []) if isinstance(profile, dict) else []
    return {
        str(column.get("name")): column
        for column in columns
        if isinstance(column, dict) and column.get("name")
    }


def _preferred_dimension_candidates(profile: dict[str, Any]) -> list[str]:
    health_signals = profile.get("health_signals", {}) if isinstance(profile, dict) else {}
    preferred = [
        *health_signals.get("geography_columns", []),
        *health_signals.get("service_columns", []),
        *profile.get("categorical_cols", []),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    column_index = _column_profile_index(profile)
    for raw_name in preferred:
        name = str(raw_name)
        if not name or name in seen:
            continue
        seen.add(name)
        column = column_index.get(name, {})
        if column.get("is_pii_candidate") or column.get("is_id_like"):
            continue
        unique_count = int(column.get("unique_count") or 0)
        if 2 <= unique_count <= 20:
            ordered.append(name)
    return ordered


def _numeric_metric_candidates(profile: dict[str, Any]) -> list[str]:
    column_index = _column_profile_index(profile)
    candidates: list[str] = []
    for raw_name in profile.get("numeric_cols", []):
        name = str(raw_name)
        column = column_index.get(name, {})
        if column.get("is_pii_candidate") or column.get("is_id_like"):
            continue
        unique_count = int(column.get("unique_count") or 0)
        if unique_count >= 3:
            candidates.append(name)
    return candidates


def _time_candidates(profile: dict[str, Any]) -> list[str]:
    health_signals = profile.get("health_signals", {}) if isinstance(profile, dict) else {}
    preferred = [
        *health_signals.get("time_columns", []),
        *profile.get("datetime_cols", []),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_name in preferred:
        name = str(raw_name)
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _cohort_operator_options(inferred_type: str) -> set[str]:
    return set(COHORT_OPERATORS_BY_TYPE.get(inferred_type, COHORT_OPERATORS_BY_TYPE["string"]))


def _cohort_value_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None and str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _coerce_cohort_scalar(raw_value: Any, inferred_type: str, field: str) -> Any:
    if inferred_type == "number":
        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Cohort field '{field}' expects a numeric value.") from exc

    if inferred_type == "datetime":
        parsed = pd.to_datetime(raw_value, errors="coerce")
        if pd.isna(parsed):
            raise HTTPException(status_code=400, detail=f"Cohort field '{field}' expects a valid date/time value.")
        return parsed

    return str(raw_value).strip()


def _normalize_cohort_criterion(
    criterion: CohortCriterion,
    df: pd.DataFrame,
    profile: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    field = criterion.field.strip()
    operator = criterion.operator.strip().lower()
    if not field:
        raise HTTPException(status_code=400, detail="Cohort criteria require a field name.")

    column_index = _column_profile_index(profile)
    column_profile = column_index.get(field)
    if column_profile is None or field not in df.columns:
        raise HTTPException(status_code=400, detail=f"Cohort field '{field}' is not available in this dataset.")
    if column_profile.get("is_pii_candidate") or column_profile.get("is_id_like"):
        raise HTTPException(status_code=400, detail=f"Cohort field '{field}' is blocked by governance controls.")

    inferred_type = str(column_profile.get("inferred_type") or "string")
    allowed_operators = _cohort_operator_options(inferred_type)
    if operator not in allowed_operators:
        raise HTTPException(
            status_code=400,
            detail=f"Operator '{operator}' is not allowed for cohort field '{field}' ({inferred_type}).",
        )

    series = df[field]
    normalized_value: Any = None

    if operator == "is_null":
        mask = series.isna()
    elif operator == "not_null":
        mask = series.notna()
    elif inferred_type == "number":
        numeric_series = pd.to_numeric(series, errors="coerce")
        if operator in {"in", "not_in"}:
            normalized_value = [_coerce_cohort_scalar(item, inferred_type, field) for item in _cohort_value_list(criterion.value)]
            if not normalized_value:
                raise HTTPException(status_code=400, detail=f"Cohort field '{field}' requires at least one value.")
            mask = numeric_series.isin(normalized_value)
            if operator == "not_in":
                mask = ~mask
        elif operator == "between":
            bounds = [_coerce_cohort_scalar(item, inferred_type, field) for item in _cohort_value_list(criterion.value)]
            if len(bounds) != 2:
                raise HTTPException(status_code=400, detail=f"Cohort field '{field}' requires two values for 'between'.")
            low, high = sorted(bounds)
            normalized_value = [low, high]
            mask = numeric_series.between(low, high, inclusive="both")
        else:
            normalized_value = _coerce_cohort_scalar(criterion.value, inferred_type, field)
            comparisons = {
                "eq": numeric_series.eq(normalized_value),
                "neq": numeric_series.ne(normalized_value),
                "gt": numeric_series.gt(normalized_value),
                "gte": numeric_series.ge(normalized_value),
                "lt": numeric_series.lt(normalized_value),
                "lte": numeric_series.le(normalized_value),
            }
            mask = comparisons[operator]
    elif inferred_type == "datetime":
        time_series = pd.to_datetime(series, errors="coerce")
        if operator in {"in", "not_in"}:
            normalized_value = [_coerce_cohort_scalar(item, inferred_type, field) for item in _cohort_value_list(criterion.value)]
            if not normalized_value:
                raise HTTPException(status_code=400, detail=f"Cohort field '{field}' requires at least one value.")
            normalized_index = {value.isoformat() for value in normalized_value}
            comparable = time_series.dt.strftime("%Y-%m-%dT%H:%M:%S")
            mask = comparable.isin(normalized_index)
            if operator == "not_in":
                mask = ~mask
            normalized_value = [value.isoformat() for value in normalized_value]
        elif operator == "between":
            bounds = [_coerce_cohort_scalar(item, inferred_type, field) for item in _cohort_value_list(criterion.value)]
            if len(bounds) != 2:
                raise HTTPException(status_code=400, detail=f"Cohort field '{field}' requires two values for 'between'.")
            low, high = sorted(bounds)
            normalized_value = [low.isoformat(), high.isoformat()]
            mask = time_series.between(low, high, inclusive="both")
        else:
            normalized_scalar = _coerce_cohort_scalar(criterion.value, inferred_type, field)
            normalized_value = normalized_scalar.isoformat()
            comparisons = {
                "eq": time_series.eq(normalized_scalar),
                "neq": time_series.ne(normalized_scalar),
                "gt": time_series.gt(normalized_scalar),
                "gte": time_series.ge(normalized_scalar),
                "lt": time_series.lt(normalized_scalar),
                "lte": time_series.le(normalized_scalar),
            }
            mask = comparisons[operator]
    else:
        string_series = series.astype("string").str.strip()
        comparable = string_series.str.lower()
        if operator in {"in", "not_in"}:
            normalized_value = [str(item).strip() for item in _cohort_value_list(criterion.value)]
            if not normalized_value:
                raise HTTPException(status_code=400, detail=f"Cohort field '{field}' requires at least one value.")
            normalized_lookup = {item.lower() for item in normalized_value}
            mask = comparable.isin(normalized_lookup)
            if operator == "not_in":
                mask = ~mask
        else:
            normalized_value = str(criterion.value or "").strip()
            if operator not in {"is_null", "not_null"} and not normalized_value:
                raise HTTPException(status_code=400, detail=f"Cohort field '{field}' requires a value for operator '{operator}'.")
            lowered_value = normalized_value.lower()
            comparisons = {
                "eq": comparable.eq(lowered_value),
                "neq": comparable.ne(lowered_value),
                "contains": comparable.str.contains(re.escape(lowered_value), na=False),
                "starts_with": comparable.str.startswith(lowered_value, na=False),
            }
            mask = comparisons[operator]

    normalized = {
        "field": field,
        "operator": operator,
        "operator_label": COHORT_OPERATOR_LABELS.get(operator, operator.replace("_", " ")),
        "value": normalized_value,
        "inferred_type": inferred_type,
    }
    return mask.fillna(False), normalized


def _describe_cohort_criterion(criterion: dict[str, Any]) -> str:
    field = str(criterion.get("field") or "field")
    operator = str(criterion.get("operator") or "eq")
    value = criterion.get("value")

    if operator == "is_null":
        return f"{field} is blank"
    if operator == "not_null":
        return f"{field} is present"
    if operator == "between" and isinstance(value, list) and len(value) == 2:
        return f"{field} is between {value[0]} and {value[1]}"
    if operator in {"in", "not_in"} and isinstance(value, list):
        joined = ", ".join(str(item) for item in value[:5])
        verb = "is one of" if operator == "in" else "is not one of"
        return f"{field} {verb} {joined}"
    if operator == "contains":
        return f"{field} contains '{value}'"
    if operator == "starts_with":
        return f"{field} starts with '{value}'"
    return f"{field} {COHORT_OPERATOR_LABELS.get(operator, operator)} {value}"


def _cohort_preview_columns(df: pd.DataFrame, profile: dict[str, Any], criteria_fields: list[str]) -> tuple[list[str], list[str]]:
    column_index = _column_profile_index(profile)
    blocked = {
        name
        for name, column in column_index.items()
        if column.get("is_pii_candidate") or column.get("is_id_like")
    }

    ordered_candidates = [
        *criteria_fields,
        *_preferred_dimension_candidates(profile),
        *_time_candidates(profile),
        *_numeric_metric_candidates(profile),
    ]

    preview_columns: list[str] = []
    seen: set[str] = set()
    for raw_name in ordered_candidates:
        name = str(raw_name)
        if not name or name in seen or name in blocked or name not in df.columns:
            continue
        seen.add(name)
        preview_columns.append(name)
        if len(preview_columns) >= 8:
            break

    if not preview_columns:
        for column in df.columns:
            name = str(column)
            if name in blocked:
                continue
            preview_columns.append(name)
            if len(preview_columns) >= 8:
                break

    return preview_columns, sorted(blocked)


def _build_cohort_analysis(
    df: pd.DataFrame,
    profile: dict[str, Any],
    request: CohortBuildRequest,
) -> dict[str, Any]:
    if not request.criteria:
        raise HTTPException(status_code=400, detail="Add at least one cohort criterion before running the builder.")

    combined_mask = pd.Series(True, index=df.index)
    normalized_criteria: list[dict[str, Any]] = []
    for criterion in request.criteria:
        mask, normalized = _normalize_cohort_criterion(criterion, df, profile)
        combined_mask &= mask
        normalized_criteria.append(normalized)

    filtered_df = df.loc[combined_mask].copy()
    preview_columns, blocked_columns = _cohort_preview_columns(
        filtered_df if not filtered_df.empty else df,
        profile,
        [item["field"] for item in normalized_criteria],
    )
    preview_source = filtered_df[preview_columns] if preview_columns and not filtered_df.empty else filtered_df
    preview_rows = _to_json_compatible_rows(preview_source, request.limit) if not preview_source.empty else []

    population_row_count = int(len(df.index))
    row_count = int(len(filtered_df.index))
    match_percent = round((row_count / population_row_count) * 100, 2) if population_row_count else 0.0
    criteria_summary = "; ".join(_describe_cohort_criterion(item) for item in normalized_criteria[:4])
    cohort_name = request.name.strip() if request.name and request.name.strip() else "Governed cohort"

    if row_count:
        summary = (
            f"{cohort_name} matched {row_count} row(s) out of {population_row_count} "
            f"({match_percent:.2f}% of the governed dataset) using {criteria_summary}."
        )
    else:
        summary = (
            f"{cohort_name} matched no rows in the governed dataset. "
            f"Review the criteria combination: {criteria_summary}."
        )

    preferred_dimension = next((item for item in _preferred_dimension_candidates(profile) if item in df.columns), None)
    preferred_time = next((item for item in _time_candidates(profile) if item in df.columns), None)
    preferred_metric = next((item for item in _numeric_metric_candidates(profile) if item in df.columns), None)
    suggested_questions = [
        preferred_dimension and row_count
        and f"Break this cohort down by {preferred_dimension} and explain the largest segment differences.",
        preferred_time and row_count and f"How has this cohort changed over time using {preferred_time}?",
        preferred_metric and row_count and f"Compare {preferred_metric} for this cohort against the full dataset.",
    ]

    return {
        "generated_at": _utc_now_iso(),
        "name": cohort_name,
        "description": request.description.strip() if request.description and request.description.strip() else None,
        "row_count": row_count,
        "population_row_count": population_row_count,
        "criteria_count": len(normalized_criteria),
        "criteria": normalized_criteria,
        "preview_columns": preview_columns,
        "preview_rows": preview_rows,
        "excluded_columns": blocked_columns,
        "summary": summary,
        "suggested_questions": [item for item in suggested_questions if item][:4],
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return _load_json(path)
    return None


def _workflow_priority_from_context(profile: dict[str, Any] | None, anomalies: dict[str, Any] | None) -> str:
    if anomalies and int(anomalies.get("anomaly_count") or 0) > 0:
        strongest = (anomalies.get("anomalies") or [{}])[0]
        return "high" if str(strongest.get("severity") or "").lower() == "high" else "medium"
    if profile and float(profile.get("quality_score") or 100.0) < 70:
        return "medium"
    return "normal"


def _workflow_evidence(
    dataset_id: str,
    meta: dict[str, Any],
    profile: dict[str, Any] | None,
    facts: dict[str, Any] | None,
    anomalies: dict[str, Any] | None,
    cohort: dict[str, Any] | None,
) -> list[str]:
    evidence: list[str] = []
    dataset_label = str(meta.get("display_name") or (meta.get("file") or {}).get("filename") or dataset_id)
    evidence.append(f"Session: {dataset_label}")

    if profile:
        shape = profile.get("shape") or {}
        evidence.append(
            f"Profile quality score {float(profile.get('quality_score') or 0.0):.1f} across "
            f"{int(shape.get('rows') or 0)} rows and {int(shape.get('cols') or 0)} columns."
        )

    if facts:
        coverage = facts.get("data_coverage") or {}
        evidence.append(
            f"Facts bundle coverage mode {coverage.get('mode', 'auto')} using {int(coverage.get('rows_used') or 0)} rows."
        )

    if anomalies and int(anomalies.get("anomaly_count") or 0) > 0:
        first_anomaly = (anomalies.get("anomalies") or [{}])[0]
        evidence.append(f"Strongest anomaly: {first_anomaly.get('title', 'governed anomaly')}.")

    if cohort and int(cohort.get("row_count") or 0) > 0:
        evidence.append(
            f"Latest cohort matched {int(cohort.get('row_count') or 0)} of {int(cohort.get('population_row_count') or 0)} rows."
        )

    return evidence[:6]


def _workflow_action_content(
    dataset_id: str,
    meta: dict[str, Any],
    request: WorkflowDraftRequest,
    profile: dict[str, Any] | None,
    facts: dict[str, Any] | None,
    anomalies: dict[str, Any] | None,
    cohort: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any], list[str]]:
    action_type = request.action_type.strip().lower()
    if action_type not in WORKFLOW_ACTION_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported workflow action type '{request.action_type}'.")

    dataset_label = str(meta.get("display_name") or (meta.get("file") or {}).get("filename") or dataset_id)
    target = request.target.strip() if request.target and request.target.strip() else None
    objective = request.objective.strip() if request.objective and request.objective.strip() else None
    evidence = _workflow_evidence(dataset_id, meta, profile, facts, anomalies, cohort)
    priority = _workflow_priority_from_context(profile, anomalies)
    signal = evidence[1] if len(evidence) > 1 else f"Governed analytics are available for {dataset_label}."
    anomaly_signal = evidence[3] if len(evidence) > 3 else signal

    if action_type == "draft_email":
        title = request.title.strip() if request.title and request.title.strip() else f"Operations follow-up for {dataset_label}"
        audience = target or "Operations team"
        summary = f"Prepared a governed email draft for {audience} and queued it for approval before manual sending."
        payload = {
            "channel": "email",
            "to": audience,
            "subject": title,
            "body": (
                f"Hello {audience},\n\n"
                f"We reviewed governed analytics for {dataset_label}. {signal} {anomaly_signal}\n\n"
                f"Requested objective: {objective or 'Review the surfaced signal and agree on next steps.'}\n\n"
                "This draft is approval-gated and has not been sent automatically."
            ),
        }
        return title, summary, payload, evidence

    if action_type == "create_ticket":
        title = request.title.strip() if request.title and request.title.strip() else f"Investigate governed signal for {dataset_label}"
        owner = target or "Quality operations"
        summary = f"Prepared an investigation ticket draft for {owner} with evidence-backed context and approval controls."
        payload = {
            "system": "governed_investigation_queue",
            "owner_recommendation": owner,
            "priority": priority,
            "title": title,
            "description": (
                f"Objective: {objective or 'Investigate the governed analytics signal and confirm impact.'}\n"
                f"Dataset: {dataset_label}\n"
                + "\n".join(f"- {item}" for item in evidence)
            ),
        }
        return title, summary, payload, evidence

    if action_type == "action_plan":
        title = request.title.strip() if request.title and request.title.strip() else f"Action plan for {dataset_label}"
        owner = target or "Quality improvement lead"
        summary = f"Prepared a governed action plan for {owner} with explicit next steps and review requirements."
        payload = {
            "plan_owner": owner,
            "objective": objective or "Stabilize the surfaced signal and confirm follow-up analysis.",
            "steps": [
                "Review the governed evidence with the responsible team.",
                "Assign an owner and due date for the corrective action.",
                "Run a follow-up governed analysis after the intervention.",
            ],
            "notes": evidence,
        }
        return title, summary, payload, evidence

    title = request.title.strip() if request.title and request.title.strip() else f"Schedule report for {dataset_label}"
    audience = target or "Quality committee"
    summary = f"Prepared a governed report scheduling draft for {audience}; execution remains approval-gated."
    payload = {
        "schedule": "monthly",
        "report_template": "health_report",
        "sections": ["quality", "kpis", "trends", "limitations"],
        "audience": audience,
        "objective": objective or "Deliver a recurring governed summary with approved evidence.",
        "delivery_note": "Scheduler integration is not connected yet; execution is recorded as a manual governed task.",
        "evidence": evidence,
    }
    return title, summary, payload, evidence


def _workflow_action_execution_result(action_type: str) -> str:
    if action_type == "draft_email":
        return "Email draft marked ready for manual sending; external delivery integration is pending."
    if action_type == "create_ticket":
        return "Ticket draft marked executed in the governed workflow journal; external issue tracker integration is pending."
    if action_type == "action_plan":
        return "Action plan marked executed and ready for operational handoff."
    return "Report scheduling request marked executed as a governed manual task."


def _create_report_schedule_from_action(
    dataset_id: str,
    action: dict[str, Any],
    actor: str,
) -> dict[str, Any]:
    payload = dict(action.get("payload") or {})
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        sections = ["quality", "kpis", "trends", "limitations"]
    schedule = {
        "schedule_id": str(uuid.uuid4()),
        "title": str(action.get("title") or "Scheduled report"),
        "frequency": str(payload.get("schedule") or "monthly"),
        "report_template": str(payload.get("report_template") or "health_report"),
        "sections": [str(item) for item in sections if str(item).strip()],
        "audience": str(payload.get("audience") or action.get("target") or "").strip() or None,
        "objective": str(payload.get("objective") or action.get("objective") or "").strip() or None,
        "delivery_note": str(payload.get("delivery_note") or "").strip() or None,
        "status": "active",
        "source_action_id": str(action.get("action_id") or "") or None,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "created_by": actor,
        "last_run_at": None,
        "last_job_id": None,
        "last_run_status": None,
    }
    schedules = [schedule, *_load_report_schedules(dataset_id)]
    _save_report_schedules(dataset_id, schedules)
    return schedule


def _queue_report_job(dataset_id: str, actor: str, template: str | None, sections: list[str] | None) -> dict[str, Any]:
    payload = ReportRequest(template=template or "health_report", sections=sections or ["quality", "kpis", "trends", "limitations"])
    job = create_job(
        job_type="report",
        dataset_id=dataset_id,
        payload={"template": payload.template, "sections": payload.sections},
    )
    try:
        from backend.tasks import generate_report_task

        generate_report_task.delay(job["job_id"], dataset_id, payload.template, payload.sections)
        _append_audit(dataset_id, "report_job_queued", actor, {"job_id": job["job_id"], "template": payload.template})
    except Exception as exc:
        try:
            generate_report_task(job["job_id"], dataset_id, payload.template, payload.sections)
        except Exception as inner_exc:
            update_job(job["job_id"], status="failed", error=str(inner_exc))
            raise HTTPException(status_code=503, detail=f"Failed to run report job: {exc}; {inner_exc}") from inner_exc
    return {"dataset_id": dataset_id, "job_id": job["job_id"], "status": job["status"]}


def _workflow_action_index(actions: list[dict[str, Any]], action_id: str) -> int:
    for index, action in enumerate(actions):
        if str(action.get("action_id")) == action_id:
            return index
    raise HTTPException(status_code=404, detail="Workflow action not found.")


def _anomaly_priority(item: dict[str, Any]) -> tuple[int, float]:
    severity_rank = {"high": 3, "medium": 2, "low": 1}
    return (severity_rank.get(str(item.get("severity", "low")), 0), float(item.get("score") or 0.0))


def _top_period_contributor(
    df: pd.DataFrame,
    parsed_periods: pd.Series,
    period_label: str,
    metric: str,
    dimension_candidates: list[str],
) -> dict[str, Any] | None:
    metric_series = pd.to_numeric(df[metric], errors="coerce")
    period_mask = parsed_periods.dt.to_period("M").astype("string") == period_label

    for dimension in dimension_candidates:
        contributor_df = pd.DataFrame({dimension: df[dimension], metric: metric_series})[period_mask].dropna()
        if contributor_df.empty:
            continue
        grouped = contributor_df.groupby(dimension)[metric].sum().sort_values(ascending=False)
        if grouped.empty:
            continue
        top_segment = str(grouped.index[0])
        top_value = float(grouped.iloc[0])
        total_value = float(grouped.sum())
        share = round((top_value / total_value) * 100, 1) if total_value else 0.0
        return {
            "dimension": dimension,
            "segment": top_segment,
            "value": round(top_value, 2),
            "share_percent": share,
        }
    return None


def _build_quality_anomalies(profile: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for column in profile.get("columns", []):
        if not isinstance(column, dict):
            continue
        if column.get("is_pii_candidate") or column.get("is_id_like"):
            continue

        name = str(column.get("name") or "column")
        missing_percent = float(column.get("missing_percent") or 0.0)
        outlier_percent = float(column.get("outlier_percent") or 0.0)
        outlier_count = int(column.get("outlier_count") or 0)

        if missing_percent >= 25.0:
            severity = "high" if missing_percent >= 45.0 else "medium"
            anomalies.append(
                {
                    "anomaly_id": f"quality_missing_{name}",
                    "kind": "quality",
                    "severity": severity,
                    "score": missing_percent,
                    "title": f"High missingness in {name}",
                    "summary": f"{name} is missing in {missing_percent:.1f}% of rows, which can distort downstream analysis.",
                    "metric": name,
                    "evidence": [
                        f"{missing_percent:.1f}% of records are missing this field.",
                        "The governed profile flagged this column as materially incomplete.",
                    ],
                    "root_cause_hints": [
                        "Check whether this field is optional in the source workflow.",
                        "Inspect whether one facility, payer, or period contributes most of the missing values.",
                    ],
                    "recommended_question": f"Which segments contribute most to missing values in {name}?",
                }
            )

        if outlier_count >= 5 and outlier_percent >= 5.0:
            severity = "high" if outlier_percent >= 10.0 else "medium"
            anomalies.append(
                {
                    "anomaly_id": f"quality_outliers_{name}",
                    "kind": "distribution",
                    "severity": severity,
                    "score": outlier_percent,
                    "title": f"Outlier-heavy distribution in {name}",
                    "summary": f"{name} contains {outlier_count} outlier rows ({outlier_percent:.1f}% of the dataset).",
                    "metric": name,
                    "evidence": [
                        f"{outlier_count} rows were flagged as IQR outliers during profiling.",
                        "The governed profile identified an unusually wide spread for this metric.",
                    ],
                    "root_cause_hints": [
                        "Validate whether these values are legitimate operational spikes or entry errors.",
                        "Cut the metric by district, facility, or service line to isolate concentration points.",
                    ],
                    "recommended_question": f"Which segments are driving the outlier pattern in {name}?",
                }
            )
    return anomalies


def _build_segment_anomalies(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    dimension_candidates = _preferred_dimension_candidates(profile)[:3]
    metric_candidates = _numeric_metric_candidates(profile)[:3]

    for metric in metric_candidates:
        metric_series = pd.to_numeric(df[metric], errors="coerce")
        if int(metric_series.notna().sum()) < 12:
            continue

        for dimension in dimension_candidates:
            grouped_df = pd.DataFrame({dimension: df[dimension], metric: metric_series}).dropna()
            if grouped_df.empty:
                continue

            group_sizes = grouped_df.groupby(dimension).size()
            grouped_metric = grouped_df.groupby(dimension)[metric].mean()
            grouped_metric = grouped_metric[group_sizes >= 3]
            if len(grouped_metric) < 3:
                continue

            baseline = float(grouped_metric.mean())
            spread = float(grouped_metric.std(ddof=0) or 0.0)
            if baseline == 0.0 or spread == 0.0:
                continue

            strongest: dict[str, Any] | None = None
            for segment, value in grouped_metric.items():
                ratio = float(value) / baseline if baseline else 0.0
                magnitude = max(ratio, 1 / ratio) if ratio > 0 else 0.0
                if magnitude < 1.4:
                    continue
                z_score = abs((float(value) - baseline) / spread)
                if z_score < 1.5:
                    continue
                candidate = {
                    "segment": str(segment),
                    "value": float(value),
                    "ratio": ratio,
                    "z_score": z_score,
                    "row_count": int(group_sizes.get(segment, 0)),
                }
                if strongest is None or candidate["z_score"] > strongest["z_score"]:
                    strongest = candidate

            if strongest is None:
                continue

            severity = "high" if strongest["z_score"] >= 2.4 else "medium"
            ratio_label = "above" if strongest["ratio"] >= 1 else "below"
            anomalies.append(
                {
                    "anomaly_id": f"segment_{dimension}_{metric}_{strongest['segment']}",
                    "kind": "segment_outlier",
                    "severity": severity,
                    "score": round(float(strongest["z_score"]), 2),
                    "title": f"{dimension} outlier in {metric}",
                    "summary": (
                        f"{strongest['segment']} is {abs(strongest['ratio'] - 1) * 100:.1f}% {ratio_label} "
                        f"the grouped baseline for {metric}."
                    ),
                    "metric": metric,
                    "dimension": dimension,
                    "segment": strongest["segment"],
                    "evidence": [
                        f"{strongest['segment']} average {metric} is {strongest['value']:.2f}.",
                        f"Grouped baseline average is {baseline:.2f} across {len(grouped_metric)} segments.",
                        f"{strongest['row_count']} row(s) contributed to this segment estimate.",
                    ],
                    "root_cause_hints": [
                        f"Inspect {strongest['segment']} over time to see whether this is persistent or recent.",
                        f"Compare {strongest['segment']} against peer {dimension} groups using the same metric definition.",
                    ],
                    "recommended_question": f"Why is {strongest['segment']} an outlier for {metric} within {dimension}?",
                }
            )

    return anomalies


def _build_time_anomalies(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    time_candidates = _time_candidates(profile)[:2]
    metric_candidates = _numeric_metric_candidates(profile)[:2]
    dimension_candidates = _preferred_dimension_candidates(profile)

    for time_col in time_candidates:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce")
        if int(parsed_time.notna().sum()) < 6:
            continue

        for metric in metric_candidates:
            metric_series = pd.to_numeric(df[metric], errors="coerce")
            time_df = pd.DataFrame({"period": parsed_time.dt.to_period("M").astype("string"), metric: metric_series}).dropna()
            grouped = time_df.groupby("period")[metric].sum().sort_index()
            if len(grouped) < 4:
                continue

            baseline = float(grouped.median())
            if baseline <= 0:
                continue

            strongest: dict[str, Any] | None = None
            for period_label, value in grouped.items():
                current_value = float(value)
                ratio = current_value / baseline if baseline else 0.0
                magnitude = max(ratio, 1 / ratio) if ratio > 0 else 0.0
                if magnitude < 1.5:
                    continue
                candidate = {
                    "period": str(period_label),
                    "value": current_value,
                    "ratio": ratio,
                    "magnitude": magnitude,
                }
                if strongest is None or candidate["magnitude"] > strongest["magnitude"]:
                    strongest = candidate

            if strongest is None:
                continue

            contributor = _top_period_contributor(df, parsed_time, strongest["period"], metric, dimension_candidates)
            direction = "spike" if strongest["ratio"] >= 1 else "dip"
            severity = "high" if strongest["magnitude"] >= 2.0 else "medium"
            hints = [
                f"Validate whether source activity truly changed in {strongest['period']} or whether data capture shifted.",
            ]
            if contributor:
                hints.append(
                    f"{contributor['segment']} contributed the largest share within {contributor['dimension']} "
                    f"({contributor['share_percent']:.1f}% of the period total)."
                )

            anomalies.append(
                {
                    "anomaly_id": f"time_{time_col}_{metric}_{strongest['period']}",
                    "kind": "time_spike",
                    "severity": severity,
                    "score": round(float(strongest["magnitude"]), 2),
                    "title": f"{metric} {direction} in {strongest['period']}",
                    "summary": (
                        f"{metric} shows a {direction} in {strongest['period']}, "
                        f"reaching {strongest['value']:.2f} against a median baseline of {baseline:.2f}."
                    ),
                    "metric": metric,
                    "period": strongest["period"],
                    "evidence": [
                        f"The period total is {strongest['value']:.2f}.",
                        f"Median period baseline is {baseline:.2f}.",
                        f"Detection used {time_col} aggregated to calendar month.",
                    ],
                    "root_cause_hints": hints,
                    "recommended_question": f"What changed in {strongest['period']} that caused the {metric} {direction}?",
                }
            )

    return anomalies


def _build_anomaly_analysis(df: pd.DataFrame, profile: dict[str, Any], limit: int = 6) -> dict[str, Any]:
    anomalies = [
        *_build_quality_anomalies(profile),
        *_build_segment_anomalies(df, profile),
        *_build_time_anomalies(df, profile),
    ]
    anomalies.sort(key=_anomaly_priority, reverse=True)
    trimmed = anomalies[:limit]

    suggested_questions = [
        item.get("recommended_question")
        for item in trimmed
        if isinstance(item.get("recommended_question"), str) and item.get("recommended_question")
    ]
    if not suggested_questions:
        suggested_questions = [
            "Which segment is contributing most to the strongest anomaly?",
            "Is the anomaly concentrated in one period or persistent over time?",
        ]

    if trimmed:
        summary = (
            f"Detected {len(trimmed)} governed anomaly signal(s) across quality, segment, and time-based scans. "
            "Use the suggested questions to investigate likely drivers before making decisions."
        )
    else:
        summary = (
            "No high-signal anomalies were detected from governed quality, segment, and time-based scans. "
            "You can still ask follow-up questions to inspect specific metrics or periods."
        )

    return {
        "generated_at": _utc_now_iso(),
        "anomaly_count": len(trimmed),
        "summary": summary,
        "anomalies": trimmed,
        "suggested_questions": suggested_questions[:6],
    }


def _forecast_last_value(history: list[float], horizon: int) -> list[float]:
    baseline = history[-1] if history else 0.0
    return [round(float(baseline), 4) for _ in range(horizon)]


def _forecast_moving_average(history: list[float], horizon: int, window: int = 3) -> list[float]:
    rolling = [float(value) for value in history]
    predictions: list[float] = []
    for _ in range(horizon):
        slice_width = min(window, len(rolling))
        baseline = sum(rolling[-slice_width:]) / max(slice_width, 1) if slice_width else 0.0
        baseline = round(float(baseline), 4)
        predictions.append(baseline)
        rolling.append(baseline)
    return predictions


def _forecast_linear_trend(history: list[float], horizon: int) -> list[float]:
    if not history:
        return [0.0 for _ in range(horizon)]
    if len(history) == 1:
        return [round(float(history[0]), 4) for _ in range(horizon)]

    x_values = list(range(len(history)))
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(history) / len(history)
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, history, strict=True)) / denominator if denominator else 0.0
    intercept = y_mean - (slope * x_mean)
    return [round(float(max(intercept + (slope * (len(history) + step)), 0.0)), 4) for step in range(horizon)]


def _forecast_error_metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted, strict=True)]
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted, strict=True)]
    percentage_errors = [abs((a - p) / a) * 100 for a, p in zip(actual, predicted, strict=True) if a != 0]
    mae = round(sum(absolute_errors) / len(absolute_errors), 4) if absolute_errors else 0.0
    rmse = round((sum(squared_errors) / len(squared_errors)) ** 0.5, 4) if squared_errors else 0.0
    mape = round(sum(percentage_errors) / len(percentage_errors), 2) if percentage_errors else None
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _aggregate_forecast_series(
    df: pd.DataFrame,
    *,
    time_field: str,
    metric_field: str,
    aggregation: str,
) -> pd.Series:
    if aggregation not in {"sum", "mean"}:
        raise HTTPException(status_code=400, detail="Aggregation must be one of: sum, mean.")
    if time_field not in df.columns:
        raise HTTPException(status_code=400, detail=f"Time field {time_field} was not found in the dataset.")
    if metric_field not in df.columns:
        raise HTTPException(status_code=400, detail=f"Metric field {metric_field} was not found in the dataset.")

    scoped = df[[time_field, metric_field]].copy()
    scoped[time_field] = pd.to_datetime(scoped[time_field], errors="coerce")
    scoped[metric_field] = pd.to_numeric(scoped[metric_field], errors="coerce")
    scoped = scoped.dropna()

    if scoped.empty:
        raise HTTPException(status_code=400, detail="The selected forecast fields did not produce any valid time-series rows.")

    grouped_source = scoped.assign(period=scoped[time_field].dt.to_period("M")).groupby("period")[metric_field]
    grouped = grouped_source.mean().sort_index() if aggregation == "mean" else grouped_source.sum().sort_index()
    grouped = grouped.dropna()

    if len(grouped) < 6:
        raise HTTPException(status_code=400, detail="Forecast training requires at least 6 monthly periods with valid values.")

    return grouped


def _build_forecast_run(
    df: pd.DataFrame,
    profile: dict[str, Any],
    request: ForecastTrainRequest,
    *,
    dataset_hash: str,
) -> dict[str, Any]:
    time_candidates = _time_candidates(profile)
    metric_candidates = _numeric_metric_candidates(profile)

    time_field = request.time_field.strip() if request.time_field and request.time_field.strip() else (time_candidates[0] if time_candidates else "")
    metric_field = request.metric_field.strip() if request.metric_field and request.metric_field.strip() else (metric_candidates[0] if metric_candidates else "")

    if not time_field:
        raise HTTPException(status_code=400, detail="No governed time field is available for forecasting.")
    if not metric_field:
        raise HTTPException(status_code=400, detail="No governed numeric metric is available for forecasting.")

    grouped = _aggregate_forecast_series(
        df,
        time_field=time_field,
        metric_field=metric_field,
        aggregation=request.aggregation.strip().lower() or "sum",
    )

    values = [round(float(value), 4) for value in grouped.tolist()]
    holdout_points = min(max(int(request.horizon), 2), max(2, len(values) // 3))
    if len(values) - holdout_points < 4:
        raise HTTPException(status_code=400, detail="Forecast training needs more historical periods before a holdout evaluation can run.")

    train_values = values[:-holdout_points]
    actual_values = values[-holdout_points:]
    candidate_builders = {
        "last_value": lambda history, horizon: _forecast_last_value(history, horizon),
        "moving_average_3": lambda history, horizon: _forecast_moving_average(history, horizon, window=3),
        "linear_trend": lambda history, horizon: _forecast_linear_trend(history, horizon),
    }

    candidate_models: list[dict[str, Any]] = []
    for model_name, builder in candidate_builders.items():
        holdout_predictions = builder(train_values, holdout_points)
        metrics = _forecast_error_metrics(actual_values, holdout_predictions)
        candidate_models.append(
            {
                "model_name": model_name,
                "holdout_points": holdout_points,
                **metrics,
            }
        )

    candidate_models.sort(key=lambda item: (float(item["mae"]), float(item["rmse"])))
    champion = candidate_models[0]
    champion_model = str(champion["model_name"])
    future_values = candidate_builders[champion_model](values, int(request.horizon))

    period_index = grouped.index
    warnings: list[str] = []
    if len(grouped) < 12:
        warnings.append("Forecast uses fewer than 12 monthly periods, so long-horizon confidence is limited.")
    expected_index = pd.period_range(start=period_index[0], end=period_index[-1], freq="M")
    if len(expected_index) != len(period_index):
        warnings.append("Missing monthly periods were detected; the forecast treats gaps as missing history rather than zero activity.")

    latest_actual = values[-1]
    latest_forecast = future_values[-1]
    if latest_actual == 0:
        direction = "flat"
    else:
        delta_ratio = (latest_forecast - latest_actual) / latest_actual
        if delta_ratio > 0.05:
            direction = "upward"
        elif delta_ratio < -0.05:
            direction = "downward"
        else:
            direction = "stable"

    summary = (
        f"Champion model {champion_model} forecasted {metric_field} with {champion['mae']:.2f} MAE "
        f"across {holdout_points} holdout month(s). The governed projection for the next {request.horizon} month(s) is {direction}."
    )
    baseline_mean = round(sum(values) / len(values), 4) if values else 0.0
    baseline_std = round(float(pd.Series(values).std(ddof=0) or 0.0), 4) if values else 0.0

    latest_period = period_index[-1]
    forecast_points = [
        {"period": str(latest_period + step), "value": round(float(value), 4)}
        for step, value in enumerate(future_values, start=1)
    ]
    historical_points = [{"period": str(period), "value": round(float(value), 4)} for period, value in grouped.tail(18).items()]

    return {
        "generated_at": _utc_now_iso(),
        "name": request.name.strip() if request.name and request.name.strip() else f"{metric_field} monthly forecast",
        "time_field": time_field,
        "metric_field": metric_field,
        "aggregation": request.aggregation.strip().lower() or "sum",
        "periods_used": len(grouped),
        "holdout_points": holdout_points,
        "horizon": int(request.horizon),
        "champion_model": champion_model,
        "training_data_hash": dataset_hash,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "latest_actual": round(float(values[-1]), 4) if values else None,
        "summary": summary,
        "warnings": warnings,
        "candidate_models": candidate_models,
        "historical": historical_points,
        "forecast": forecast_points,
    }


def _build_forecast_drift(
    dataset_id: str,
    meta: dict[str, Any],
    df: pd.DataFrame,
    run: dict[str, Any],
    *,
    window: int,
) -> dict[str, Any]:
    payload = run.get("payload") if isinstance(run, dict) else {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="The selected forecast run is malformed.")

    time_field = str(payload.get("time_field") or "").strip()
    metric_field = str(payload.get("metric_field") or "").strip()
    aggregation = str(payload.get("aggregation") or "sum").strip().lower() or "sum"
    if not time_field or not metric_field:
        raise HTTPException(status_code=400, detail="The selected forecast run is missing its governed field configuration.")

    grouped = _aggregate_forecast_series(df, time_field=time_field, metric_field=metric_field, aggregation=aggregation)
    recent_window = max(3, min(window, len(grouped)))
    recent_values = [float(value) for value in grouped.tail(recent_window).tolist()]

    historical_points = payload.get("historical") if isinstance(payload.get("historical"), list) else []
    baseline_values = [
        float(item.get("value"))
        for item in historical_points
        if isinstance(item, dict) and item.get("value") is not None
    ]
    if not baseline_values:
        raise HTTPException(status_code=400, detail="The selected forecast run does not include historical baseline values for drift analysis.")

    baseline_window = baseline_values[-recent_window:]
    baseline_mean = float(payload.get("baseline_mean")) if payload.get("baseline_mean") is not None else float(sum(baseline_window) / len(baseline_window))
    baseline_std = float(payload.get("baseline_std")) if payload.get("baseline_std") is not None else float(pd.Series(baseline_window).std(ddof=0) or 0.0)
    recent_mean = float(sum(recent_values) / len(recent_values))
    recent_std = float(pd.Series(recent_values).std(ddof=0) or 0.0)

    mean_shift = abs(recent_mean - baseline_mean) / abs(baseline_mean) if baseline_mean else 0.0
    volatility_base = baseline_std if baseline_std > 0 else max(recent_std, 1.0)
    volatility_shift = abs(recent_std - baseline_std) / volatility_base if volatility_base else 0.0

    current_hash = str(meta.get("file_hash") or _dataset_signature(meta) or "")
    training_hash = str(payload.get("training_data_hash") or "") or None
    stale_model = bool(training_hash and current_hash and training_hash != current_hash)

    signals: list[dict[str, Any]] = []
    if stale_model:
        signals.append(
            {
                "code": "data_refresh",
                "severity": "medium",
                "message": "The session file hash changed after this model run was trained, so the forecast should be revalidated.",
            }
        )
    if mean_shift >= 0.2:
        signals.append(
            {
                "code": "mean_shift",
                "severity": "high" if mean_shift >= 0.35 else "medium",
                "message": f"Recent monthly {metric_field} mean shifted by {mean_shift * 100:.1f}% versus the training baseline.",
            }
        )
    if volatility_shift >= 0.2:
        signals.append(
            {
                "code": "volatility_shift",
                "severity": "high" if volatility_shift >= 0.35 else "medium",
                "message": f"Recent monthly {metric_field} volatility shifted by {volatility_shift * 100:.1f}% versus the training baseline.",
            }
        )

    drift_score = round(max(mean_shift, volatility_shift, 0.25 if stale_model else 0.0), 4)
    if drift_score >= 0.35:
        summary = (
            f"Forecast drift is high for {metric_field}. Recent data moved materially away from the baseline used by "
            f"{payload.get('champion_model', 'the champion model')}."
        )
    elif drift_score >= 0.2:
        summary = (
            f"Forecast drift is moderate for {metric_field}. Review the recent series before trusting existing projections."
        )
    else:
        summary = f"Forecast drift is currently low for {metric_field}; the recent series is still close to the training baseline."

    recommended_actions = (
        [
            "Retrain the forecast run on the refreshed dataset.",
            "Compare candidate models again before promoting a new champion.",
            "Review the latest monthly periods and any anomaly signals together.",
        ]
        if drift_score >= 0.2 or stale_model
        else [
            "Continue monitoring this model after the next data refresh.",
            "Re-run drift analysis if operational conditions or seasonality change.",
        ]
    )

    return {
        "generated_at": _utc_now_iso(),
        "run_id": str(run.get("run_id") or ""),
        "run_name": str(payload.get("name") or run.get("run_id") or "Forecast run"),
        "champion_model": str(payload.get("champion_model") or "unknown"),
        "time_field": time_field,
        "metric_field": metric_field,
        "aggregation": aggregation,
        "training_data_hash": training_hash,
        "current_data_hash": current_hash or None,
        "stale_model": stale_model,
        "drift_score": drift_score,
        "periods_analyzed": recent_window,
        "baseline_mean": round(baseline_mean, 4),
        "recent_mean": round(recent_mean, 4),
        "baseline_std": round(baseline_std, 4),
        "recent_std": round(recent_std, 4),
        "summary": summary,
        "signals": signals,
        "recommended_actions": recommended_actions,
    }


def _champion_candidate_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    champion_model = str(payload.get("champion_model") or "")
    candidates = payload.get("candidate_models") if isinstance(payload.get("candidate_models"), list) else []
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("model_name") or "") == champion_model:
            return candidate
    return {}


def _build_evaluation_run_summary(run: dict[str, Any], drift: dict[str, Any], *, source: str) -> dict[str, Any]:
    payload = run.get("payload") if isinstance(run, dict) else {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="The selected model run payload is malformed.")
    metrics = _champion_candidate_metrics(payload)
    return {
        "run_id": str(run.get("run_id") or ""),
        "name": str(payload.get("name") or run.get("run_id") or "Forecast run"),
        "champion_model": str(payload.get("champion_model") or "unknown"),
        "metric_field": str(payload.get("metric_field") or ""),
        "mae": round(float(metrics.get("mae") or 0.0), 4),
        "rmse": round(float(metrics.get("rmse") or 0.0), 4),
        "mape": round(float(metrics.get("mape")), 2) if metrics.get("mape") is not None else None,
        "drift_score": round(float(drift.get("drift_score") or 0.0), 4),
        "stale_model": bool(drift.get("stale_model")),
        "source": source,
    }


def _build_model_evaluation(
    dataset_id: str,
    meta: dict[str, Any],
    df: pd.DataFrame,
    active_run: dict[str, Any],
    challenger_run: dict[str, Any],
) -> dict[str, Any]:
    active_drift = _build_forecast_drift(dataset_id, meta, df, active_run, window=6)
    challenger_drift = _build_forecast_drift(dataset_id, meta, df, challenger_run, window=6)

    active_summary = _build_evaluation_run_summary(active_run, active_drift, source="registry")
    challenger_summary = _build_evaluation_run_summary(challenger_run, challenger_drift, source="candidate")

    rationale: list[str] = []
    if active_summary["stale_model"]:
        rationale.append("The active registry model is stale against the current dataset hash.")
    if challenger_summary["stale_model"]:
        rationale.append("The challenger run is also stale and should not be promoted without retraining.")

    mae_improvement = 0.0
    if active_summary["mae"] > 0:
        mae_improvement = (active_summary["mae"] - challenger_summary["mae"]) / active_summary["mae"]

    if challenger_summary["mae"] < active_summary["mae"]:
        rationale.append(
            f"The challenger has lower holdout MAE ({challenger_summary['mae']:.2f}) than the active model ({active_summary['mae']:.2f})."
        )
    else:
        rationale.append(
            f"The active model keeps the stronger holdout MAE ({active_summary['mae']:.2f}) compared with the challenger ({challenger_summary['mae']:.2f})."
        )

    if challenger_summary["drift_score"] < active_summary["drift_score"]:
        rationale.append(
            f"The challenger shows lower current-data drift ({challenger_summary['drift_score']:.2f}) than the active model ({active_summary['drift_score']:.2f})."
        )
    elif challenger_summary["drift_score"] > active_summary["drift_score"]:
        rationale.append(
            f"The active model is more stable on current data ({active_summary['drift_score']:.2f}) than the challenger ({challenger_summary['drift_score']:.2f})."
        )
    else:
        rationale.append("Both runs show similar current-data drift against the refreshed dataset.")

    promote_challenger = False
    if active_summary["stale_model"] and not challenger_summary["stale_model"]:
        promote_challenger = challenger_summary["mae"] <= active_summary["mae"] * 1.05
    elif not challenger_summary["stale_model"] and mae_improvement >= 0.05 and challenger_summary["drift_score"] <= active_summary["drift_score"] + 0.05:
        promote_challenger = True

    if promote_challenger:
        recommendation = (
            f"Promote challenger run {challenger_summary['run_id']} after reviewer approval. "
            "It outperforms or safely replaces the active model under current governed checks."
        )
        winner = "challenger"
        suggested_actions = [
            "Promote the challenger run to the governed registry.",
            "Record a reviewer note explaining the promotion rationale.",
            "Re-run drift monitoring after the next data refresh.",
        ]
    else:
        recommendation = (
            f"Keep active run {active_summary['run_id']} as the governed model for now. "
            "The challenger does not yet justify promotion under current holdout and drift checks."
        )
        winner = "active"
        suggested_actions = [
            "Keep the active model in the registry.",
            "Retrain a fresh challenger if the active model becomes stale or drift worsens.",
            "Review anomaly and drift signals before the next promotion decision.",
        ]

    return {
        "generated_at": _utc_now_iso(),
        "active_run": active_summary,
        "challenger_run": challenger_summary,
        "recommendation": recommendation,
        "winner": winner,
        "rationale": rationale,
        "suggested_actions": suggested_actions,
    }


def _pii_aliases(profile: dict[str, Any]) -> dict[str, str]:
    pii_columns = profile.get("pii_candidates", []) if profile else []
    return {column: f"pii_field_{index + 1}" for index, column in enumerate(pii_columns)}


def _build_quality_issues(profile: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    missing_percent = profile.get("missing_percent", {})
    high_missing = [column for column, value in missing_percent.items() if float(value) >= 20.0]
    if high_missing:
        issues.append(
            {
                "code": "HIGH_MISSINGNESS",
                "severity": "high",
                "message": "Columns exceed 20% missing values.",
                "columns": high_missing[:20],
            }
        )

    duplicate_pct = float(profile.get("duplicate_percent", 0.0))
    if duplicate_pct > 2.0:
        issues.append(
            {
                "code": "HIGH_DUPLICATES",
                "severity": "medium",
                "message": "Duplicate row percentage is above recommended threshold.",
                "columns": [],
            }
        )

    pii_candidates = profile.get("pii_candidates", [])
    if pii_candidates:
        issues.append(
            {
                "code": "PII_CANDIDATES",
                "severity": "high",
                "message": "Potential PII columns detected; masking should be applied on exports.",
                "columns": pii_candidates[:20],
            }
        )
    return issues


def _build_facts_bundle(
    df: pd.DataFrame | None,
    profile: dict[str, Any],
    dataset_id: str = "",
    dataset_hash: str = "",
) -> dict[str, Any]:
    if df is None:
        df = pd.DataFrame()

    rows = int(profile.get("shape", {}).get("rows", len(df)))
    cols = int(profile.get("shape", {}).get("cols", len(df.columns)))
    quality_score = float(profile.get("quality_score", 0.0))
    duplicate_rows = int(profile.get("duplicate_rows", 0))
    data_coverage = profile.get("data_coverage") or {
        "mode": "full",
        "rows_total": rows,
        "rows_used": rows,
        "sampling_method": "uniform",
        "seed": None,
        "bias_notes": "Full dataset used.",
    }

    insight_facts: list[dict[str, Any]] = []

    def add_fact(kind: str, value: dict[str, Any], evidence: dict[str, Any]) -> str:
        fact_id = f"fact_{len(insight_facts) + 1:03d}"
        insight_facts.append({"id": fact_id, "type": kind, "value": value, "evidence": evidence})
        return fact_id

    fact_rows = add_fact(
        "comparison",
        {"metric": "row_count", "value": rows},
        {"source": "profiling.shape.rows"},
    )
    fact_cols = add_fact(
        "comparison",
        {"metric": "column_count", "value": cols},
        {"source": "profiling.shape.cols"},
    )
    fact_quality = add_fact(
        "comparison",
        {"metric": "quality_score", "value": quality_score},
        {"source": "quality.score"},
    )
    fact_duplicates = add_fact(
        "comparison",
        {"metric": "duplicate_rows", "value": duplicate_rows},
        {"source": "profiling.duplicate_rows"},
    )

    missing_percent = profile.get("missing_percent", {})
    top_missing = sorted(
        ((str(column), float(value)) for column, value in missing_percent.items() if float(value) > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    if top_missing:
        add_fact(
            "distribution",
            {"metric": "top_missing_columns", "columns": top_missing},
            {"source": "profiling.missing_percent"},
        )

    numeric_cols = [str(value) for value in profile.get("numeric_cols", [])]
    categorical_cols = [str(value) for value in profile.get("categorical_cols", [])]
    datetime_cols = [str(value) for value in profile.get("datetime_cols", [])]

    if datetime_cols and numeric_cols and all(col in df.columns for col in (datetime_cols[0], numeric_cols[0])):
        dt_column = datetime_cols[0]
        value_column = numeric_cols[0]
        scoped = df[[dt_column, value_column]].copy()
        scoped[dt_column] = pd.to_datetime(scoped[dt_column], errors="coerce")
        scoped[value_column] = pd.to_numeric(scoped[value_column], errors="coerce")
        scoped = scoped.dropna()
        if len(scoped) >= 3:
            monthly = (
                scoped.set_index(dt_column)[value_column]
                .resample("MS")
                .sum()
                .reset_index()
                .sort_values(dt_column)
            )
            if len(monthly) >= 2:
                latest = float(monthly.iloc[-1][value_column])
                previous = float(monthly.iloc[-2][value_column])
                delta_pct = ((latest - previous) / previous * 100.0) if previous else None
                add_fact(
                    "trend",
                    {
                        "metric": value_column,
                        "latest_period_value": round(latest, 4),
                        "previous_period_value": round(previous, 4),
                        "pct_change": round(float(delta_pct), 2) if delta_pct is not None else None,
                    },
                    {"source": f"trend.{value_column}", "time_field": dt_column, "grain": "month"},
                )

    if categorical_cols and numeric_cols and all(col in df.columns for col in (categorical_cols[0], numeric_cols[0])):
        cat_col = categorical_cols[0]
        val_col = numeric_cols[0]
        scoped = df[[cat_col, val_col]].copy()
        scoped[val_col] = pd.to_numeric(scoped[val_col], errors="coerce")
        scoped = scoped.dropna()
        if not scoped.empty:
            grouped = (
                scoped.groupby(cat_col, dropna=False)[val_col]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            if not grouped.empty:
                top_name = str(grouped.index[0])
                top_value = float(grouped.iloc[0])
                add_fact(
                    "comparison",
                    {"metric": f"top_{val_col}_segment", "segment_field": cat_col, "segment": top_name, "value": round(top_value, 4)},
                    {"source": f"grouped_mean.{cat_col}.{val_col}"},
                )

    if len(numeric_cols) >= 2 and all(col in df.columns for col in numeric_cols[:2]):
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        corr_df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(corr_df) >= 4:
            corr = float(corr_df[x_col].corr(corr_df[y_col]))
            if pd.notna(corr):
                add_fact(
                    "comparison",
                    {"metric": "pair_correlation", "x": x_col, "y": y_col, "value": round(corr, 4)},
                    {"source": "correlation.pearson"},
                )

    kpis = [
        {"id": "kpi_rows", "name": "Rows", "value": rows, "unit": "rows", "facts_refs": [fact_rows]},
        {"id": "kpi_cols", "name": "Columns", "value": cols, "unit": "cols", "facts_refs": [fact_cols]},
        {"id": "kpi_quality", "name": "Quality Score", "value": round(quality_score, 2), "unit": "score", "facts_refs": [fact_quality]},
        {"id": "kpi_duplicates", "name": "Duplicate Rows", "value": duplicate_rows, "unit": "rows", "facts_refs": [fact_duplicates]},
    ]

    chart_candidates: list[dict[str, Any]] = []
    if datetime_cols and numeric_cols:
        chart_candidates.append(
            {
                "id": "chartcand_001",
                "chart_type": "line",
                "x": datetime_cols[0],
                "y": numeric_cols[0],
                "group_by": None,
                "filters": [],
                "score": 0.96,
            }
        )
    if categorical_cols and numeric_cols:
        chart_candidates.append(
            {
                "id": "chartcand_002",
                "chart_type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "group_by": None,
                "filters": [],
                "score": 0.9,
            }
        )
    if numeric_cols:
        chart_candidates.append(
            {
                "id": "chartcand_003",
                "chart_type": "hist",
                "x": numeric_cols[0],
                "y": None,
                "group_by": None,
                "filters": [],
                "score": 0.85,
            }
        )

    quality_issues = _build_quality_issues(profile)
    bundle = {
        "dataset_id": dataset_id or "unknown_dataset",
        "dataset_hash": dataset_hash or "unknown_hash",
        "created_at": _utc_now_iso(),
        "data_coverage": data_coverage,
        "profiling": {
            "shape": profile.get("shape", {"rows": rows, "cols": cols}),
            "dtypes": profile.get("dtypes", {}),
            "missing_percent": profile.get("missing_percent", {}),
            "pii_candidates": profile.get("pii_candidates", []),
        },
        "quality": {
            "score": round(quality_score, 2),
            "issues": quality_issues,
        },
        "kpis": kpis,
        "insight_facts": insight_facts,
        "chart_candidates": chart_candidates,
    }
    validate_schema(bundle, FACTS_BUNDLE_SCHEMA)
    return bundle


def _is_facts_bundle_compatible(bundle: dict[str, Any]) -> bool:
    required = {
        "dataset_id",
        "dataset_hash",
        "created_at",
        "data_coverage",
        "profiling",
        "quality",
        "kpis",
        "insight_facts",
        "chart_candidates",
    }
    if not required.issubset(set(bundle.keys())):
        return False
    if not isinstance(bundle.get("kpis"), list):
        return False
    if not isinstance(bundle.get("insight_facts"), list):
        return False
    if not isinstance(bundle.get("chart_candidates"), list):
        return False
    return True


def _mask_recursive(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, str):
        return aliases.get(value, value)
    if isinstance(value, list):
        return [_mask_recursive(item, aliases) for item in value]
    if isinstance(value, tuple):
        return [_mask_recursive(item, aliases) for item in value]
    if isinstance(value, dict):
        return {key: _mask_recursive(inner_value, aliases) for key, inner_value in value.items()}
    return value


def _apply_cleaning_actions(df: pd.DataFrame, actions: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for action in actions:
        if action == "drop_empty_cols":
            cleaned = cleaned.dropna(axis=1, how="all")
        elif action == "drop_empty_rows":
            cleaned = cleaned.dropna(axis=0, how="all")
        elif action == "normalize_dates":
            for column in cleaned.columns:
                if any(key in str(column).lower() for key in TIME_KEYWORDS):
                    cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
        elif action == "trim_strings":
            for column in cleaned.select_dtypes(include=["object", "string"]).columns:
                cleaned[column] = cleaned[column].astype(str).str.strip()
        elif action == "dedupe_rows":
            cleaned = cleaned.drop_duplicates()
    return cleaned


def _safe_answer_from_facts(question: str, facts_bundle: dict[str, Any]) -> tuple[str, list[str], str, float, str]:
    question_lower = question.lower()
    insight_facts = facts_bundle.get("insight_facts", [])
    data_coverage = str((facts_bundle.get("data_coverage") or {}).get("mode", "sample"))

    trend_fact = next((fact for fact in insight_facts if fact.get("type") == "trend"), None)
    if trend_fact and any(token in question_lower for token in ("trend", "change", "increase", "decrease")):
        payload = trend_fact.get("value", {})
        pct = payload.get("pct_change")
        latest = payload.get("latest_period_value")
        previous = payload.get("previous_period_value")
        if isinstance(latest, (float, int)) and isinstance(previous, (float, int)):
            if pct is not None:
                answer = (
                    f"The most recent period is {float(latest):.2f} vs {float(previous):.2f} "
                    f"({float(pct):.2f}% change)."
                )
            else:
                answer = f"The most recent period is {float(latest):.2f} vs {float(previous):.2f}."
        else:
            answer = "A trend fact exists, but comparable periods were not fully available."
        return answer, [str(trend_fact.get("id", ""))], "Medium", 1.0, data_coverage

    if any(token in question_lower for token in ("missing", "completeness", "quality")):
        quality = facts_bundle.get("quality", {})
        issues = quality.get("issues", [])
        if issues:
            issue = issues[0]
            columns = ", ".join(issue.get("columns", [])[:5]) or "N/A"
            answer = f"Primary quality issue: {issue.get('code')} ({issue.get('severity')}). Columns: {columns}."
            coverage = 1.0
        else:
            answer = f"Quality score is {quality.get('score')} and no major issues were flagged."
            coverage = 0.7
        return answer, [], "Medium", coverage, data_coverage

    kpis = facts_bundle.get("kpis", [])
    if any(token in question_lower for token in ("rows", "columns", "size", "shape")) and kpis:
        rows = next((kpi for kpi in kpis if kpi.get("id") == "kpi_rows"), None)
        cols = next((kpi for kpi in kpis if kpi.get("id") == "kpi_cols"), None)
        if rows and cols:
            answer = f"Dataset has {rows.get('value')} rows and {cols.get('value')} columns."
            refs = list(rows.get("facts_refs", [])) + list(cols.get("facts_refs", []))
            return answer, [str(value) for value in refs if isinstance(value, str)], "High", 1.0, data_coverage

    return ("No matching precomputed fact was found for this question.", [], "Low", 0.0, data_coverage)


def _facts_context_for_llm(facts_bundle: dict[str, Any]) -> dict[str, Any]:
    profiling = facts_bundle.get("profiling", {})
    dtypes = profiling.get("dtypes", {})
    return {
        "kpis": facts_bundle.get("kpis", [])[:12],
        "quality": facts_bundle.get("quality", {}),
        "insight_facts": facts_bundle.get("insight_facts", [])[:25],
        "chart_candidates": facts_bundle.get("chart_candidates", [])[:12],
        "data_coverage": facts_bundle.get("data_coverage", {}),
        "columns": list(dtypes.keys())[:120],
        "dtypes": dtypes,
    }


def _generate_dashboard_spec_llm(dataset_id: str, template: str, facts_bundle: dict[str, Any]) -> dict[str, Any]:
    llm = _get_llm_client()
    context = _facts_context_for_llm(facts_bundle)
    system_prompt = (
        "You generate dashboard specs for healthcare analytics. "
        "You must only cite fact keys from the supplied facts bundle."
    )
    user_prompt = (
        f"Dataset ID: {dataset_id}\n"
        f"Template: {template}\n"
        f"Facts context JSON:\n{json.dumps(context, ensure_ascii=True)}\n"
        "Build a compact dashboard spec for healthcare analytics."
    )
    output = llm.generate_json(DASHBOARD_SPEC_SCHEMA, system_prompt, user_prompt, timeout=30)
    validate_schema(output, DASHBOARD_SPEC_SCHEMA)
    validate_facts_references(output, facts_bundle)
    return output


def _generate_query_plan_llm(
    question: str,
    facts_bundle: dict[str, Any],
    profile: dict[str, Any],
    semantic_layer: dict[str, Any],
) -> dict[str, Any]:
    del facts_bundle  # facts are used later for grounding the narrative, not for plan schema.
    llm = _get_llm_client()
    context = {
        "question": question,
        "columns": [str(name) for name in profile.get("dtypes", {}).keys()][:120],
        "semantic_layer": semantic_prompt_context(semantic_layer),
    }
    system_prompt = (
        "You are a query planner for healthcare analytics. "
        "Generate a safe query plan using only the approved semantic layer. "
        "Prefer metric_id values instead of raw field names whenever possible. "
        "Never use PII-blocked fields."
    )
    user_prompt = f"Build a query plan for this context:\n{json.dumps(context, ensure_ascii=True)}"
    output = llm.generate_json(QUERY_PLAN_SCHEMA, system_prompt, user_prompt, timeout=30)
    validate_schema(output, QUERY_PLAN_SCHEMA)
    return output


def _fallback_query_plan(question: str, profile: dict[str, Any], semantic_layer: dict[str, Any]) -> dict[str, Any]:
    del profile
    metrics = [item for item in semantic_layer.get("metrics", []) if isinstance(item, dict)]
    dimensions = [item for item in semantic_layer.get("dimensions", []) if isinstance(item, dict)]
    metric_id = str(metrics[0]["id"]) if metrics else ""
    group_by: list[str] = []
    chart_hint = "table"
    intent = "aggregate"
    default_time_field = semantic_layer.get("default_time_field")
    preferred_dimension = semantic_layer.get("preferred_dimension")

    if default_time_field and any(token in question.lower() for token in ("trend", "time", "month", "week", "year")):
        group_by = [str(default_time_field)]
        chart_hint = "line"
        intent = "trend"
    elif preferred_dimension:
        group_by = [str(preferred_dimension)]
        chart_hint = "bar"
        intent = "compare"
    elif dimensions:
        group_by = [str(dimensions[0]["field"])]
        chart_hint = "bar"
        intent = "compare"

    return {
        "intent": intent,
        "metrics": [{"metric_id": metric_id}] if metric_id else [],
        "group_by": group_by,
        "filters": [],
        "time": {
            "field": default_time_field,
            "grain": "month" if default_time_field else None,
            "start": None,
            "end": None,
        },
        "limit": 1000,
        "chart_hint": chart_hint,
    }


def _apply_filter_op(df: pd.DataFrame, field: str, op: str, value: Any) -> pd.DataFrame:
    if not field or field not in df.columns:
        return df

    series = df[field]
    if op == "eq":
        return df[series == value]
    if op == "neq":
        return df[series != value]
    if op in {"gt", "gte", "lt", "lte"}:
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if op == "gt":
            return df[numeric_series > numeric_value]
        if op == "gte":
            return df[numeric_series >= numeric_value]
        if op == "lt":
            return df[numeric_series < numeric_value]
        return df[numeric_series <= numeric_value]
    if op == "in" and isinstance(value, list):
        return df[series.isin(value)]
    if op == "between" and isinstance(value, list) and len(value) == 2:
        start, end = value
        if pd.api.types.is_datetime64_any_dtype(series) or any(token in field.lower() for token in TIME_KEYWORDS):
            dt_series = pd.to_datetime(series, errors="coerce")
            start_dt = pd.to_datetime(start, errors="coerce")
            end_dt = pd.to_datetime(end, errors="coerce")
            return df[(dt_series >= start_dt) & (dt_series <= end_dt)]
        numeric_series = pd.to_numeric(series, errors="coerce")
        start_num = pd.to_numeric(pd.Series([start]), errors="coerce").iloc[0]
        end_num = pd.to_numeric(pd.Series([end]), errors="coerce").iloc[0]
        return df[(numeric_series >= start_num) & (numeric_series <= end_num)]
    return df


def _execute_query_plan(df: pd.DataFrame, plan: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    scoped = df.copy()
    for flt in plan.get("filters", []):
        if not isinstance(flt, dict):
            continue
        field = str(flt.get("field", ""))
        op = str(flt.get("op", "eq"))
        scoped = _apply_filter_op(scoped, field, op, flt.get("value"))

    time_filter = plan.get("time") or {}
    time_field = time_filter.get("field")
    if isinstance(time_field, str) and time_field in scoped.columns:
        scoped = scoped.copy()
        scoped[time_field] = pd.to_datetime(scoped[time_field], errors="coerce")
        start = pd.to_datetime(time_filter.get("start"), errors="coerce")
        end = pd.to_datetime(time_filter.get("end"), errors="coerce")
        if pd.notna(start):
            scoped = scoped[scoped[time_field] >= start]
        if pd.notna(end):
            scoped = scoped[scoped[time_field] <= end]

    metrics = [metric for metric in plan.get("metrics", []) if isinstance(metric, dict)]
    if not metrics:
        return scoped.head(min(int(plan.get("limit", 1000)), 1000)), "No metrics in query plan; returning rows."

    group_by = [str(field) for field in plan.get("group_by", []) if isinstance(field, str) and field in scoped.columns]
    limit = min(int(plan.get("limit", 1000)), 10000)

    if group_by:
        grouped = scoped.groupby(group_by, dropna=False)
        frames: list[pd.DataFrame] = []
        for metric in metrics:
            field = str(metric.get("field", ""))
            op = str(metric.get("op", "count"))
            if field not in scoped.columns:
                continue
            alias = f"{op}_{field}"
            if op == "count":
                metric_df = grouped[field].count().reset_index(name=alias)
            else:
                typed = scoped[group_by + [field]].copy()
                typed[field] = pd.to_numeric(typed[field], errors="coerce")
                grouped_num = typed.groupby(group_by, dropna=False)[field]
                if op == "sum":
                    metric_df = grouped_num.sum().reset_index(name=alias)
                elif op == "mean":
                    metric_df = grouped_num.mean().reset_index(name=alias)
                elif op == "min":
                    metric_df = grouped_num.min().reset_index(name=alias)
                else:
                    metric_df = grouped_num.max().reset_index(name=alias)
            frames.append(metric_df)

        if not frames:
            return scoped.head(min(limit, 100)), "No valid metric fields found."

        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(frame, on=group_by, how="outer")
        value_cols = [col for col in result.columns if col not in group_by]
        if value_cols:
            result = result.sort_values(value_cols[0], ascending=False)
        return result.head(limit), "Grouped aggregate computed."

    row: dict[str, Any] = {}
    for metric in metrics:
        field = str(metric.get("field", ""))
        op = str(metric.get("op", "count"))
        if field not in scoped.columns:
            continue
        alias = f"{op}_{field}"
        if op == "count":
            row[alias] = int(scoped[field].count())
            continue
        series = pd.to_numeric(scoped[field], errors="coerce")
        if op == "sum":
            row[alias] = float(series.sum())
        elif op == "mean":
            row[alias] = float(series.mean())
        elif op == "min":
            row[alias] = float(series.min())
        else:
            row[alias] = float(series.max())

    if not row:
        return scoped.head(min(limit, 100)), "No valid metric fields found."
    return pd.DataFrame([row]).head(limit), "Aggregate computed."


def _default_fact_ids(facts_bundle: dict[str, Any], limit: int = 3) -> list[str]:
    return [
        str(fact.get("id"))
        for fact in facts_bundle.get("insight_facts", [])[:limit]
        if isinstance(fact.get("id"), str)
    ]


def _summarize_query_result_llm(
    question: str,
    result_df: pd.DataFrame,
    facts_bundle: dict[str, Any],
    fact_ids: list[str],
) -> tuple[str, list[str]]:
    llm = _get_llm_client()
    preview = json.loads(result_df.head(10).to_json(orient="records", date_format="iso"))
    grounded_ids = [value for value in fact_ids if isinstance(value, str)]
    context = {
        "question": question,
        "result_preview": preview,
        "fact_ids": grounded_ids,
        "data_coverage": facts_bundle.get("data_coverage", {}),
    }
    system_prompt = (
        "You summarize deterministic analysis outputs. "
        "Do not invent numbers. Reference only provided fact IDs."
    )
    user_prompt = f"Summarize this result context:\n{json.dumps(context, ensure_ascii=True)}"
    output = llm.generate_json(ASK_NARRATIVE_SCHEMA, system_prompt, user_prompt, timeout=30)
    validate_schema(output, ASK_NARRATIVE_SCHEMA)
    validate_facts_references(output, facts_bundle)
    answer = output.get("answer", "")
    facts_used = [value for value in output.get("fact_ids", []) if isinstance(value, str)]
    return answer, facts_used


def _build_dashboard_spec(dataset_id: str, profile: dict[str, Any], facts_bundle: dict[str, Any]) -> dict[str, Any]:
    del dataset_id  # Stored with session metadata and artifact path.
    numeric_cols = [str(value) for value in profile.get("numeric_cols", [])]
    categorical_cols = [str(value) for value in profile.get("categorical_cols", [])]
    datetime_cols = [str(value) for value in profile.get("datetime_cols", [])]
    insight_facts = facts_bundle.get("insight_facts", [])

    kpis: list[dict[str, str]] = []
    for item in facts_bundle.get("kpis", [])[:4]:
        refs = [value for value in item.get("facts_refs", []) if isinstance(value, str)]
        if refs:
            kpis.append({"name": str(item.get("name", "KPI")), "fact_id": refs[0]})

    filters: list[dict[str, str]] = []
    if datetime_cols:
        filters.append({"field": datetime_cols[0], "type": "date"})
    for field in categorical_cols[:2]:
        filters.append({"field": field, "type": "categorical"})
    if numeric_cols:
        filters.append({"field": numeric_cols[0], "type": "numeric"})

    charts: list[dict[str, Any]] = []
    for index, candidate in enumerate(facts_bundle.get("chart_candidates", [])[:6]):
        chart_type = str(candidate.get("chart_type", "table"))
        resolved_type = chart_type if chart_type in {"line", "bar", "hist", "scatter", "heatmap", "table"} else "table"
        linked_fact = (
            str(insight_facts[index].get("id"))
            if index < len(insight_facts)
            else (str(insight_facts[0].get("id")) if insight_facts else "")
        )
        charts.append(
            {
                "title": f"Chart {index + 1}",
                "type": resolved_type,
                "x": candidate.get("x"),
                "y": candidate.get("y"),
                "group_by": candidate.get("group_by"),
                "aggregation": "sum" if resolved_type == "line" else ("mean" if resolved_type in {"bar", "scatter"} else "count"),
                "fact_ids": [linked_fact] if linked_fact else [],
                "layout": {"row": index // 2, "col": index % 2, "w": 6, "h": 4},
            }
        )

    if not charts:
        charts.append(
            {
                "title": "Overview Table",
                "type": "table",
                "x": categorical_cols[0] if categorical_cols else None,
                "y": numeric_cols[0] if numeric_cols else None,
                "group_by": None,
                "aggregation": "count",
                "fact_ids": [str(insight_facts[0].get("id"))] if insight_facts else [],
                "layout": {"row": 0, "col": 0, "w": 12, "h": 4},
            }
        )

    insight_cards: list[dict[str, Any]] = []
    for fact in insight_facts[:4]:
        value = fact.get("value", {})
        metric = value.get("metric", fact.get("type", "fact"))
        insight_cards.append(
            {
                "title": str(metric),
                "text": f"Computed {fact.get('type')} fact available for {metric}.",
                "fact_ids": [str(fact.get("id"))],
            }
        )

    spec = {
        "title": "Auto Healthcare Analytics Dashboard",
        "kpis": kpis,
        "filters": filters,
        "charts": charts,
        "insight_cards": insight_cards,
    }
    validate_schema(spec, DASHBOARD_SPEC_SCHEMA)
    validate_facts_references(spec, facts_bundle)
    return spec


def _render_dashboard_html(spec: dict[str, Any]) -> str:
    chart_items = "".join(
        f"<li><strong>{html.escape(chart.get('title', ''))}</strong> "
        f"({html.escape(str(chart.get('type', '')))}): "
        f"x={html.escape(str(chart.get('x')))}, y={html.escape(str(chart.get('y')))}</li>"
        for chart in spec.get("charts", [])
    )
    kpi_items = "".join(
        f"<li>{html.escape(kpi.get('name', ''))} -> {html.escape(kpi.get('fact_id', ''))}</li>"
        for kpi in spec.get("kpis", [])
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Dashboard Preview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #101828; }}
    h1 {{ margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
  <h1>Dashboard Preview</h1>
  <p>Title: {html.escape(spec.get("title", "Auto Dashboard"))}</p>
  <h3>KPIs</h3>
  <ul>{kpi_items or '<li>No KPIs</li>'}</ul>
  <h3>Charts</h3>
  <ul>{chart_items or '<li>No charts</li>'}</ul>
</body>
</html>
"""


def _chart_figure(df: pd.DataFrame, chart: dict[str, Any]):
    chart_type = chart.get("type")
    title = chart.get("title", chart_type)
    x_col = chart.get("x")
    y_col = chart.get("y")
    group_by = chart.get("group_by")
    agg = chart.get("aggregation", "sum")

    if chart_type == "line":
        if x_col not in df.columns or y_col not in df.columns:
            return None
        scoped = df[[x_col, y_col]].copy()
        scoped[x_col] = pd.to_datetime(scoped[x_col], errors="coerce")
        scoped[y_col] = pd.to_numeric(scoped[y_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            return None
        grouped = (
            scoped.set_index(x_col)[y_col]
            .resample("MS")
            .agg(agg if agg in {"sum", "mean", "count"} else "sum")
            .reset_index()
        )
        return px.line(grouped, x=x_col, y=y_col, title=title, markers=True)

    if chart_type == "bar":
        if x_col not in df.columns or y_col not in df.columns:
            return None
        scoped = df[[x_col, y_col]].copy()
        scoped[y_col] = pd.to_numeric(scoped[y_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            return None
        grouped = (
            scoped.groupby(x_col, dropna=False)[y_col]
            .agg(agg if agg in {"sum", "mean", "count"} else "mean")
            .reset_index()
            .sort_values(y_col, ascending=False)
            .head(10)
        )
        return px.bar(grouped, x=x_col, y=y_col, title=title)

    if chart_type == "hist":
        if x_col not in df.columns:
            return None
        scoped = pd.to_numeric(df[x_col], errors="coerce").dropna()
        if scoped.empty:
            return None
        histogram_df = pd.DataFrame({x_col: scoped})
        return px.histogram(histogram_df, x=x_col, nbins=30, title=title)

    if chart_type == "scatter":
        if x_col not in df.columns or y_col not in df.columns:
            return None
        scoped = df[[x_col, y_col]].copy()
        scoped[x_col] = pd.to_numeric(scoped[x_col], errors="coerce")
        scoped[y_col] = pd.to_numeric(scoped[y_col], errors="coerce")
        if isinstance(group_by, str) and group_by in df.columns:
            scoped[group_by] = df[group_by]
        scoped = scoped.dropna()
        if scoped.empty:
            return None
        return px.scatter(scoped, x=x_col, y=y_col, color=group_by if isinstance(group_by, str) and group_by in scoped.columns else None, title=title)

    if chart_type == "table":
        if x_col not in df.columns:
            return None
        grouped = (
            df[x_col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(15)
            .rename_axis(x_col)
            .reset_index(name="count")
        )
        return px.bar(grouped, x=x_col, y="count", title=title)

    return None


def _chart_images_base64(df: pd.DataFrame, charts: list[dict[str, Any]]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for chart in charts:
        fig = _chart_figure(df, chart)
        if fig is None:
            continue
        try:
            png_bytes = fig.to_image(format="png")
            encoded = base64.b64encode(png_bytes).decode("utf-8")
            images.append({"title": chart.get("title", "Chart"), "png": encoded})
        except Exception:
            continue
    return images


def _render_html_report(
    dataset_id: str,
    meta: dict[str, Any],
    profile: dict[str, Any],
    facts_bundle: dict[str, Any],
    spec: dict[str, Any],
    chart_images: list[dict[str, str]] | None = None,
) -> str:
    facts_html = "".join(
        "<li>"
        f"<strong>{html.escape(str(fact.get('id')))}</strong>: "
        f"{html.escape(str(fact.get('value')))}"
        "</li>"
        for fact in facts_bundle.get("insight_facts", [])
    )
    insights_html = "".join(
        "<li>"
        f"<strong>{html.escape(card.get('title', 'Insight'))}</strong><br/>"
        f"{html.escape(card.get('text', ''))}<br/>"
        f"<small>Fact IDs: {', '.join(card.get('fact_ids', []))}</small>"
        "</li>"
        for card in spec.get("insight_cards", [])
    )

    missing_items = sorted(
        profile.get("missing_percent", {}).items(), key=lambda item: item[1], reverse=True
    )[:10]
    missing_html = "".join(
        f"<li>{html.escape(column)}: {value}% missing</li>" for column, value in missing_items
    )

    chart_titles = "".join(
        f"<li>{html.escape(chart['title'])} ({html.escape(chart['type'])})</li>"
        for chart in spec.get("charts", [])
    )
    chart_images = chart_images or []
    chart_images_html = "".join(
        f"<div class=\"card\"><strong>{html.escape(img['title'])}</strong><br/>"
        f"<img src=\"data:image/png;base64,{img['png']}\" style=\"max-width:100%;\"/></div>"
        for img in chart_images
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analytics Report - {html.escape(dataset_id)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 2rem;
      color: #1d2939;
      line-height: 1.45;
    }}
    h1, h2 {{
      margin-bottom: 0.3rem;
    }}
    .muted {{
      color: #667085;
      font-size: 0.9rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.75rem;
      margin-top: 1rem;
    }}
    .card {{
      border: 1px solid #d0d5dd;
      border-radius: 10px;
      padding: 0.85rem;
      background: #fcfcfd;
    }}
    ul {{
      margin-top: 0.45rem;
    }}
  </style>
</head>
<body>
  <h1>AI Analytics Report</h1>
  <p class="muted">Dataset ID: {html.escape(dataset_id)} | Generated: {_utc_now_iso()}</p>

  <h2>1) Data Quality</h2>
  <div class="grid">
    <div class="card"><strong>Rows</strong><br/>{profile["shape"]["rows"]}</div>
    <div class="card"><strong>Columns</strong><br/>{profile["shape"]["cols"]}</div>
    <div class="card"><strong>Quality Score</strong><br/>{profile.get("quality_score")}/100</div>
    <div class="card"><strong>Duplicate Rows</strong><br/>{profile.get("duplicate_rows")}</div>
  </div>
  <h3>Top Missing Columns</h3>
  <ul>{missing_html or '<li>None</li>'}</ul>

  <h2>2) Key Insights (Facts-Only)</h2>
  <ul>{insights_html or '<li>No insights generated</li>'}</ul>

  <h2>3) Computed Facts</h2>
  <ul>{facts_html or '<li>No facts generated</li>'}</ul>

  <h2>4) Dashboard Plan</h2>
  <p>Title: {html.escape(spec.get("title", "Auto Dashboard"))}</p>
  <ul>{chart_titles or '<li>No chart suggestions</li>'}</ul>
  <div class="grid">{chart_images_html or ''}</div>

  <h2>5) Methods & Limitations</h2>
  <ul>
    <li>Profiling, quality checks, facts, and charts are computed in Python from uploaded data.</li>
    <li>Insights are deterministic summaries tied to fact citations.</li>
    <li>PII detection is heuristic and must be reviewed by a human for compliance.</li>
    <li>This MVP currently exports HTML; PDF/DOCX can be added in later phases.</li>
  </ul>

  <h2>6) Governance Metadata</h2>
  <p class="muted">Created by: {html.escape(meta.get("created_by", "anonymous"))}</p>
  <p class="muted">PII masking enabled: {str(meta.get("pii_masking_enabled", False)).lower()}</p>
</body>
</html>
"""


def _require_session_dir(dataset_id: str) -> Path:
    path = _dataset_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    return path


def _normalize_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    status = str(job.get("status", "queued"))
    if status == "completed":
        status = "succeeded"
    progress_value = job.get("progress")
    progress = 0
    if isinstance(progress_value, (int, float)):
        progress = int(max(0.0, min(100.0, float(progress_value))))
    artifacts = job.get("artifacts") or job.get("artifact_links") or {"facts": None, "spec": None, "report_pdf": None}
    error = job.get("error")
    if isinstance(error, str):
        error = {"code": "job_error", "message": error, "trace": ""}
    return {
        "job_id": job.get("job_id"),
        "dataset_id": job.get("dataset_id"),
        "type": job.get("type"),
        "status": status,
        "progress": progress,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "artifacts": artifacts,
        "error": error,
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "timestamp": _utc_now_iso()}


@app.get("/system/status", response_model=SystemStatusResponse)
def get_system_status(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> SystemStatusResponse:
    actor, role, permissions = _resolve_auth_context(x_api_key, x_user_role)
    visible_sessions = _iter_accessible_sessions(actor, permissions)
    visible_dataset_ids = {dataset_id for dataset_id, _ in visible_sessions}
    visible_documents = _iter_accessible_documents(actor, role)
    now = datetime.now(timezone.utc)
    recent_cutoff = now.replace(microsecond=0) - timedelta(hours=24)

    job_counts = {"active": 0, "queued": 0, "failed": 0}
    for job in list_jobs(dataset_id=None):
        dataset_id = str(job.get("dataset_id") or "")
        if dataset_id and dataset_id not in visible_dataset_ids:
            continue
        status = _normalize_job_payload(job).get("status")
        if status in {"running", "processing"}:
            job_counts["active"] += 1
        elif status == "queued":
            job_counts["queued"] += 1
        elif status == "failed":
            job_counts["failed"] += 1

    active_models = 0
    stale_models = 0
    pending_sensitive_exports = 0
    pending_workflow_reviews = 0
    recent_audit_events = 0
    for dataset_id, meta in visible_sessions:
        registry_entries = _load_ml_registry(dataset_id) if _ml_registry_path(dataset_id).exists() else []
        if any(str(entry.get("status") or "") == "active" for entry in registry_entries):
            active_models += 1

        drift_payload = _load_ml_drift(dataset_id)
        if drift_payload and bool(drift_payload.get("stale_model")):
            stale_models += 1

        approval = _sensitive_export_approval(meta)
        if str(approval.get("status") or "") == "pending":
            pending_sensitive_exports += 1

        workflow_actions = _load_workflow_actions(dataset_id)
        pending_workflow_reviews += sum(1 for action in workflow_actions if str(action.get("status") or "") == WORKFLOW_STATUS_PENDING)
        recent_audit_events += _count_recent_audit_events(dataset_id, recent_cutoff)

    superseded_documents = sum(1 for _, meta in visible_documents if _document_freshness(meta)[0] == "superseded")
    alerts: list[SystemStatusAlert] = []
    if stale_models:
        alerts.append(SystemStatusAlert(level="warning", message=f"{stale_models} governed model run(s) are stale against refreshed data."))
    if pending_sensitive_exports:
        alerts.append(SystemStatusAlert(level="warning", message=f"{pending_sensitive_exports} sensitive export approval request(s) are waiting for review."))
    if pending_workflow_reviews:
        alerts.append(SystemStatusAlert(level="warning", message=f"{pending_workflow_reviews} workflow draft(s) are waiting for reviewer approval."))
    if job_counts["failed"]:
        alerts.append(SystemStatusAlert(level="error", message=f"{job_counts['failed']} backend job(s) are in failed state."))
    if not alerts:
        alerts.append(SystemStatusAlert(level="info", message="Governed backend checks are healthy for the current role scope."))

    return SystemStatusResponse(
        status="ok",
        timestamp=_utc_now_iso(),
        actor=actor,
        role=role,
        counts=SystemStatusCounts(
            visible_sessions=len(visible_sessions),
            visible_documents=len(visible_documents),
            active_jobs=job_counts["active"],
            queued_jobs=job_counts["queued"],
            failed_jobs=job_counts["failed"],
            active_models=active_models,
            stale_models=stale_models,
            pending_sensitive_exports=pending_sensitive_exports,
            pending_workflow_reviews=pending_workflow_reviews,
            superseded_documents=superseded_documents,
            recent_audit_events_24h=recent_audit_events,
        ),
        alerts=alerts,
    )


@app.get("/system/review-queue", response_model=ReviewQueueResponse)
def get_review_queue(
    limit: int = Query(default=12, ge=1, le=40),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ReviewQueueResponse:
    actor, role, permissions = _resolve_auth_context(x_api_key, x_user_role)
    visible_sessions = _iter_accessible_sessions(actor, permissions)
    visible_dataset_ids = {dataset_id for dataset_id, _ in visible_sessions}
    session_meta_by_id = {dataset_id: meta for dataset_id, meta in visible_sessions}
    items: list[dict[str, Any]] = []

    def _dataset_label(dataset_id: str, meta: dict[str, Any]) -> str:
        file_meta = meta.get("file") if isinstance(meta.get("file"), dict) else {}
        return str(meta.get("display_name") or file_meta.get("name") or dataset_id[:8])

    def _queue_item(
        *,
        item_id: str,
        dataset_id: str,
        dataset_label: str,
        category: str,
        severity: str,
        status: str,
        title: str,
        summary: str,
        created_at: str | None,
        updated_at: str | None,
        action_hint: str | None,
    ) -> None:
        fallback_timestamp = _utc_now_iso()
        items.append(
            {
                "item_id": item_id,
                "dataset_id": dataset_id,
                "dataset_label": dataset_label,
                "category": category,
                "severity": severity,
                "status": status,
                "title": title,
                "summary": summary,
                "created_at": created_at or updated_at or fallback_timestamp,
                "updated_at": updated_at or created_at or fallback_timestamp,
                "action_hint": action_hint,
            }
        )

    for dataset_id, meta in visible_sessions:
        dataset_label = _dataset_label(dataset_id, meta)
        approval = _sensitive_export_approval(meta)
        if str(approval.get("status") or "") == "pending":
            justification = str(approval.get("justification") or "").strip()
            summary = justification or "Sensitive export access is waiting for reviewer approval before unmasked export can be enabled."
            _queue_item(
                item_id=f"export:{dataset_id}",
                dataset_id=dataset_id,
                dataset_label=dataset_label,
                category="sensitive_export",
                severity="warning",
                status="pending_review",
                title="Sensitive export approval pending",
                summary=summary,
                created_at=str(approval.get("requested_at") or meta.get("updated_at") or ""),
                updated_at=str(approval.get("requested_at") or meta.get("updated_at") or ""),
                action_hint="Open AI Analytics to review export approval.",
            )

        for action in _load_workflow_actions(dataset_id):
            if str(action.get("status") or "") != WORKFLOW_STATUS_PENDING:
                continue
            _queue_item(
                item_id=f"workflow:{dataset_id}:{action.get('action_id')}",
                dataset_id=dataset_id,
                dataset_label=dataset_label,
                category="workflow_review",
                severity="warning",
                status="pending_review",
                title=str(action.get("title") or "Workflow draft awaiting review"),
                summary=str(action.get("summary") or "A governed workflow draft needs reviewer approval."),
                created_at=str(action.get("generated_at") or ""),
                updated_at=str(action.get("updated_at") or action.get("generated_at") or ""),
                action_hint="Open AI Analytics to approve or reject the workflow draft.",
            )

        for schedule in _load_report_schedules(dataset_id):
            if str(schedule.get("status") or "") != "active":
                continue
            if schedule.get("last_run_at"):
                continue
            frequency = str(schedule.get("frequency") or "scheduled")
            _queue_item(
                item_id=f"schedule:{dataset_id}:{schedule.get('schedule_id')}",
                dataset_id=dataset_id,
                dataset_label=dataset_label,
                category="report_schedule",
                severity="info",
                status="ready_to_run",
                title=str(schedule.get("title") or "Governed report schedule"),
                summary=f"Active {frequency} schedule has not been run yet for this session.",
                created_at=str(schedule.get("created_at") or ""),
                updated_at=str(schedule.get("updated_at") or schedule.get("created_at") or ""),
                action_hint="Open AI Analytics to run the scheduled report now.",
            )

        drift_payload = _load_ml_drift(dataset_id)
        if drift_payload:
            drift_score = float(drift_payload.get("drift_score") or 0.0)
            stale_model = bool(drift_payload.get("stale_model"))
            if stale_model or drift_score >= 0.2:
                severity = "warning" if stale_model or drift_score >= 0.35 else "info"
                summary = str(drift_payload.get("summary") or "Recent data has moved away from the forecast training baseline.")
                _queue_item(
                    item_id=f"model:{dataset_id}:{drift_payload.get('run_id') or 'current'}",
                    dataset_id=dataset_id,
                    dataset_label=dataset_label,
                    category="model_attention",
                    severity=severity,
                    status="attention",
                    title="Forecast model requires review",
                    summary=summary,
                    created_at=str(drift_payload.get("generated_at") or meta.get("updated_at") or ""),
                    updated_at=str(drift_payload.get("generated_at") or meta.get("updated_at") or ""),
                    action_hint="Open AI Analytics to inspect drift and evaluate a challenger run.",
                )

    for document_id, document_meta in _iter_accessible_documents(actor, role):
        freshness, freshness_note, _ = _document_freshness(document_meta)
        if freshness != "superseded":
            continue
        _queue_item(
            item_id=f"document:{document_id}",
            dataset_id="documents",
            dataset_label="Trusted document library",
            category="document_freshness",
            severity="info",
            status="superseded",
            title=str(document_meta.get("title") or document_meta.get("file_name") or document_id),
            summary=freshness_note or "A newer trusted document version exists for this source.",
            created_at=str(document_meta.get("created_at") or ""),
            updated_at=str(document_meta.get("updated_at") or document_meta.get("created_at") or ""),
            action_hint="Open AI Analytics to inspect current document versions and citations.",
        )

    for job in list_jobs(dataset_id=None):
        dataset_id = str(job.get("dataset_id") or "")
        if dataset_id and dataset_id not in visible_dataset_ids:
            continue
        payload = _normalize_job_payload(job)
        if str(payload.get("status") or "") != "failed":
            continue
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        dataset_label = _dataset_label(dataset_id, session_meta_by_id.get(dataset_id, {})) if dataset_id else "Global"
        _queue_item(
            item_id=f"job:{payload.get('job_id')}",
            dataset_id=dataset_id or "system",
            dataset_label=dataset_label,
            category="failed_job",
            severity="error",
            status="failed",
            title=f"{str(payload.get('type') or 'backend').replace('_', ' ').title()} job failed",
            summary=str(error.get("message") or "A backend job failed and needs operator follow-up."),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or ""),
            action_hint="Open AI Analytics or the job trace to retry after reviewing the failure.",
        )

    severity_rank = {"error": 0, "warning": 1, "info": 2}
    sorted_items = sorted(
        items,
        key=lambda item: (
            severity_rank.get(str(item.get("severity") or "info"), 3),
            str(item.get("updated_at") or ""),
            str(item.get("item_id") or ""),
        ),
    )
    sorted_items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    sorted_items.sort(key=lambda item: severity_rank.get(str(item.get("severity") or "info"), 3))

    return ReviewQueueResponse(
        status="ok",
        timestamp=_utc_now_iso(),
        actor=actor,
        role=role,
        total_items=len(sorted_items),
        items=[ReviewQueueItem(**item) for item in sorted_items[:limit]],
    )


@app.get("/auth/me", response_model=AuthContextResponse)
def get_auth_context(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> AuthContextResponse:
    actor, role, permissions = _resolve_auth_context(x_api_key, x_user_role)
    return AuthContextResponse(actor=actor, role=role, permissions=permissions)


@app.get("/documents")
def list_documents(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    actor, role, _ = _resolve_auth_context(x_api_key, x_user_role)
    documents = [
        _document_summary_from_meta(document_id, meta).model_dump()
        for document_id, meta in _iter_accessible_documents(actor, role)
    ]
    documents.sort(key=lambda item: item["updated_at"], reverse=True)
    return {"documents": documents}


@app.post("/documents", response_model=DocumentMetaResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    source_name: str | None = Form(default=None),
    version_label: str | None = Form(default=None),
    effective_date: str | None = Form(default=None),
    supersedes_document_id: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> DocumentMetaResponse:
    actor, role, permissions = _resolve_auth_context(x_api_key, x_user_role)
    if "docs:create" not in permissions and "admin:all" not in permissions:
        raise HTTPException(status_code=403, detail=f"{role} is not allowed to upload trusted documents.")

    filename = file.filename or "document"
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Supported document types are: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="Document too large (max 5MB).")

    try:
        text = _normalize_document_text(_read_document_text(content, extension))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not text:
        raise HTTPException(status_code=400, detail="Document did not contain readable text.")

    chunks = _chunk_document_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document did not produce retrievable text chunks.")

    normalized_effective_date = _normalize_effective_date(effective_date)
    supersedes_id = (supersedes_document_id or "").strip()
    superseded_meta: dict[str, Any] | None = None
    if supersedes_id:
        superseded_meta, _, _ = _authorized_document_context(
            supersedes_id,
            action="write",
            x_api_key=x_api_key,
            x_user_role=x_user_role,
        )

    document_id = str(uuid.uuid4())
    document_dir = _document_path(document_id)
    document_dir.mkdir(parents=True, exist_ok=True)
    _document_content_path(document_id).write_text(text, encoding="utf-8")
    _save_json(_document_chunks_path(document_id), {"chunks": chunks})

    now = _utc_now_iso()
    meta = {
        "document_id": document_id,
        "created_at": now,
        "updated_at": now,
        "created_by": actor,
        "title": (title or "").strip() or Path(filename).stem,
        "source_name": (source_name or "").strip() or Path(filename).stem,
        "status": "ready",
        "file_name": filename,
        "file_type": extension.lstrip("."),
        "chunk_count": len(chunks),
        "char_count": len(text),
        "version_label": (version_label or "").strip() or None,
        "effective_date": normalized_effective_date,
        "supersedes_document_id": supersedes_id or None,
        "superseded_by_document_id": None,
    }
    _save_json(_document_meta_path(document_id), meta)
    if superseded_meta is not None:
        superseded_meta["superseded_by_document_id"] = document_id
        superseded_meta["status"] = "superseded"
        _save_document_meta(supersedes_id, superseded_meta)
    return _document_summary_from_meta(document_id, meta)


@app.get("/documents/{document_id}", response_model=DocumentMetaResponse)
def get_document(
    document_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> DocumentMetaResponse:
    meta, _, _ = _authorized_document_context(document_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return _document_summary_from_meta(document_id, meta)


@app.post("/documents/search", response_model=DocumentSearchResponse)
def search_documents(
    payload: DocumentSearchRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> DocumentSearchResponse:
    actor, role, _ = _resolve_auth_context(x_api_key, x_user_role)
    hits = _search_documents_internal(actor, role, payload.query, document_ids=payload.document_ids, limit=payload.limit)
    return DocumentSearchResponse(query=payload.query, results=hits)


@app.post("/documents/ask", response_model=DocumentAskResponse)
def ask_documents(
    payload: DocumentAskRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> DocumentAskResponse:
    actor, role, _ = _resolve_auth_context(x_api_key, x_user_role)
    hits = _search_documents_internal(actor, role, payload.question, document_ids=payload.document_ids, limit=payload.limit)
    return _answer_documents(payload.question, hits)


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    payload: CreateSessionRequest | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> CreateSessionResponse:
    created_by, role, permissions = _resolve_auth_context(
        x_api_key,
        x_user_role,
        fallback_actor=(payload.created_by if payload else "anonymous"),
    )
    if "sessions:create" not in permissions:
        raise HTTPException(status_code=403, detail=f"{role} is not allowed to create sessions.")
    dataset_id = str(uuid.uuid4())
    session_dir = _dataset_path(dataset_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    now = _utc_now_iso()
    meta = {
        "dataset_id": dataset_id,
        "created_at": now,
        "updated_at": now,
        "status": "created",
        "created_by": created_by,
        "user_id": created_by,
        "display_name": (payload.display_name.strip() if payload and payload.display_name else "") or None,
        "description": (payload.description.strip() if payload and payload.description else "") or "",
        "pii_masking_enabled": False,
        "allow_sensitive_export": False,
        "sensitive_export_approval": _default_sensitive_export_approval(),
        "file": None,
        "artifacts": {},
    }
    _save_json(_meta_path(dataset_id), meta)
    _append_audit(dataset_id, "session_created", created_by, {})
    return CreateSessionResponse(dataset_id=dataset_id, created_at=now)


@app.get("/sessions")
def list_sessions(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    actor, role, permissions = _resolve_auth_context(x_api_key, x_user_role)
    items = [
        _session_summary_from_meta(dataset_id, meta)
        for dataset_id, meta in _iter_accessible_sessions(actor, permissions)
    ]

    items.sort(key=lambda item: item.updated_at, reverse=True)
    return {"sessions": [item.model_dump() for item in items]}


@app.get("/sessions/{dataset_id}", response_model=SessionMetaResponse)
def get_session(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> SessionMetaResponse:
    meta, _, _ = _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return SessionMetaResponse(
        dataset_id=meta["dataset_id"],
        status=meta.get("status", "unknown"),
        created_at=meta.get("created_at", ""),
        updated_at=meta.get("updated_at", ""),
        created_by=meta.get("created_by", "anonymous"),
        user_id=meta.get("user_id"),
        pii_masking_enabled=bool(meta.get("pii_masking_enabled", False)),
        allow_sensitive_export=bool(meta.get("allow_sensitive_export", False)),
        sensitive_export_approval=_sensitive_export_approval(meta),
        file=meta.get("file"),
        file_hash=meta.get("file_hash"),
        artifacts=meta.get("artifacts", {}),
    )


@app.patch("/sessions/{dataset_id}")
def update_session(
    dataset_id: str,
    payload: UpdateSessionRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(dataset_id, action="write", x_api_key=x_api_key, x_user_role=x_user_role)

    if payload.display_name is not None:
        normalized_name = payload.display_name.strip()
        meta["display_name"] = normalized_name or None
    if payload.description is not None:
        meta["description"] = payload.description.strip()

    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "session_updated",
        actor,
        {"display_name": meta.get("display_name"), "description": meta.get("description", "")},
    )
    return {"dataset_id": dataset_id, "session": _session_summary_from_meta(dataset_id, meta).model_dump()}


@app.delete("/sessions/{dataset_id}")
def delete_session(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    session_dir = _require_session_dir(dataset_id)
    meta, actor, _ = _authorized_session_context(dataset_id, action="write", x_api_key=x_api_key, x_user_role=x_user_role)
    _append_audit(dataset_id, "session_deleted", actor, {})
    cache.invalidate_prefix(f"cache:{CACHE_VERSION}:{dataset_id}:")

    for child in sorted(session_dir.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            try:
                child.rmdir()
            except OSError:
                continue
    session_dir.rmdir()
    return {"dataset_id": dataset_id, "deleted": True}


@app.post("/sessions/{dataset_id}/masking")
def update_masking(
    dataset_id: str,
    payload: MaskingRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(dataset_id, action="write", x_api_key=x_api_key, x_user_role=x_user_role)
    meta["pii_masking_enabled"] = bool(payload.enabled)
    _save_meta(dataset_id, meta)
    _append_audit(dataset_id, "masking_toggled", actor, {"enabled": payload.enabled})
    return {"dataset_id": dataset_id, "pii_masking_enabled": meta["pii_masking_enabled"]}


@app.post("/sessions/{dataset_id}/sensitive-export")
def update_sensitive_export(
    dataset_id: str,
    payload: SensitiveExportRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(dataset_id, action="write", x_api_key=x_api_key, x_user_role=x_user_role)
    approval = _sensitive_export_approval(meta)

    if payload.enabled:
        if approval.get("status") != "approved":
            raise HTTPException(
                status_code=409,
                detail="Sensitive export approval is required before enabling unmasked export.",
            )
        meta["allow_sensitive_export"] = True
    else:
        meta["allow_sensitive_export"] = False
        if approval.get("status") == "pending":
            approval["status"] = "cancelled"
        elif approval.get("status") == "approved":
            approval["status"] = "revoked"
        approval["reviewed_by"] = actor
        approval["reviewed_at"] = _utc_now_iso()
        approval["review_note"] = "Sensitive export disabled."
        meta["sensitive_export_approval"] = approval

    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "sensitive_export_toggled",
        actor,
        {"enabled": payload.enabled, "approval_status": approval.get("status")},
    )
    return {
        "dataset_id": dataset_id,
        "allow_sensitive_export": meta["allow_sensitive_export"],
        "sensitive_export_approval": _sensitive_export_approval(meta),
    }


@app.get("/sessions/{dataset_id}/sensitive-export")
def get_sensitive_export_status(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, _, _ = _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return {
        "dataset_id": dataset_id,
        "allow_sensitive_export": bool(meta.get("allow_sensitive_export", False)),
        "sensitive_export_approval": _sensitive_export_approval(meta),
    }


@app.post("/sessions/{dataset_id}/sensitive-export/request")
def request_sensitive_export_approval(
    dataset_id: str,
    payload: SensitiveExportApprovalRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(
        dataset_id,
        action="request_sensitive_export",
        x_api_key=x_api_key,
        x_user_role=x_user_role,
    )
    approval = _sensitive_export_approval(meta)

    approval.update(
        {
            "status": "pending",
            "requested_by": actor,
            "requested_at": _utc_now_iso(),
            "justification": payload.justification.strip(),
            "reviewed_by": None,
            "reviewed_at": None,
            "review_note": None,
        }
    )
    meta["allow_sensitive_export"] = False
    meta["sensitive_export_approval"] = approval
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "sensitive_export_requested",
        actor,
        {"justification": payload.justification.strip()},
    )
    return {
        "dataset_id": dataset_id,
        "allow_sensitive_export": False,
        "sensitive_export_approval": approval,
    }


@app.post("/sessions/{dataset_id}/sensitive-export/decision")
def decide_sensitive_export_approval(
    dataset_id: str,
    payload: SensitiveExportApprovalDecision,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(
        dataset_id,
        action="review_sensitive_export",
        x_api_key=x_api_key,
        x_user_role=x_user_role,
    )
    approval = _sensitive_export_approval(meta)

    if approval.get("status") != "pending":
        raise HTTPException(status_code=409, detail="No pending sensitive export request to review.")

    approval["status"] = "approved" if payload.approved else "rejected"
    approval["reviewed_by"] = actor
    approval["reviewed_at"] = _utc_now_iso()
    approval["review_note"] = payload.note.strip() if payload.note else None
    meta["allow_sensitive_export"] = bool(payload.approved)
    meta["sensitive_export_approval"] = approval
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "sensitive_export_reviewed",
        actor,
        {"approved": payload.approved, "note": approval.get("review_note")},
    )
    return {
        "dataset_id": dataset_id,
        "allow_sensitive_export": meta["allow_sensitive_export"],
        "sensitive_export_approval": approval,
    }


@app.post("/sessions/{dataset_id}/upload")
async def upload_file(
    dataset_id: str,
    file: UploadFile = File(...),
    sheet_name: str | None = Query(default=None, description="Excel sheet to load. Defaults to first sheet."),
    uploaded_by: str = Form(default="anonymous"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    session_dir = _require_session_dir(dataset_id)
    meta, actor, _ = _authorized_session_context(
        dataset_id,
        action="write",
        x_api_key=x_api_key,
        x_user_role=x_user_role,
        fallback_actor=uploaded_by.strip() or "anonymous",
    )

    filename = file.filename or "uploaded_file"
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported in MVP.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 30MB).")

    file_meta: dict[str, Any] = {
        "filename": filename,
        "extension": extension,
        "size_bytes": len(content),
    }

    try:
        if extension == ".csv":
            encoding = _detect_csv_encoding(content)
            file_meta["encoding"] = encoding
        else:
            sheets, selected_sheet = _validate_excel_sheet(content, sheet_name)
            file_meta["sheets"] = sheets
            file_meta["sheet_name"] = selected_sheet
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File validation failed: {exc}") from exc

    stored_name = f"raw{extension}"
    raw_path = session_dir / stored_name
    with raw_path.open("wb") as handle:
        handle.write(content)
    file_meta["raw_path"] = str(raw_path)
    file_meta["uploaded_at"] = _utc_now_iso()
    file_meta["uploaded_by"] = actor

    preserved_artifacts: dict[str, str] = {}
    if _saved_investigations_path(dataset_id).exists():
        preserved_artifacts["investigations"] = str(_saved_investigations_path(dataset_id))
    if _saved_playbooks_path(dataset_id).exists():
        preserved_artifacts["playbooks"] = str(_saved_playbooks_path(dataset_id))
    if _workflow_actions_path(dataset_id).exists():
        preserved_artifacts["workflow_actions"] = str(_workflow_actions_path(dataset_id))
    if _feedback_path(dataset_id).exists():
        preserved_artifacts["feedback"] = str(_feedback_path(dataset_id))
    if _report_schedules_path(dataset_id).exists():
        preserved_artifacts["report_schedules"] = str(_report_schedules_path(dataset_id))
    if _ml_runs_path(dataset_id).exists():
        preserved_artifacts["ml_runs"] = str(_ml_runs_path(dataset_id))
    if _ml_registry_path(dataset_id).exists():
        preserved_artifacts["ml_registry"] = str(_ml_registry_path(dataset_id))

    # New upload invalidates derived artifacts, but forecast runs remain visible so they can be drift-checked against refreshed data.
    meta["file"] = file_meta
    if not meta.get("display_name"):
        meta["display_name"] = Path(filename).stem
    meta["status"] = "uploaded"
    meta["artifacts"] = preserved_artifacts
    meta["allow_sensitive_export"] = False
    meta["sensitive_export_approval"] = _default_sensitive_export_approval()
    meta["file_hash"] = _dataset_signature(meta)
    cache.invalidate_prefix(f"cache:{CACHE_VERSION}:{dataset_id}:")
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "file_uploaded",
        actor,
        {"filename": filename, "extension": extension, "size_bytes": len(content)},
    )

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "stored_as": stored_name,
        "sheet_name": file_meta.get("sheet_name"),
        "encoding": file_meta.get("encoding"),
    }


@app.get("/sessions/{dataset_id}/profile", response_model=ProfileResponse)
def profile_dataset(
    dataset_id: str,
    response: FastAPIResponse,
    mask_pii: bool | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ProfileResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    file_hash = meta.get("file_hash") or _dataset_signature(meta)
    if file_hash and meta.get("file_hash") != file_hash:
        meta["file_hash"] = file_hash

    try:
        df = _read_uploaded_file(meta)
        schema_hash = _compute_schema_hash(df)
        meta["schema_hash"] = schema_hash
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed reading dataset: {exc}") from exc

    cache_key = _cache_key(dataset_id, "profile", file_hash, schema_hash=schema_hash)
    cached = cache.get(cache_key)
    if cached:
        profile = cached
    else:
        profile = _build_profile(df)
        cache.set(cache_key, profile)

    _save_json(_profile_path(dataset_id), profile)
    meta["artifacts"]["profile"] = str(_profile_path(dataset_id))
    meta["status"] = "profiled"
    _save_meta(dataset_id, meta)
    _append_audit(dataset_id, "profile_generated", actor, {})

    if response is not None:
        response.headers["ETag"] = _etag_for(dataset_id, "profile", file_hash, schema_hash=schema_hash)

    mask_enabled = bool(meta.get("pii_masking_enabled", False)) if mask_pii is None else bool(mask_pii)
    response_profile = _apply_profile_masking(profile, mask_enabled)
    return ProfileResponse(dataset_id=dataset_id, profile=response_profile)


@app.get("/sessions/{dataset_id}/cohorts", response_model=CohortAnalysisResponse)
def get_saved_cohort_analysis(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> CohortAnalysisResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    path = _cohort_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No saved cohort analysis exists for this session.")
    cohort = _load_json(path)
    return CohortAnalysisResponse(dataset_id=dataset_id, cohort=CohortAnalysisPayload(**cohort))


@app.post("/sessions/{dataset_id}/cohorts", response_model=CohortAnalysisResponse)
def build_cohort_analysis(
    dataset_id: str,
    request: CohortBuildRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> CohortAnalysisResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    try:
        df = _read_uploaded_file(meta)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed reading dataset: {exc}") from exc

    profile = _load_profile_if_exists(dataset_id)
    if profile is None:
        profile = _build_profile(df)
        _save_json(_profile_path(dataset_id), profile)
        meta.setdefault("artifacts", {})["profile"] = str(_profile_path(dataset_id))

    analysis = _build_cohort_analysis(df, profile, request)
    _save_json(_cohort_path(dataset_id), analysis)
    meta.setdefault("artifacts", {})["cohort_analysis"] = str(_cohort_path(dataset_id))
    meta["status"] = "analyzed"
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "cohort_analysis_generated",
        actor,
        {
            "criteria_count": analysis.get("criteria_count", 0),
            "row_count": analysis.get("row_count", 0),
            "name": analysis.get("name", "Governed cohort"),
        },
    )
    return CohortAnalysisResponse(dataset_id=dataset_id, cohort=CohortAnalysisPayload(**analysis))


@app.get("/sessions/{dataset_id}/investigations", response_model=SavedInvestigationsResponse)
def list_saved_investigations(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> SavedInvestigationsResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return SavedInvestigationsResponse(
        dataset_id=dataset_id,
        investigations=[SavedInvestigationRecord(**item) for item in _load_saved_investigations(dataset_id)],
    )


@app.post("/sessions/{dataset_id}/investigations", response_model=SavedInvestigationRecord)
def save_investigation(
    dataset_id: str,
    payload: InvestigationSaveRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> SavedInvestigationRecord:
    meta, actor, _ = _authorized_session_context(dataset_id, action="write", x_api_key=x_api_key, x_user_role=x_user_role)
    title = payload.title.strip() if payload.title and payload.title.strip() else payload.question.strip()[:120]
    record = {
        "investigation_id": str(uuid.uuid4()),
        "title": title,
        "question": payload.question.strip(),
        "context_type": payload.context_type.strip() or "ask",
        "note": payload.note.strip() if payload.note and payload.note.strip() else None,
        "result": payload.result,
        "created_at": _utc_now_iso(),
        "created_by": actor,
    }
    investigations = [record, *_load_saved_investigations(dataset_id)]
    _save_saved_investigations(dataset_id, investigations)
    meta.setdefault("artifacts", {})["investigations"] = str(_saved_investigations_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "investigation_saved",
        actor,
        {"investigation_id": record["investigation_id"], "context_type": record["context_type"], "title": title},
    )
    return SavedInvestigationRecord(**record)


@app.get("/sessions/{dataset_id}/playbooks", response_model=SavedPlaybooksResponse)
def list_saved_playbooks(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> SavedPlaybooksResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return SavedPlaybooksResponse(
        dataset_id=dataset_id,
        playbooks=[SavedPlaybookRecord(**item) for item in _load_saved_playbooks(dataset_id)],
    )


@app.post("/sessions/{dataset_id}/playbooks", response_model=SavedPlaybookRecord)
def save_playbook(
    dataset_id: str,
    payload: PlaybookSaveRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> SavedPlaybookRecord:
    meta, actor, _ = _authorized_session_context(dataset_id, action="write", x_api_key=x_api_key, x_user_role=x_user_role)
    record = {
        "playbook_id": str(uuid.uuid4()),
        "name": payload.name.strip(),
        "question_template": payload.question_template.strip(),
        "description": payload.description.strip() if payload.description and payload.description.strip() else None,
        "context_type": payload.context_type.strip() or "ask",
        "created_at": _utc_now_iso(),
        "created_by": actor,
    }
    playbooks = [record, *_load_saved_playbooks(dataset_id)]
    _save_saved_playbooks(dataset_id, playbooks)
    meta.setdefault("artifacts", {})["playbooks"] = str(_saved_playbooks_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "playbook_saved",
        actor,
        {"playbook_id": record["playbook_id"], "name": record["name"], "context_type": record["context_type"]},
    )
    return SavedPlaybookRecord(**record)


@app.get("/sessions/{dataset_id}/report-schedules", response_model=ReportSchedulesResponse)
def list_report_schedules(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ReportSchedulesResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return ReportSchedulesResponse(
        dataset_id=dataset_id,
        schedules=[ReportScheduleRecord(**item) for item in _load_report_schedules(dataset_id)],
    )


@app.post("/sessions/{dataset_id}/report-schedules/{schedule_id}/run", response_model=ReportScheduleRunResponse)
def run_report_schedule(
    dataset_id: str,
    schedule_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ReportScheduleRunResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    index, schedule, schedules = _report_schedule_record(dataset_id, schedule_id)
    if str(schedule.get("status") or "active") != "active":
        raise HTTPException(status_code=409, detail="Only active report schedules can be run.")

    queued = _queue_report_job(
        dataset_id,
        actor,
        str(schedule.get("report_template") or "health_report"),
        [str(item) for item in (schedule.get("sections") or [])],
    )
    schedule["last_run_at"] = _utc_now_iso()
    schedule["last_job_id"] = queued["job_id"]
    schedule["last_run_status"] = queued["status"]
    schedule["updated_at"] = schedule["last_run_at"]
    schedules[index] = schedule
    _save_report_schedules(dataset_id, schedules)
    meta.setdefault("artifacts", {})["report_schedules"] = str(_report_schedules_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "report_schedule_run",
        actor,
        {"schedule_id": schedule_id, "job_id": queued["job_id"], "title": schedule.get("title")},
    )
    return ReportScheduleRunResponse(
        dataset_id=dataset_id,
        schedule_id=schedule_id,
        job_id=queued["job_id"],
        status=queued["status"],
    )


@app.get("/sessions/{dataset_id}/ml-runs", response_model=ForecastRunsResponse)
def list_ml_runs(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ForecastRunsResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return ForecastRunsResponse(
        dataset_id=dataset_id,
        runs=[ForecastRunRecord(**item) for item in _load_ml_runs(dataset_id)],
    )


@app.post("/sessions/{dataset_id}/ml-runs/forecast", response_model=ForecastRunRecord)
def train_forecast_run(
    dataset_id: str,
    request: ForecastTrainRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ForecastRunRecord:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    try:
        df = _read_uploaded_file(meta)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed reading dataset: {exc}") from exc

    profile = _load_profile_if_exists(dataset_id)
    if profile is None:
        profile = _build_profile(df)
        _save_json(_profile_path(dataset_id), profile)
        meta.setdefault("artifacts", {})["profile"] = str(_profile_path(dataset_id))

    payload = _build_forecast_run(
        df,
        profile,
        request,
        dataset_hash=str(meta.get("file_hash") or _dataset_signature(meta) or ""),
    )
    record = {
        "run_id": str(uuid.uuid4()),
        "model_kind": "forecast",
        "status": "succeeded",
        "created_at": _utc_now_iso(),
        "created_by": actor,
        "payload": payload,
    }
    runs = [record, *_load_ml_runs(dataset_id)]
    _save_ml_runs(dataset_id, runs)
    meta.setdefault("artifacts", {})["ml_runs"] = str(_ml_runs_path(dataset_id))
    meta["status"] = "modeled"
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "ml_forecast_trained",
        actor,
        {
            "run_id": record["run_id"],
            "metric_field": payload["metric_field"],
            "time_field": payload["time_field"],
            "champion_model": payload["champion_model"],
            "horizon": payload["horizon"],
        },
    )
    return ForecastRunRecord(**record)


@app.get("/sessions/{dataset_id}/ml-runs/drift", response_model=ForecastDriftResponse)
def get_forecast_drift(
    dataset_id: str,
    run_id: str | None = Query(default=None),
    window: int = Query(default=6, ge=3, le=12),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ForecastDriftResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    runs = _load_ml_runs(dataset_id)
    if not runs:
        raise HTTPException(status_code=404, detail="No saved forecast runs are available for this session.")

    selected = runs[0]
    if run_id:
        selected = next((item for item in runs if str(item.get("run_id")) == run_id), None)
        if selected is None:
            raise HTTPException(status_code=404, detail="Requested forecast run was not found.")

    try:
        df = _read_uploaded_file(meta)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed reading dataset: {exc}") from exc

    drift = _build_forecast_drift(dataset_id, meta, df, selected, window=window)
    _save_json(_ml_drift_path(dataset_id), drift)
    meta.setdefault("artifacts", {})["ml_drift"] = str(_ml_drift_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "ml_drift_scanned",
        actor,
        {"run_id": drift.get("run_id"), "drift_score": drift.get("drift_score"), "stale_model": drift.get("stale_model")},
    )
    return ForecastDriftResponse(dataset_id=dataset_id, drift=ForecastDriftPayload(**drift))


@app.get("/sessions/{dataset_id}/ml-registry", response_model=ModelRegistryResponse)
def list_ml_registry(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ModelRegistryResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return ModelRegistryResponse(
        dataset_id=dataset_id,
        entries=[ModelRegistryEntry(**item) for item in _load_ml_registry(dataset_id)],
    )


@app.post("/sessions/{dataset_id}/ml-registry/promote/{run_id}", response_model=ModelRegistryEntry)
def promote_ml_run(
    dataset_id: str,
    run_id: str,
    request: ModelPromotionRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ModelRegistryEntry:
    meta, actor, _ = _authorized_session_context(dataset_id, action="promote_ml", x_api_key=x_api_key, x_user_role=x_user_role)
    runs = _load_ml_runs(dataset_id)
    selected = next((item for item in runs if str(item.get("run_id")) == run_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Requested forecast run was not found.")

    payload = selected.get("payload") if isinstance(selected, dict) else {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="The selected forecast run payload is malformed.")

    training_hash = str(payload.get("training_data_hash") or "") or None
    current_hash = str(meta.get("file_hash") or _dataset_signature(meta) or "") or None
    if training_hash and current_hash and training_hash != current_hash:
        raise HTTPException(status_code=400, detail="Stale forecast runs cannot be promoted. Retrain on the current dataset first.")

    promoted_at = _utc_now_iso()
    existing_entries = _load_ml_registry(dataset_id)
    updated_entries: list[dict[str, Any]] = []
    for entry in existing_entries:
        current = dict(entry)
        if current.get("status") == "active":
            current["status"] = "archived"
        updated_entries.append(current)

    record = {
        "registry_id": str(uuid.uuid4()),
        "run_id": run_id,
        "model_kind": str(selected.get("model_kind") or "forecast"),
        "status": "active",
        "promoted_at": promoted_at,
        "promoted_by": actor,
        "note": request.note.strip() if request.note and request.note.strip() else None,
        "name": str(payload.get("name") or run_id),
        "champion_model": str(payload.get("champion_model") or "unknown"),
        "metric_field": str(payload.get("metric_field") or ""),
        "time_field": str(payload.get("time_field") or ""),
        "aggregation": str(payload.get("aggregation") or "sum"),
        "source_data_hash": training_hash,
    }
    entries = [record, *updated_entries]
    _save_ml_registry(dataset_id, entries)
    meta.setdefault("artifacts", {})["ml_registry"] = str(_ml_registry_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "ml_model_promoted",
        actor,
        {
            "registry_id": record["registry_id"],
            "run_id": run_id,
            "model_kind": record["model_kind"],
            "champion_model": record["champion_model"],
        },
    )
    return ModelRegistryEntry(**record)


@app.get("/sessions/{dataset_id}/ml-evaluation", response_model=ModelEvaluationResponse)
def evaluate_ml_models(
    dataset_id: str,
    challenger_run_id: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> ModelEvaluationResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    registry_entries = _load_ml_registry(dataset_id)
    active_entry = next((entry for entry in registry_entries if str(entry.get("status") or "") == "active"), None)
    if active_entry is None:
        raise HTTPException(status_code=404, detail="No active governed model is registered for this session.")

    runs = _load_ml_runs(dataset_id)
    if not runs:
        raise HTTPException(status_code=404, detail="No saved forecast runs are available for this session.")

    active_run_id = str(active_entry.get("run_id") or "")
    active_run = _find_ml_run(runs, active_run_id)
    if active_run is None:
        raise HTTPException(status_code=404, detail="The active registry run could not be found in saved model runs.")

    challenger_run = None
    if challenger_run_id:
        challenger_run = _find_ml_run(runs, challenger_run_id)
        if challenger_run is None:
            raise HTTPException(status_code=404, detail="Requested challenger run was not found.")
    else:
        challenger_run = next((item for item in runs if str(item.get("run_id") or "") != active_run_id), None)
    if challenger_run is None:
        raise HTTPException(status_code=400, detail="No challenger run is available. Train another forecast run first.")

    try:
        df = _read_uploaded_file(meta)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed reading dataset: {exc}") from exc

    evaluation = _build_model_evaluation(dataset_id, meta, df, active_run, challenger_run)
    _save_json(_ml_evaluation_path(dataset_id), evaluation)
    meta.setdefault("artifacts", {})["ml_evaluation"] = str(_ml_evaluation_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "ml_models_evaluated",
        actor,
        {
            "active_run_id": evaluation.get("active_run", {}).get("run_id"),
            "challenger_run_id": evaluation.get("challenger_run", {}).get("run_id"),
            "winner": evaluation.get("winner"),
        },
    )
    return ModelEvaluationResponse(dataset_id=dataset_id, evaluation=ModelEvaluationPayload(**evaluation))


@app.get("/sessions/{dataset_id}/workflow-actions", response_model=WorkflowActionsResponse)
def list_workflow_actions(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> WorkflowActionsResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    actions = _load_workflow_actions(dataset_id)
    return WorkflowActionsResponse(
        dataset_id=dataset_id,
        actions=[WorkflowActionRecord(**action) for action in actions],
    )


@app.post("/sessions/{dataset_id}/workflow-actions/draft", response_model=WorkflowActionRecord)
def draft_workflow_action(
    dataset_id: str,
    request: WorkflowDraftRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> WorkflowActionRecord:
    meta, actor, _ = _authorized_session_context(dataset_id, action="draft_workflow", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    profile = _load_profile_if_exists(dataset_id)
    if profile is None:
        try:
            df = _read_uploaded_file(meta)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed reading dataset: {exc}") from exc
        profile = _build_profile(df)
        _save_json(_profile_path(dataset_id), profile)
        meta.setdefault("artifacts", {})["profile"] = str(_profile_path(dataset_id))

    facts = _load_optional_json(_facts_path(dataset_id))
    anomalies = _load_optional_json(_anomaly_path(dataset_id))
    cohort = _load_optional_json(_cohort_path(dataset_id))
    title, summary, payload, evidence = _workflow_action_content(
        dataset_id,
        meta,
        request,
        profile,
        facts,
        anomalies,
        cohort,
    )

    timestamp = _utc_now_iso()
    action = {
        "action_id": str(uuid.uuid4()),
        "action_type": request.action_type.strip().lower(),
        "status": WORKFLOW_STATUS_PENDING,
        "title": title,
        "target": request.target.strip() if request.target and request.target.strip() else None,
        "objective": request.objective.strip() if request.objective and request.objective.strip() else None,
        "summary": summary,
        "payload": payload,
        "evidence": evidence,
        "generated_at": timestamp,
        "updated_at": timestamp,
        "created_by": actor,
        "review_note": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "executed_by": None,
        "executed_at": None,
        "execution_note": None,
        "requires_approval": True,
    }
    actions = [action, *_load_workflow_actions(dataset_id)]
    _save_workflow_actions(dataset_id, actions)
    meta.setdefault("artifacts", {})["workflow_actions"] = str(_workflow_actions_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "workflow_action_drafted",
        actor,
        {"action_id": action["action_id"], "action_type": action["action_type"], "title": title},
    )
    return WorkflowActionRecord(**action)


@app.post("/sessions/{dataset_id}/workflow-actions/{action_id}/decision", response_model=WorkflowActionRecord)
def review_workflow_action(
    dataset_id: str,
    action_id: str,
    request: WorkflowDecisionRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> WorkflowActionRecord:
    meta, actor, _ = _authorized_session_context(dataset_id, action="review_workflow", x_api_key=x_api_key, x_user_role=x_user_role)
    actions = _load_workflow_actions(dataset_id)
    index = _workflow_action_index(actions, action_id)
    current = dict(actions[index])
    if current.get("status") == WORKFLOW_STATUS_EXECUTED:
        raise HTTPException(status_code=400, detail="Executed workflow actions cannot be reviewed again.")

    current["status"] = WORKFLOW_STATUS_APPROVED if request.approved else WORKFLOW_STATUS_REJECTED
    current["review_note"] = request.note
    current["reviewed_by"] = actor
    current["reviewed_at"] = _utc_now_iso()
    current["updated_at"] = current["reviewed_at"]
    actions[index] = current
    _save_workflow_actions(dataset_id, actions)
    meta.setdefault("artifacts", {})["workflow_actions"] = str(_workflow_actions_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "workflow_action_reviewed",
        actor,
        {"action_id": action_id, "approved": request.approved, "status": current["status"]},
    )
    return WorkflowActionRecord(**current)


@app.post("/sessions/{dataset_id}/workflow-actions/{action_id}/execute", response_model=WorkflowActionRecord)
def execute_workflow_action(
    dataset_id: str,
    action_id: str,
    request: WorkflowExecutionRequest | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> WorkflowActionRecord:
    meta, actor, _ = _authorized_session_context(dataset_id, action="execute_workflow", x_api_key=x_api_key, x_user_role=x_user_role)
    actions = _load_workflow_actions(dataset_id)
    index = _workflow_action_index(actions, action_id)
    current = dict(actions[index])

    if current.get("status") != WORKFLOW_STATUS_APPROVED:
        raise HTTPException(status_code=400, detail="Workflow actions must be approved before execution.")

    executed_at = _utc_now_iso()
    current["status"] = WORKFLOW_STATUS_EXECUTED
    current["executed_by"] = actor
    current["executed_at"] = executed_at
    current["execution_note"] = request.note if request else None
    current["updated_at"] = executed_at
    payload = dict(current.get("payload") or {})
    if str(current.get("action_type") or "") == "schedule_report":
        schedule = _create_report_schedule_from_action(dataset_id, current, actor)
        payload["schedule_id"] = schedule["schedule_id"]
        payload["schedule_status"] = schedule["status"]
        meta.setdefault("artifacts", {})["report_schedules"] = str(_report_schedules_path(dataset_id))
        _append_audit(
            dataset_id,
            "report_schedule_created",
            actor,
            {"schedule_id": schedule["schedule_id"], "title": schedule["title"], "frequency": schedule["frequency"]},
        )
    payload["execution_mode"] = "governed_manual"
    payload["execution_result"] = _workflow_action_execution_result(str(current.get("action_type") or ""))
    current["payload"] = payload
    actions[index] = current
    _save_workflow_actions(dataset_id, actions)
    meta.setdefault("artifacts", {})["workflow_actions"] = str(_workflow_actions_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "workflow_action_executed",
        actor,
        {"action_id": action_id, "action_type": current.get("action_type")},
    )
    return WorkflowActionRecord(**current)


@app.get("/sessions/{dataset_id}/anomalies", response_model=AnomalyAnalysisResponse)
def get_anomaly_analysis(
    dataset_id: str,
    limit: int = Query(default=6, ge=1, le=12),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> AnomalyAnalysisResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    try:
        df = _read_uploaded_file(meta)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed reading dataset: {exc}") from exc

    profile = _load_profile_if_exists(dataset_id)
    if profile is None:
        profile = _build_profile(df)
        _save_json(_profile_path(dataset_id), profile)
        meta.setdefault("artifacts", {})["profile"] = str(_profile_path(dataset_id))

    analysis = _build_anomaly_analysis(df, profile, limit=limit)
    _save_json(_anomaly_path(dataset_id), analysis)
    meta.setdefault("artifacts", {})["anomaly_analysis"] = str(_anomaly_path(dataset_id))
    meta["status"] = "analyzed"
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "anomaly_analysis_generated",
        actor,
        {"limit": limit, "anomaly_count": analysis.get("anomaly_count", 0)},
    )
    return AnomalyAnalysisResponse(dataset_id=dataset_id, analysis=AnomalyAnalysisPayload(**analysis))


@app.get("/sessions/{dataset_id}/semantic-layer", response_model=SemanticLayerResponse)
def get_semantic_layer(
    dataset_id: str,
    mask_pii: bool | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> SemanticLayerResponse:
    meta, _, _ = _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    profile_path = _profile_path(dataset_id)
    if profile_path.exists():
        profile = _load_json(profile_path)
    else:
        profile = _build_profile(_read_uploaded_file(meta))
        _save_json(profile_path, profile)

    mask_enabled = bool(meta.get("pii_masking_enabled", False)) if mask_pii is None else bool(mask_pii)
    response_profile = _apply_profile_masking(profile, mask_enabled)
    semantic_layer = build_semantic_layer(response_profile)
    return SemanticLayerResponse(dataset_id=dataset_id, semantic_layer=semantic_layer)


@app.get("/sessions/{dataset_id}/facts")
def generate_facts(
    dataset_id: str,
    response: FastAPIResponse,
    mode: str = Query(default="auto"),
    force: bool = Query(default=False),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    file_hash = meta.get("file_hash") or _dataset_signature(meta)
    if file_hash and meta.get("file_hash") != file_hash:
        meta["file_hash"] = file_hash
        _save_meta(dataset_id, meta)
    df: pd.DataFrame | None = None
    schema_hash = meta.get("schema_hash", "")
    if not schema_hash:
        df = _read_uploaded_file(meta)
        schema_hash = _compute_schema_hash(df)
        meta["schema_hash"] = schema_hash
        _save_meta(dataset_id, meta)

    cache_key = _cache_key(dataset_id, "facts", file_hash, schema_hash=schema_hash, params=mode)
    cached = cache.get(cache_key)
    if cached and not force:
        if meta.get("pii_masking_enabled", False):
            profile_for_mask = _load_profile_if_exists(dataset_id) or {}
            aliases = _pii_aliases(profile_for_mask)
            if aliases:
                cached = _mask_recursive(json.loads(json.dumps(cached)), aliases)
        if response is not None:
            response.headers["ETag"] = _etag_for(dataset_id, "facts", file_hash, schema_hash=schema_hash, params=mode)
        return {"dataset_id": dataset_id, "facts_bundle": cached, "cached": True}

    facts_path = _facts_path(dataset_id)
    if facts_path.exists() and not force:
        facts_bundle = _load_json(facts_path)
        if facts_bundle.get("dataset_hash") == file_hash:
            cache.set(cache_key, facts_bundle)
            if meta.get("pii_masking_enabled", False):
                profile_for_mask = _load_profile_if_exists(dataset_id) or {}
                aliases = _pii_aliases(profile_for_mask)
                if aliases:
                    facts_bundle = _mask_recursive(json.loads(json.dumps(facts_bundle)), aliases)
            if response is not None:
                response.headers["ETag"] = _etag_for(
                    dataset_id, "facts", file_hash, schema_hash=schema_hash, params=mode
                )
            return {"dataset_id": dataset_id, "facts_bundle": facts_bundle, "cached": True}

    if df is None:
        df = _read_uploaded_file(meta)
    rows, cols = df.shape
    if mode not in {"auto", "sample", "full"}:
        raise HTTPException(status_code=400, detail="Mode must be one of: auto, sample, full.")

    if mode == "auto":
        requested_mode = "full" if rows <= SMALL_ROW_MAX and cols <= MID_COL_MAX else "sample"
    else:
        requested_mode = mode

    requires_async = rows > SMALL_ROW_MAX
    if requires_async and not force:
        seed = _seed_from_dataset_hash(file_hash)
        job = create_job(
            "facts",
            dataset_id,
            {"mode": requested_mode, "seed": seed, "file_hash": file_hash, "schema_hash": schema_hash},
        )
        try:
            from backend.tasks import generate_facts_task

            generate_facts_task.delay(job["job_id"], dataset_id, requested_mode, seed)
            _append_audit(dataset_id, "facts_job_queued", actor, {"job_id": job["job_id"], "mode": requested_mode})
        except Exception as exc:
            try:
                generate_facts_task(job["job_id"], dataset_id, requested_mode, seed)
                facts_bundle = _load_json(_facts_path(dataset_id))
                cache.set(cache_key, facts_bundle)
                if response is not None:
                    response.headers["ETag"] = _etag_for(
                        dataset_id, "facts", file_hash, schema_hash=schema_hash, params=mode
                    )
                return {"dataset_id": dataset_id, "facts_bundle": facts_bundle, "cached": False, "queue_fallback": True}
            except Exception as inner_exc:
                update_job(job["job_id"], status="failed", error=str(inner_exc))
                raise HTTPException(status_code=503, detail=f"Failed to run facts job: {exc}; {inner_exc}") from inner_exc
        return Response(
            content=json.dumps(
                {"dataset_id": dataset_id, "job_id": job["job_id"], "status": job["status"], "queued": True}
            ),
            media_type="application/json",
            status_code=202,
        )

    sampling_seed = _seed_from_dataset_hash(file_hash)
    if requested_mode == "sample":
        sampled, sampling_method = _determine_sampling_strategy(df, _build_profile(df), sampling_seed, SAMPLE_MAX_ROWS)
        bias_note = (
            "Large dataset sampled for interactive use; run async full mode for complete computation."
            if rows > MID_ROW_MAX
            else "Sample used for interactive speed; run full mode for final reporting."
        )
        profile = _build_profile(sampled)
        profile["data_coverage"] = {
            "mode": "sample",
            "rows_total": rows,
            "rows_used": int(len(sampled)),
            "sampling_method": sampling_method,
            "seed": sampling_seed,
            "bias_notes": bias_note,
        }
        facts_bundle = _build_facts_bundle(sampled, profile, dataset_id=dataset_id, dataset_hash=file_hash)
    else:
        profile = _build_profile(df)
        profile["data_coverage"] = {
            "mode": "full",
            "rows_total": rows,
            "rows_used": rows,
            "sampling_method": "uniform",
            "seed": None,
            "bias_notes": "Full dataset used.",
        }
        facts_bundle = _build_facts_bundle(df, profile, dataset_id=dataset_id, dataset_hash=file_hash)
    _save_json(_facts_path(dataset_id), facts_bundle)
    _save_json(_profile_path(dataset_id), profile)
    meta["artifacts"]["facts"] = str(_facts_path(dataset_id))
    meta["artifacts"]["profile"] = str(_profile_path(dataset_id))
    meta["schema_hash"] = schema_hash
    meta["status"] = "facts_generated"
    _save_meta(dataset_id, meta)
    _append_audit(dataset_id, "facts_generated", actor, {"mode": requested_mode})

    cache.set(cache_key, facts_bundle)
    if response is not None:
        response.headers["ETag"] = _etag_for(dataset_id, "facts", file_hash, schema_hash=schema_hash, params=mode)
    if meta.get("pii_masking_enabled", False):
        pii_columns = profile.get("pii_candidates", [])
        aliases = {column: f"pii_field_{index + 1}" for index, column in enumerate(pii_columns)}
        masked_bundle = _mask_recursive(json.loads(json.dumps(facts_bundle)), aliases)
        return {"dataset_id": dataset_id, "facts_bundle": masked_bundle}

    return {"dataset_id": dataset_id, "facts_bundle": facts_bundle}


@app.post("/sessions/{dataset_id}/facts")
def regenerate_facts(
    dataset_id: str,
    mode: str = Query(default="full"),
    seed: int | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    file_hash = meta.get("file_hash") or _dataset_signature(meta)
    if file_hash and meta.get("file_hash") != file_hash:
        meta["file_hash"] = file_hash
        _save_meta(dataset_id, meta)
    resolved_seed = seed if seed is not None else _seed_from_dataset_hash(file_hash)
    schema_hash = meta.get("schema_hash", "")
    cache.invalidate_prefix(f"cache:{CACHE_VERSION}:{dataset_id}:facts:")
    job = create_job(
        "facts",
        dataset_id,
        {"mode": mode, "seed": resolved_seed, "file_hash": file_hash, "schema_hash": schema_hash, "force": True},
    )
    try:
        from backend.tasks import generate_facts_task

        generate_facts_task.delay(job["job_id"], dataset_id, mode, resolved_seed)
        _append_audit(dataset_id, "facts_job_forced", actor, {"job_id": job["job_id"], "mode": mode})
    except Exception as exc:
        try:
            generate_facts_task(job["job_id"], dataset_id, mode, resolved_seed)
        except Exception as inner_exc:
            update_job(job["job_id"], status="failed", error=str(inner_exc))
            raise HTTPException(status_code=503, detail=f"Failed to run facts job: {exc}; {inner_exc}") from inner_exc

    return Response(
        content=json.dumps({"dataset_id": dataset_id, "job_id": job["job_id"], "status": job["status"]}),
        media_type="application/json",
        status_code=202,
    )


@app.post("/sessions/{dataset_id}/clean")
def clean_dataset(
    dataset_id: str,
    payload: CleanRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    df = _read_uploaded_file(meta)
    cleaned = _apply_cleaning_actions(df, payload.actions)
    cleaned_path = _cleaned_path(dataset_id)
    cleaned.to_csv(cleaned_path, index=False)

    meta["artifacts"]["cleaned"] = str(cleaned_path)
    meta["status"] = "cleaned"
    cache.invalidate_prefix(f"cache:{CACHE_VERSION}:{dataset_id}:")
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "dataset_cleaned",
        actor,
        {"actions": payload.actions, "pii_mask": payload.pii_mask},
    )

    return {"dataset_id": dataset_id, "status": "cleaned", "cleaned_path": str(cleaned_path)}


@app.get("/sessions/{dataset_id}/dashboard-spec", response_model=DashboardSpecResponse)
@app.post("/sessions/{dataset_id}/dashboard-spec", response_model=DashboardSpecResponse)
def generate_dashboard_spec(
    dataset_id: str,
    response: FastAPIResponse,
    payload: DashboardSpecRequest | None = None,
    use_llm: bool = Query(default=True),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> DashboardSpecResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    file_hash = meta.get("file_hash") or _dataset_signature(meta)
    if file_hash and meta.get("file_hash") != file_hash:
        meta["file_hash"] = file_hash
        _save_meta(dataset_id, meta)
    df = _read_uploaded_file(meta)
    schema_hash = meta.get("schema_hash") or _compute_schema_hash(df)
    meta["schema_hash"] = schema_hash
    cache_key = _cache_key(dataset_id, "dashboard_spec", file_hash, schema_hash=schema_hash, params=str(use_llm))
    cached = cache.get(cache_key)
    if cached:
        if response is not None:
            response.headers["ETag"] = _etag_for(
                dataset_id, "dashboard_spec", file_hash, schema_hash=schema_hash, params=str(use_llm)
            )
        return DashboardSpecResponse(dataset_id=dataset_id, dashboard_spec=cached)

    profile_path = _profile_path(dataset_id)
    facts_path = _facts_path(dataset_id)
    if profile_path.exists():
        profile = _load_json(profile_path)
    else:
        profile = _build_profile(df)
        _save_json(profile_path, profile)

    if facts_path.exists():
        facts_bundle = _load_json(facts_path)
        if not _is_facts_bundle_compatible(facts_bundle):
            facts_bundle = _build_facts_bundle(df, profile, dataset_id=dataset_id, dataset_hash=file_hash)
            _save_json(facts_path, facts_bundle)
    else:
        facts_bundle = _build_facts_bundle(df, profile, dataset_id=dataset_id, dataset_hash=file_hash)
        _save_json(facts_path, facts_bundle)

    template = payload.template if payload else "health_core"
    if use_llm:
        try:
            spec = _generate_dashboard_spec_llm(dataset_id, template, facts_bundle)
        except (SchemaValidationError, FactsGroundingError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid dashboard spec from LLM: {exc}") from exc
        except Exception as exc:
            spec = _build_dashboard_spec(dataset_id, profile, facts_bundle)
            _append_audit(
                dataset_id,
                "dashboard_spec_generated",
                actor,
                {"use_llm": False, "fallback_reason": str(exc), "template": template},
            )
    else:
        spec = _build_dashboard_spec(dataset_id, profile, facts_bundle)

    _save_json(_dashboard_spec_path(dataset_id), spec)
    cache.set(cache_key, spec)
    meta["artifacts"]["profile"] = str(profile_path)
    meta["artifacts"]["facts"] = str(facts_path)
    meta["artifacts"]["dashboard_spec"] = str(_dashboard_spec_path(dataset_id))
    meta["status"] = "dashboard_spec_generated"
    _save_meta(dataset_id, meta)
    _append_audit(dataset_id, "dashboard_spec_generated", actor, {"use_llm": use_llm, "template": template})
    if response is not None:
        response.headers["ETag"] = _etag_for(
            dataset_id, "dashboard_spec", file_hash, schema_hash=schema_hash, params=str(use_llm)
        )
    return DashboardSpecResponse(dataset_id=dataset_id, dashboard_spec=spec)


@app.get("/sessions/{dataset_id}/dashboard")
def get_dashboard(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    spec_path = _dashboard_spec_path(dataset_id)
    if not spec_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard spec not generated yet.")
    spec = _load_json(spec_path)
    return {"dataset_id": dataset_id, "dashboard_html": _render_dashboard_html(spec), "spec_version": spec.get("version")}


def _execute_governed_ask(
    dataset_id: str,
    question: str,
    meta: dict[str, Any],
    actor: str,
    *,
    audit_action: str = "ask_data",
    audit_details: dict[str, Any] | None = None,
) -> AskResponse:
    facts_path = _facts_path(dataset_id)
    if facts_path.exists():
        facts_bundle = _load_json(facts_path)
        if not _is_facts_bundle_compatible(facts_bundle):
            if not meta.get("file"):
                raise HTTPException(status_code=400, detail="No file uploaded for this session.")
            refreshed_df = _read_uploaded_file(meta)
            refreshed_profile = _build_profile(refreshed_df)
            refreshed_profile["data_coverage"] = {
                "mode": "full",
                "rows_total": int(len(refreshed_df)),
                "rows_used": int(len(refreshed_df)),
                "sampling_method": "uniform",
                "seed": None,
                "bias_notes": "Full dataset used.",
            }
            facts_bundle = _build_facts_bundle(
                refreshed_df,
                refreshed_profile,
                dataset_id=dataset_id,
                dataset_hash=str(meta.get("file_hash") or _dataset_signature(meta)),
            )
            _save_json(facts_path, facts_bundle)
    else:
        raise HTTPException(status_code=400, detail="Facts bundle not generated. Run /sessions/{id}/facts first.")

    profile_path = _profile_path(dataset_id)
    if profile_path.exists():
        profile = _load_json(profile_path)
    else:
        if not meta.get("file"):
            raise HTTPException(status_code=400, detail="No file uploaded for this session.")
        profile = _build_profile(_read_uploaded_file(meta))
        _save_json(profile_path, profile)

    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")
    df = _read_uploaded_file(meta)
    semantic_layer = build_semantic_layer(profile)

    try:
        query_plan = _generate_query_plan_llm(question, facts_bundle, profile, semantic_layer)
    except Exception:
        query_plan = _fallback_query_plan(question, profile, semantic_layer)

    try:
        validate_schema(query_plan, QUERY_PLAN_SCHEMA)
        query_plan = validate_and_resolve_query_plan(query_plan, profile, semantic_layer)
    except (QueryPlanValidationError, SchemaValidationError) as exc:
        details = {"question": question, "status": "invalid_plan", "error": str(exc)}
        if audit_details:
            details.update(audit_details)
        _append_audit(
            dataset_id,
            audit_action,
            actor,
            details,
        )
        raise HTTPException(status_code=422, detail=f"Invalid query plan: {exc}") from exc

    result_df, execution_note = _execute_query_plan(df, query_plan)
    candidate_fact_ids = _default_fact_ids(facts_bundle)

    try:
        answer, facts_used = _summarize_query_result_llm(question, result_df, facts_bundle, candidate_fact_ids)
    except Exception:
        fallback_answer, fallback_facts, _, _, _ = _safe_answer_from_facts(question, facts_bundle)
        answer = f"{fallback_answer} ({execution_note})"
        facts_used = fallback_facts or candidate_fact_ids

    details = {"question": question, "facts_used": facts_used, "status": "ok"}
    if audit_details:
        details.update(audit_details)
    _append_audit(
        dataset_id,
        audit_action,
        actor,
        details,
    )

    data_coverage = (facts_bundle.get("data_coverage") or {}).get("mode", "sample")
    confidence = "Medium" if facts_used else "Low"
    fact_coverage = 1.0 if facts_used else 0.0
    result_rows = json.loads(result_df.head(min(int(query_plan.get("limit", 25)), 100)).to_json(orient="records", date_format="iso"))
    chart_type = query_plan.get("chart_hint", "table")
    columns = list(result_df.columns)
    used_metric_ids = [
        str(metric["metric_id"])
        for metric in query_plan.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("metric_id"), str)
    ]
    chart_payload: dict[str, Any] | None = None
    if chart_type != "table" and columns:
        x_col = columns[0]
        y_col = columns[1] if len(columns) > 1 else columns[0]
        chart_payload = {
            "type": chart_type,
            "x": x_col,
            "y": y_col,
            "title": f"Ask result: {chart_type}",
        }
    else:
        chart_payload = {"type": "table", "columns": columns}

    return AskResponse(
        dataset_id=dataset_id,
        answer=answer,
        facts_used=facts_used,
        confidence=confidence,
        fact_coverage=fact_coverage,
        data_coverage=data_coverage,
        query_plan=query_plan,
        result_rows=result_rows,
        chart=chart_payload,
        governance={
            "validation_mode": "semantic_strict",
            "semantic_metric_ids": used_metric_ids,
            "blocked_fields": semantic_layer.get("pii_blocked_fields", []),
        },
    )


@app.post("/sessions/{dataset_id}/ask", response_model=AskResponse)
def ask_dataset(
    dataset_id: str,
    payload: AskRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> AskResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    return _execute_governed_ask(dataset_id, payload.question, meta, actor)


@app.post("/sessions/{dataset_id}/playbooks/{playbook_id}/run", response_model=AskResponse)
def run_playbook(
    dataset_id: str,
    playbook_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> AskResponse:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    playbook = _playbook_record(dataset_id, playbook_id)
    question = str(playbook.get("question_template") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Playbook does not have a runnable question template.")
    return _execute_governed_ask(
        dataset_id,
        question,
        meta,
        actor,
        audit_action="playbook_run",
        audit_details={"playbook_id": playbook_id, "name": str(playbook.get("name") or "Unnamed playbook")},
    )


@app.get("/sessions/{dataset_id}/preview")
def preview_dataset(
    dataset_id: str,
    limit: int = Query(default=1000, ge=1, le=5000),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, _, _ = _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    df = _read_uploaded_file(meta)
    return {
        "dataset_id": dataset_id,
        "rows": _to_json_compatible_rows(df, limit=limit),
        "columns": [str(column) for column in df.columns],
        "row_count": int(df.shape[0]),
    }


@app.post("/sessions/{dataset_id}/report")
def generate_report(
    dataset_id: str,
    payload: ReportRequest | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    meta, actor, _ = _authorized_session_context(dataset_id, action="compute", x_api_key=x_api_key, x_user_role=x_user_role)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    payload = payload or ReportRequest()
    job = _queue_report_job(dataset_id, actor, payload.template, payload.sections)

    return Response(
        content=json.dumps(job),
        media_type="application/json",
        status_code=202,
    )


@app.get("/sessions/{dataset_id}/report/html")
def get_report_html(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
    actor: str | None = Query(default=None),
    role: str | None = Query(default=None),
) -> HTMLResponse:
    meta, resolved_actor, _ = _authorized_session_context(
        dataset_id,
        action="export_masked",
        x_api_key=x_api_key,
        x_user_role=x_user_role,
        actor_query=actor,
        role_query=role,
    )
    path = _report_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    _append_audit(dataset_id, "export_html", resolved_actor, {"format": "html"})
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/sessions/{dataset_id}/report/pdf")
def get_report_pdf(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
    actor: str | None = Query(default=None),
    role: str | None = Query(default=None),
) -> Response:
    meta, resolved_actor, _ = _authorized_session_context(
        dataset_id,
        action="export_masked",
        x_api_key=x_api_key,
        x_user_role=x_user_role,
        actor_query=actor,
        role_query=role,
    )
    path = _report_pdf_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report PDF not generated yet.")
    _append_audit(dataset_id, "export_pdf", resolved_actor, {"format": "pdf"})
    return Response(content=path.read_bytes(), media_type="application/pdf")


@app.get("/sessions/{dataset_id}/export/pdf")
def export_pdf_direct(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
    actor: str | None = Query(default=None),
    role: str | None = Query(default=None),
) -> Response:
    return get_report_pdf(dataset_id, x_api_key=x_api_key, x_user_role=x_user_role, actor=actor, role=role)


@app.get("/jobs")
def list_all_jobs(
    dataset_id: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    actor, role, permissions = _resolve_auth_context(x_api_key, x_user_role)
    if dataset_id:
        meta, _, _ = _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
        del meta
        return {"jobs": [_normalize_job_payload(job) for job in list_jobs(dataset_id=dataset_id)]}

    jobs = list_jobs(dataset_id=None)
    if "sessions:read_all" in permissions:
        return {"jobs": [_normalize_job_payload(job) for job in jobs]}

    visible_jobs: list[dict[str, Any]] = []
    for job in jobs:
        job_dataset_id = str(job.get("dataset_id") or "")
        if not job_dataset_id:
            continue
        try:
            _authorized_session_context(
                job_dataset_id,
                action="read",
                x_api_key=x_api_key,
                x_user_role=x_user_role,
                fallback_actor=actor,
            )
            visible_jobs.append(job)
        except HTTPException:
            continue
    return {"jobs": [_normalize_job_payload(job) for job in visible_jobs]}


@app.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    dataset_id = str(job.get("dataset_id") or "")
    if dataset_id:
        _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return _normalize_job_payload(job)


@app.get("/sessions/{dataset_id}/export/{format}")
def export_dataset(
    dataset_id: str,
    format: str,
    limit: int = Query(default=5000, ge=1, le=50000),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
    actor: str | None = Query(default=None),
    role: str | None = Query(default=None),
) -> Response:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    format = format.lower()
    profile = _load_profile_if_exists(dataset_id)
    if profile is None and meta.get("file"):
        profile = _build_profile(_read_uploaded_file(meta))
        _save_json(_profile_path(dataset_id), profile)
        meta["artifacts"]["profile"] = str(_profile_path(dataset_id))
        _save_meta(dataset_id, meta)
    aliases = _pii_aliases(profile or {})
    pii_detected = bool(aliases)
    export_action = "export_sensitive" if format in {"csv", "json"} and pii_detected and _can_export_sensitive(meta) else "export_masked"
    _, actor, _ = _authorized_session_context(
        dataset_id,
        action=export_action,
        x_api_key=x_api_key,
        x_user_role=x_user_role,
        actor_query=actor,
        role_query=role,
    )

    if format in {"html", "report"}:
        path = _report_path(dataset_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Report not generated yet.")
        _append_audit(dataset_id, "export_html", actor, {"format": "html"})
        return HTMLResponse(path.read_text(encoding="utf-8"))

    if format == "pdf":
        path = _report_pdf_path(dataset_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Report PDF not generated yet.")
        _append_audit(dataset_id, "export_pdf", actor, {"format": "pdf"})
        return Response(content=path.read_bytes(), media_type="application/pdf")

    if format == "dashboard":
        spec_path = _dashboard_spec_path(dataset_id)
        if not spec_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard spec not generated yet.")
        spec = _load_json(spec_path)
        _append_audit(dataset_id, "export_dashboard", actor, {"format": "dashboard"})
        return HTMLResponse(_render_dashboard_html(spec))

    if format == "json":
        facts_path = _facts_path(dataset_id)
        if not facts_path.exists():
            raise HTTPException(status_code=404, detail="Facts bundle not generated yet.")
        facts_bundle = _load_json(facts_path)
        masked_export = pii_detected and not _can_export_sensitive(meta)
        if pii_detected and not _can_export_sensitive(meta):
            facts_bundle = _mask_recursive(facts_bundle, aliases)
        _append_audit(dataset_id, "export_json", actor, {"format": "json", "masked": masked_export})
        return Response(content=json.dumps(facts_bundle), media_type="application/json")

    if format == "csv":
        cleaned_path = _cleaned_path(dataset_id)
        if cleaned_path.exists():
            df = pd.read_csv(cleaned_path).head(limit)
        else:
            if not meta.get("file"):
                raise HTTPException(status_code=404, detail="No dataset available.")
            df = _read_uploaded_file(meta).head(limit)
        masked_export = pii_detected and not _can_export_sensitive(meta)
        if pii_detected and not _can_export_sensitive(meta) and aliases:
            df = df.rename(columns=aliases)
        _append_audit(dataset_id, "export_csv", actor, {"format": "csv", "masked": masked_export})
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        return Response(content=stream.getvalue(), media_type="text/csv")

    raise HTTPException(status_code=400, detail="Unsupported export format. Use html, pdf, dashboard, json, or csv.")


@app.get("/sessions/{dataset_id}/audit")
def get_audit_log(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> dict[str, Any]:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    path = _audit_log_path(dataset_id)
    if not path.exists():
        return {"dataset_id": dataset_id, "events": []}

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"dataset_id": dataset_id, "events": events}


@app.get("/sessions/{dataset_id}/feedback", response_model=FeedbackListResponse)
def get_feedback(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> FeedbackListResponse:
    _authorized_session_context(dataset_id, action="read", x_api_key=x_api_key, x_user_role=x_user_role)
    return FeedbackListResponse(
        dataset_id=dataset_id,
        feedback=[FeedbackRecord(**item) for item in _load_feedback(dataset_id)],
    )


@app.post("/sessions/{dataset_id}/feedback", response_model=FeedbackRecord)
def submit_feedback(
    dataset_id: str,
    payload: FeedbackRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> FeedbackRecord:
    meta, actor, _ = _authorized_session_context(dataset_id, action="write", x_api_key=x_api_key, x_user_role=x_user_role)
    normalized = _normalize_feedback_request(payload)
    record = {
        "feedback_id": str(uuid.uuid4()),
        "surface": normalized.surface,
        "target_id": normalized.target_id,
        "rating": normalized.rating,
        "question": normalized.question,
        "title": normalized.title,
        "comment": normalized.comment,
        "created_at": _utc_now_iso(),
        "created_by": actor,
    }
    items = [record, *_load_feedback(dataset_id)]
    _save_feedback(dataset_id, items)
    meta.setdefault("artifacts", {})["feedback"] = str(_feedback_path(dataset_id))
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "feedback_recorded",
        actor,
        {"surface": normalized.surface, "target_id": normalized.target_id, "rating": normalized.rating},
    )
    return FeedbackRecord(**record)
