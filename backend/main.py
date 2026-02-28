from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Response as FastAPIResponse, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from backend.api import router as ai_analytics_router
from backend.cache import CacheManager
from backend.jobs import create_job, get_job, list_jobs, update_job
from backend.llm_client import LLMClient, create_llm_client_from_env
from backend.llm_gate import FactsGroundingError, SchemaValidationError, validate_facts_references, validate_schema
from backend.llm_schemas import ASK_NARRATIVE_SCHEMA, DASHBOARD_SPEC_SCHEMA, FACTS_BUNDLE_SCHEMA, QUERY_PLAN_SCHEMA

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


class CreateSessionResponse(BaseModel):
    dataset_id: str
    created_at: str


class SessionMetaResponse(BaseModel):
    dataset_id: str
    status: str
    created_at: str
    updated_at: str
    created_by: str
    user_id: str | None = None
    pii_masking_enabled: bool
    allow_sensitive_export: bool = False
    file: dict[str, Any] | None = None
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


class ReportResponse(BaseModel):
    dataset_id: str
    report_html_path: str
    report_html: str


class MaskingRequest(BaseModel):
    enabled: bool


class SensitiveExportRequest(BaseModel):
    enabled: bool


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


class ReportRequest(BaseModel):
    template: str | None = "health_report"
    sections: list[str] = Field(default_factory=lambda: ["quality", "kpis", "trends", "limitations"])


class DashboardSpecRequest(BaseModel):
    template: str = "health_core"


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


def _profile_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "profile.json"


def _facts_path(dataset_id: str) -> Path:
    return _dataset_path(dataset_id) / "facts.json"


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
        return df, "none"

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


def _load_profile_if_exists(dataset_id: str) -> dict[str, Any] | None:
    path = _profile_path(dataset_id)
    if path.exists():
        return _load_json(path)
    return None


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


def _generate_query_plan_llm(question: str, facts_bundle: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    del facts_bundle  # facts are used later for grounding the narrative, not for plan schema.
    llm = _get_llm_client()
    context = {
        "question": question,
        "columns": [str(name) for name in profile.get("dtypes", {}).keys()][:120],
        "numeric_columns": [str(name) for name in profile.get("numeric_cols", [])][:60],
        "categorical_columns": [str(name) for name in profile.get("categorical_cols", [])][:60],
        "datetime_columns": [str(name) for name in profile.get("datetime_cols", [])][:30],
    }
    system_prompt = (
        "You are a query planner for healthcare analytics. "
        "Generate a safe query plan using only provided columns."
    )
    user_prompt = f"Build a query plan for this context:\n{json.dumps(context, ensure_ascii=True)}"
    output = llm.generate_json(QUERY_PLAN_SCHEMA, system_prompt, user_prompt, timeout=30)
    validate_schema(output, QUERY_PLAN_SCHEMA)
    return output


def _fallback_query_plan(question: str, profile: dict[str, Any]) -> dict[str, Any]:
    numeric_cols = [str(name) for name in profile.get("numeric_cols", [])]
    categorical_cols = [str(name) for name in profile.get("categorical_cols", [])]
    datetime_cols = [str(name) for name in profile.get("datetime_cols", [])]

    metric_field = numeric_cols[0] if numeric_cols else (categorical_cols[0] if categorical_cols else "")
    metric_op = "sum" if numeric_cols else "count"
    group_by: list[str] = []
    chart_hint = "table"
    intent = "aggregate"

    if datetime_cols and any(token in question.lower() for token in ("trend", "time", "month", "week", "year")):
        group_by = [datetime_cols[0]]
        chart_hint = "line"
        intent = "trend"
    elif categorical_cols:
        group_by = [categorical_cols[0]]
        chart_hint = "bar"
        intent = "compare"

    return {
        "intent": intent,
        "metrics": [{"op": metric_op, "field": metric_field}] if metric_field else [],
        "group_by": group_by,
        "filters": [],
        "time": {
            "field": datetime_cols[0] if datetime_cols else None,
            "grain": "month" if datetime_cols else None,
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


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    payload: CreateSessionRequest | None = None, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> CreateSessionResponse:
    created_by = _actor_from_header(x_api_key, fallback=(payload.created_by if payload else "anonymous"))
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
        "pii_masking_enabled": False,
        "allow_sensitive_export": False,
        "file": None,
        "artifacts": {},
    }
    _save_json(_meta_path(dataset_id), meta)
    _append_audit(dataset_id, "session_created", created_by, {})
    return CreateSessionResponse(dataset_id=dataset_id, created_at=now)


@app.get("/sessions/{dataset_id}", response_model=SessionMetaResponse)
def get_session(dataset_id: str) -> SessionMetaResponse:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    return SessionMetaResponse(
        dataset_id=meta["dataset_id"],
        status=meta.get("status", "unknown"),
        created_at=meta.get("created_at", ""),
        updated_at=meta.get("updated_at", ""),
        created_by=meta.get("created_by", "anonymous"),
        user_id=meta.get("user_id"),
        pii_masking_enabled=bool(meta.get("pii_masking_enabled", False)),
        allow_sensitive_export=bool(meta.get("allow_sensitive_export", False)),
        file=meta.get("file"),
        artifacts=meta.get("artifacts", {}),
    )


@app.post("/sessions/{dataset_id}/masking")
def update_masking(
    dataset_id: str, payload: MaskingRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    meta["pii_masking_enabled"] = bool(payload.enabled)
    _save_meta(dataset_id, meta)
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
    _append_audit(dataset_id, "masking_toggled", actor, {"enabled": payload.enabled})
    return {"dataset_id": dataset_id, "pii_masking_enabled": meta["pii_masking_enabled"]}


@app.post("/sessions/{dataset_id}/sensitive-export")
def update_sensitive_export(
    dataset_id: str, payload: SensitiveExportRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    meta["allow_sensitive_export"] = bool(payload.enabled)
    _save_meta(dataset_id, meta)
    _append_audit(
        dataset_id,
        "sensitive_export_toggled",
        _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous")),
        {"enabled": payload.enabled},
    )
    return {"dataset_id": dataset_id, "allow_sensitive_export": meta["allow_sensitive_export"]}


@app.post("/sessions/{dataset_id}/upload")
async def upload_file(
    dataset_id: str,
    file: UploadFile = File(...),
    sheet_name: str | None = Query(default=None, description="Excel sheet to load. Defaults to first sheet."),
    uploaded_by: str = Form(default="anonymous"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    session_dir = _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)

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
    actor = _actor_from_header(x_api_key, fallback=uploaded_by.strip() or meta.get("created_by", "anonymous"))
    file_meta["uploaded_at"] = _utc_now_iso()
    file_meta["uploaded_by"] = actor

    # New upload invalidates derived artifacts.
    meta["file"] = file_meta
    meta["status"] = "uploaded"
    meta["artifacts"] = {}
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
    mask_pii: bool | None = Query(default=None),
    response: FastAPIResponse | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ProfileResponse:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
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
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
    _append_audit(dataset_id, "profile_generated", actor, {})

    if response is not None:
        response.headers["ETag"] = _etag_for(dataset_id, "profile", file_hash, schema_hash=schema_hash)

    mask_enabled = bool(meta.get("pii_masking_enabled", False)) if mask_pii is None else bool(mask_pii)
    response_profile = _apply_profile_masking(profile, mask_enabled)
    return ProfileResponse(dataset_id=dataset_id, profile=response_profile)


@app.get("/sessions/{dataset_id}/facts")
def generate_facts(
    dataset_id: str,
    mode: str = Query(default="auto"),
    force: bool = Query(default=False),
    response: FastAPIResponse | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
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
            actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
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
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
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
) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
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
        actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
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
    dataset_id: str, payload: CleanRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
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
        _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous")),
        {"actions": payload.actions, "pii_mask": payload.pii_mask},
    )

    return {"dataset_id": dataset_id, "status": "cleaned", "cleaned_path": str(cleaned_path)}


@app.get("/sessions/{dataset_id}/dashboard-spec", response_model=DashboardSpecResponse)
@app.post("/sessions/{dataset_id}/dashboard-spec", response_model=DashboardSpecResponse)
def generate_dashboard_spec(
    dataset_id: str,
    payload: DashboardSpecRequest | None = None,
    use_llm: bool = Query(default=True),
    response: FastAPIResponse | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> DashboardSpecResponse:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
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
                _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous")),
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
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
    _append_audit(dataset_id, "dashboard_spec_generated", actor, {"use_llm": use_llm, "template": template})
    if response is not None:
        response.headers["ETag"] = _etag_for(
            dataset_id, "dashboard_spec", file_hash, schema_hash=schema_hash, params=str(use_llm)
        )
    return DashboardSpecResponse(dataset_id=dataset_id, dashboard_spec=spec)


@app.get("/sessions/{dataset_id}/dashboard")
def get_dashboard(dataset_id: str) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    spec_path = _dashboard_spec_path(dataset_id)
    if not spec_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard spec not generated yet.")
    spec = _load_json(spec_path)
    return {"dataset_id": dataset_id, "dashboard_html": _render_dashboard_html(spec), "spec_version": spec.get("version")}


@app.post("/sessions/{dataset_id}/ask", response_model=AskResponse)
def ask_dataset(
    dataset_id: str, payload: AskRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> AskResponse:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
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

    try:
        query_plan = _generate_query_plan_llm(payload.question, facts_bundle, profile)
    except Exception:
        query_plan = _fallback_query_plan(payload.question, profile)

    try:
        validate_schema(query_plan, QUERY_PLAN_SCHEMA)
    except SchemaValidationError as exc:
        _append_audit(dataset_id, "ask_data", actor, {"question": payload.question, "status": "invalid_plan"})
        raise HTTPException(status_code=422, detail=f"Invalid query plan: {exc}") from exc

    result_df, execution_note = _execute_query_plan(df, query_plan)
    candidate_fact_ids = _default_fact_ids(facts_bundle)

    try:
        answer, facts_used = _summarize_query_result_llm(payload.question, result_df, facts_bundle, candidate_fact_ids)
    except Exception:
        fallback_answer, fallback_facts, _, _, _ = _safe_answer_from_facts(payload.question, facts_bundle)
        answer = f"{fallback_answer} ({execution_note})"
        facts_used = fallback_facts or candidate_fact_ids

    _append_audit(
        dataset_id,
        "ask_data",
        actor,
        {"question": payload.question, "facts_used": facts_used, "status": "ok"},
    )

    data_coverage = (facts_bundle.get("data_coverage") or {}).get("mode", "sample")
    confidence = "Medium" if facts_used else "Low"
    fact_coverage = 1.0 if facts_used else 0.0
    result_rows = json.loads(result_df.head(min(int(query_plan.get("limit", 25)), 100)).to_json(orient="records", date_format="iso"))
    chart_type = query_plan.get("chart_hint", "table")
    columns = list(result_df.columns)
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
    )


@app.get("/sessions/{dataset_id}/preview")
def preview_dataset(dataset_id: str, limit: int = Query(default=1000, ge=1, le=5000)) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
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
    dataset_id: str, payload: ReportRequest | None = None, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> dict[str, Any]:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    if not meta.get("file"):
        raise HTTPException(status_code=400, detail="No file uploaded for this session.")

    payload = payload or ReportRequest()
    job = create_job(
        job_type="report",
        dataset_id=dataset_id,
        payload={"template": payload.template, "sections": payload.sections},
    )

    try:
        from backend.tasks import generate_report_task

        generate_report_task.delay(job["job_id"], dataset_id, payload.template, payload.sections)
        actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
        _append_audit(dataset_id, "report_job_queued", actor, {"job_id": job["job_id"]})
    except Exception as exc:
        try:
            generate_report_task(job["job_id"], dataset_id, payload.template, payload.sections)
        except Exception as inner_exc:
            update_job(job["job_id"], status="failed", error=str(inner_exc))
            raise HTTPException(status_code=503, detail=f"Failed to run report job: {exc}; {inner_exc}") from inner_exc

    return Response(
        content=json.dumps({"dataset_id": dataset_id, "job_id": job["job_id"], "status": job["status"]}),
        media_type="application/json",
        status_code=202,
    )


@app.get("/sessions/{dataset_id}/report/html")
def get_report_html(dataset_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> HTMLResponse:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    path = _report_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
    _append_audit(dataset_id, "export_html", actor, {"format": "html"})
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/sessions/{dataset_id}/report/pdf")
def get_report_pdf(dataset_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Response:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    path = _report_pdf_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report PDF not generated yet.")
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))
    _append_audit(dataset_id, "export_pdf", actor, {"format": "pdf"})
    return Response(content=path.read_bytes(), media_type="application/pdf")


@app.get("/sessions/{dataset_id}/export/pdf")
def export_pdf_direct(dataset_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Response:
    return get_report_pdf(dataset_id, x_api_key=x_api_key)


@app.get("/jobs")
def list_all_jobs(dataset_id: str | None = Query(default=None)) -> dict[str, Any]:
    return {"jobs": [_normalize_job_payload(job) for job in list_jobs(dataset_id=dataset_id)]}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _normalize_job_payload(job)


@app.get("/sessions/{dataset_id}/export/{format}")
def export_dataset(
    dataset_id: str,
    format: str,
    limit: int = Query(default=5000, ge=1, le=50000),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Response:
    _require_session_dir(dataset_id)
    meta = _load_meta(dataset_id)
    format = format.lower()
    profile = _load_profile_if_exists(dataset_id)
    aliases = _pii_aliases(profile or {})
    pii_detected = bool(aliases)
    actor = _actor_from_header(x_api_key, fallback=meta.get("user_id") or meta.get("created_by", "anonymous"))

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
        if pii_detected and not _can_export_sensitive(meta):
            facts_bundle = _mask_recursive(facts_bundle, aliases)
        _append_audit(dataset_id, "export_json", actor, {"format": "json"})
        return Response(content=json.dumps(facts_bundle), media_type="application/json")

    if format == "csv":
        cleaned_path = _cleaned_path(dataset_id)
        if cleaned_path.exists():
            df = pd.read_csv(cleaned_path).head(limit)
        else:
            if not meta.get("file"):
                raise HTTPException(status_code=404, detail="No dataset available.")
            df = _read_uploaded_file(meta).head(limit)
        if pii_detected and not _can_export_sensitive(meta) and aliases:
            df = df.rename(columns=aliases)
        _append_audit(dataset_id, "export_csv", actor, {"format": "csv"})
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        return Response(content=stream.getvalue(), media_type="text/csv")

    raise HTTPException(status_code=400, detail="Unsupported export format. Use html, pdf, dashboard, json, or csv.")


@app.get("/sessions/{dataset_id}/audit")
def get_audit_log(dataset_id: str) -> dict[str, Any]:
    _require_session_dir(dataset_id)
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
