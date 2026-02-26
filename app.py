from __future__ import annotations

import io
import os
import time
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120

st.set_page_config(page_title="AI Analytics Dashboard Generator", layout="wide")
st.markdown(
    """
    <style>
      :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --ink: #0f172a;
        --muted: #64748b;
        --accent: #0ea5e9;
        --accent-soft: #e0f2fe;
        --border: #e2e8f0;
      }
      .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
      .stApp { background: var(--bg); color: var(--ink); }
      .app-title { font-size: 2.0rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
      .app-subtitle { color: var(--muted); margin-bottom: 1.2rem; }
      .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
        animation: fadeUp 420ms ease-out;
      }
      .status-rail {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 0.5rem;
        margin: 0.8rem 0 1.2rem 0;
      }
      .status-pill {
        padding: 0.45rem 0.6rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #ffffff;
        font-size: 0.75rem;
        text-align: center;
        color: var(--muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .status-pill.done {
        background: var(--accent-soft);
        color: var(--accent);
        border-color: #bae6fd;
        font-weight: 600;
      }
      .status-pill.active {
        background: #ecfdf3;
        border-color: #a7f3d0;
        color: #047857;
        font-weight: 600;
      }
      .section-header { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.6rem; }
      .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.8rem;
        margin-left: 0.35rem;
      }
      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      .stDataFrame, .stJson { border-radius: 12px; border: 1px solid var(--border); }
      @media (max-width: 768px) {
        .app-title { font-size: 1.5rem; }
        .section-card { padding: 0.85rem; }
        .status-rail { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _query_param_value(key: str) -> str | None:
    try:
        value = st.query_params.get(key)
        if isinstance(value, list):
            return value[0] if value else None
        if value is None:
            return None
        return str(value)
    except Exception:
        return None


def _set_query_param(key: str, value: str) -> None:
    try:
        st.query_params[key] = value
    except Exception:
        pass


def _api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    response = requests.request(method=method, url=url, timeout=REQUEST_TIMEOUT, **kwargs)
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"{response.status_code}: {detail}")
    if response.content:
        return response.json()
    return {}


def _ensure_dataset_session() -> str:
    if "dataset_id" in st.session_state and st.session_state.dataset_id:
        _set_query_param("dataset_id", st.session_state.dataset_id)
        return st.session_state.dataset_id

    query_dataset_id = _query_param_value("dataset_id")
    if query_dataset_id:
        try:
            _api_request("GET", f"/sessions/{query_dataset_id}")
            st.session_state.dataset_id = query_dataset_id
            _set_query_param("dataset_id", query_dataset_id)
            return query_dataset_id
        except Exception:
            st.warning("`dataset_id` in URL was not found. A new session was created.")

    created = _api_request("POST", "/sessions", json={"created_by": "streamlit_user"})
    dataset_id = created["dataset_id"]
    st.session_state.dataset_id = dataset_id
    _set_query_param("dataset_id", dataset_id)
    return dataset_id


def _load_session_meta(dataset_id: str) -> dict[str, Any]:
    return _api_request("GET", f"/sessions/{dataset_id}")


def _render_kpis(kpis: list[dict[str, Any]]) -> None:
    if not kpis:
        st.info("No KPIs available.")
        return

    columns = st.columns(len(kpis))
    for column, kpi in zip(columns, kpis):
        column.metric(label=kpi["label"], value=kpi["value"])


def _render_chart(df: pd.DataFrame, chart: dict[str, Any]) -> None:
    chart_type = chart.get("type")
    source = chart.get("source", {})
    title = chart.get("title", chart_type)

    if chart_type == "line":
        dt_col = source.get("datetime_column")
        val_col = source.get("value_column")
        if dt_col not in df.columns or val_col not in df.columns:
            st.info(f"Skipping '{title}': required columns are missing.")
            return
        scoped = df[[dt_col, val_col]].copy()
        scoped[dt_col] = pd.to_datetime(scoped[dt_col], errors="coerce")
        scoped[val_col] = pd.to_numeric(scoped[val_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            st.info(f"Skipping '{title}': no valid values.")
            return
        grouped = (
            scoped.set_index(dt_col)[val_col]
            .resample(source.get("resample", "MS"))
            .agg(source.get("aggregation", "sum"))
            .reset_index()
        )
        figure = px.line(grouped, x=dt_col, y=val_col, title=title, markers=True)
        st.plotly_chart(figure, use_container_width=True)
        return

    if chart_type == "bar":
        cat_col = source.get("category_column")
        val_col = source.get("value_column")
        if cat_col not in df.columns or val_col not in df.columns:
            st.info(f"Skipping '{title}': required columns are missing.")
            return
        scoped = df[[cat_col, val_col]].copy()
        scoped[val_col] = pd.to_numeric(scoped[val_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            st.info(f"Skipping '{title}': no valid values.")
            return
        grouped = (
            scoped.groupby(cat_col, dropna=False)[val_col]
            .agg(source.get("aggregation", "mean"))
            .reset_index()
            .sort_values(val_col, ascending=False)
            .head(int(source.get("top_n", 10)))
        )
        figure = px.bar(grouped, x=cat_col, y=val_col, title=title)
        st.plotly_chart(figure, use_container_width=True)
        return

    if chart_type == "bar_count":
        cat_col = source.get("category_column")
        if cat_col not in df.columns:
            st.info(f"Skipping '{title}': required columns are missing.")
            return
        grouped = (
            df[cat_col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(int(source.get("top_n", 10)))
            .rename_axis(cat_col)
            .reset_index(name="count")
        )
        figure = px.bar(grouped, x=cat_col, y="count", title=title)
        st.plotly_chart(figure, use_container_width=True)
        return

    if chart_type == "histogram":
        val_col = source.get("value_column")
        if val_col not in df.columns:
            st.info(f"Skipping '{title}': required columns are missing.")
            return
        scoped = pd.to_numeric(df[val_col], errors="coerce").dropna()
        if scoped.empty:
            st.info(f"Skipping '{title}': no valid values.")
            return
        histogram_df = pd.DataFrame({val_col: scoped})
        figure = px.histogram(histogram_df, x=val_col, nbins=int(source.get("bins", 30)), title=title)
        st.plotly_chart(figure, use_container_width=True)
        return

    if chart_type == "scatter":
        x_col = source.get("x_column")
        y_col = source.get("y_column")
        if x_col not in df.columns or y_col not in df.columns:
            st.info(f"Skipping '{title}': required columns are missing.")
            return
        scoped = df[[x_col, y_col]].copy()
        scoped[x_col] = pd.to_numeric(scoped[x_col], errors="coerce")
        scoped[y_col] = pd.to_numeric(scoped[y_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            st.info(f"Skipping '{title}': no valid values.")
            return
        figure = px.scatter(scoped, x=x_col, y=y_col, title=title)
        st.plotly_chart(figure, use_container_width=True)
        return

    st.info(f"Unsupported chart type '{chart_type}' for '{title}'.")


st.markdown('<div class="app-title">AI Analytics -> Auto Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Healthcare-ready MVP with facts-first AI</div>', unsafe_allow_html=True)
st.caption(f"Backend API: `{API_BASE_URL}`")

status_steps = [
    ("Session", True),
    ("Upload", bool(session_meta and session_meta.get("file"))),
    ("Profiled", "profile_data" in st.session_state),
    ("Facts Running", "facts_job_id" in st.session_state and "facts_bundle" not in st.session_state),
    ("Facts Ready", "facts_bundle" in st.session_state),
    ("Report Ready", "report_html" in st.session_state),
    ("Audit", True),
]

active_index = 0
for idx, (_, done) in enumerate(status_steps):
    if not done:
        active_index = idx
        break
else:
    active_index = len(status_steps) - 1

status_html = '<div class="status-rail">'
for idx, (name, done) in enumerate(status_steps):
    if done:
        status_class = "status-pill done"
    elif idx == active_index:
        status_class = "status-pill active"
    else:
        status_class = "status-pill"
    status_html += f'<div class="{status_class}">{name}</div>'
status_html += "</div>"
st.markdown(status_html, unsafe_allow_html=True)

def _estimate_timing(rows: int, cols: int) -> tuple[str, str, str]:
    if rows <= 100_000 and cols <= 50:
        return ("Small", "1–3s", "5–10s")
    if rows <= 5_000_000 and cols <= 200:
        return ("Mid-sized", "5–15s", "15–45s")
    return ("Large", "Queued (sampling)", "Queued (async)")

if session_meta and session_meta.get("file"):
    try:
        preview = _api_request("GET", f"/sessions/{dataset_id}/preview", params={"limit": 200})
        row_count = int(preview.get("row_count", 0))
        col_count = len(preview.get("columns", []))
        size_label, profiling_eta, report_eta = _estimate_timing(row_count, col_count)
        if row_count > 5_000_000:
            coverage_note = "Sampling only (full compute queued)"
        elif row_count > 0:
            coverage_note = "Full dataset"
        else:
            coverage_note = "Unknown"
        st.info(
            f"Dataset size: {size_label} | Rows: {row_count:,} | Columns: {col_count} "
            f"| Profiling ETA: {profiling_eta} | Report ETA: {report_eta} "
            f"| Coverage: {coverage_note}"
        )
        if coverage_note.startswith("Sampling"):
            if st.button("Request full computation (async)"):
                try:
                    job = _api_request("POST", f"/sessions/{dataset_id}/facts", params={"mode": "full"})
                    st.session_state.facts_job_id = job.get("job_id")
                    st.success(f"Full compute job queued: {st.session_state.facts_job_id}")
                except Exception as exc:
                    st.error(f"Failed to queue full compute: {exc}")
    except Exception:
        pass

dataset_id = _ensure_dataset_session()

with st.sidebar:
    st.subheader("Session")
    st.code(dataset_id)
    existing_id = st.text_input("Load existing dataset_id", key="existing_dataset_id")
    if st.button("Load session"):
        if existing_id.strip():
            try:
                _api_request("GET", f"/sessions/{existing_id.strip()}")
                st.session_state.dataset_id = existing_id.strip()
                _set_query_param("dataset_id", st.session_state.dataset_id)
                st.success("Session loaded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        else:
            st.error("Enter a dataset_id.")

    if st.button("Create new session"):
        created = _api_request("POST", "/sessions", json={"created_by": "streamlit_user"})
        st.session_state.dataset_id = created["dataset_id"]
        _set_query_param("dataset_id", st.session_state.dataset_id)
        st.success("New session created.")
        st.rerun()

session_meta: dict[str, Any] | None = None
try:
    session_meta = _load_session_meta(dataset_id)
except Exception as exc:
    st.error(f"Failed to load session metadata: {exc}")

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">1) Upload dataset (CSV or XLSX)</div>', unsafe_allow_html=True)
uploaded_by = st.text_input("Uploaded by", value="streamlit_user")
uploaded_file = st.file_uploader(
    "Drag and drop or browse files",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
)

selected_sheet: str | None = None
file_bytes: bytes | None = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".xlsx"):
        try:
            workbook = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
            if workbook.sheet_names:
                selected_sheet = st.selectbox("Excel sheet", workbook.sheet_names, index=0)
        except Exception as exc:
            st.warning(f"Could not read workbook sheet names in UI: {exc}")

if st.button("Upload file", disabled=uploaded_file is None):
    with st.spinner("Uploading file..."):
        try:
            params = {}
            if selected_sheet:
                params["sheet_name"] = selected_sheet
            files = {
                "file": (
                    uploaded_file.name,  # type: ignore[union-attr]
                    file_bytes,  # type: ignore[arg-type]
                    uploaded_file.type or "application/octet-stream",  # type: ignore[union-attr]
                )
            }
            data = {"uploaded_by": uploaded_by}
            response = _api_request(
                "POST",
                f"/sessions/{dataset_id}/upload",
                params=params,
                files=files,
                data=data,
            )
            st.success(f"Uploaded: {response['filename']}")
            st.session_state.pop("profile_data", None)
            st.session_state.pop("facts_bundle", None)
            st.session_state.pop("dashboard_spec", None)
            st.session_state.pop("preview_rows", None)
            st.session_state.pop("report_html", None)
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">2) Data profiling and quality checks</div>', unsafe_allow_html=True)
mask_pii_default = bool(session_meta.get("pii_masking_enabled", False)) if session_meta else False
mask_pii = st.toggle("Mask PII field names in API responses", value=mask_pii_default)
allow_sensitive_default = bool(session_meta.get("allow_sensitive_export", False)) if session_meta else False
allow_sensitive = st.toggle("Allow sensitive exports (role/flag)", value=allow_sensitive_default)

if session_meta and mask_pii != mask_pii_default:
    try:
        _api_request("POST", f"/sessions/{dataset_id}/masking", json={"enabled": mask_pii})
        st.success("Masking preference updated.")
        session_meta = _load_session_meta(dataset_id)
    except Exception as exc:
        st.error(f"Could not update masking preference: {exc}")

if session_meta and allow_sensitive != allow_sensitive_default:
    try:
        _api_request("POST", f"/sessions/{dataset_id}/sensitive-export", json={"enabled": allow_sensitive})
        st.success("Sensitive export preference updated.")
        session_meta = _load_session_meta(dataset_id)
    except Exception as exc:
        st.error(f"Could not update sensitive export preference: {exc}")

if st.button("Run profiling"):
    with st.spinner("Profiling dataset..."):
        try:
            profile_response = _api_request(
                "GET",
                f"/sessions/{dataset_id}/profile",
                params={"mask_pii": str(mask_pii).lower()},
            )
            st.session_state.profile_data = profile_response["profile"]
            st.success("Profile generated.")
        except Exception as exc:
            st.error(f"Profiling failed: {exc}")

profile_data = st.session_state.get("profile_data")
if profile_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", profile_data["shape"]["rows"])
    c2.metric("Columns", profile_data["shape"]["cols"])
    c3.metric("Duplicate rows", profile_data["duplicate_rows"])
    c4.metric("Quality score", profile_data["quality_score"])

    if profile_data.get("pii_candidates"):
        st.warning(f"PII candidates detected: {profile_data['pii_candidates']}")

    if profile_data.get("health_template", {}).get("name") == "hmis":
        fields = profile_data["health_template"].get("matched_fields", [])
        st.info(f"Healthcare template detected (HMIS-like fields): {fields}")

    st.write("Column profile")
    st.dataframe(pd.DataFrame(profile_data.get("columns", [])), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">3) Facts bundle and no-hallucination insights</div>', unsafe_allow_html=True)
if st.button("Generate facts bundle"):
    with st.spinner("Computing deterministic facts..."):
        try:
            facts_response = _api_request("GET", f"/sessions/{dataset_id}/facts")
            if facts_response.get("queued"):
                st.session_state.facts_job_id = facts_response.get("job_id")
                st.warning("Facts job queued. Check status below.")
            else:
                st.session_state.facts_bundle = facts_response["facts_bundle"]
                st.success("Facts generated.")
        except Exception as exc:
            st.error(f"Facts generation failed: {exc}")

facts_bundle = st.session_state.get("facts_bundle")
facts_job_id = st.session_state.get("facts_job_id")
if facts_job_id:
    st.caption(f"Facts job: {facts_job_id}")
    facts_auto_refresh = st.toggle("Auto-refresh facts job (every 5s)", value=False, key="auto_refresh_facts")
    if st.button("Check facts job status") or facts_auto_refresh:
        try:
            status = _api_request("GET", f"/jobs/{facts_job_id}")
            st.write(status)
            if status.get("progress") is not None:
                st.progress(float(status.get("progress", 0.0)))
            if status.get("status") == "completed":
                facts_response = _api_request("GET", f"/sessions/{dataset_id}/facts")
                st.session_state.facts_bundle = facts_response.get("facts_bundle")
                st.session_state.pop("facts_job_id", None)
                st.success("Facts ready.")
            elif facts_auto_refresh and status.get("status") not in {"completed", "failed"}:
                time.sleep(5)
                st.rerun()
        except Exception as exc:
            st.error(f"Facts job check failed: {exc}")
if facts_bundle:
    st.caption(f"Policy: {facts_bundle.get('generation_policy')}")
    coverage = facts_bundle.get("data_coverage", {})
    if coverage:
        st.info(
            f"Data coverage: {coverage.get('mode')} | "
            f"Rows used: {coverage.get('rows_used')} of {coverage.get('rows_total')} | "
            f"Sampling: {coverage.get('sampling_method')} | Seed: {coverage.get('seed')}"
        )
    st.write("Facts")
    st.dataframe(pd.DataFrame(facts_bundle.get("facts", [])), use_container_width=True)
    st.write("Insights")
    for insight in facts_bundle.get("insights", []):
        st.markdown(f"**{insight['title']}**")
        st.write(insight["statement"])
        st.caption(f"Citations: {', '.join(insight.get('citations', []))}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">4) Ask-Your-Data (safe, facts-only)</div>', unsafe_allow_html=True)
question = st.text_input("Ask a question about the dataset", value="What changed most recently in the trend?")
if st.button("Ask"):
    with st.spinner("Answering from facts..."):
        try:
            response = _api_request("POST", f"/sessions/{dataset_id}/ask", json={"question": question, "mode": "safe"})
            st.write(response.get("answer"))
            st.caption(f"Facts used: {', '.join(response.get('facts_used', [])) or 'None'}")
            st.caption(
                f"Confidence: {response.get('confidence')} | "
                f"Fact coverage: {response.get('fact_coverage')} | "
                f"Data coverage: {response.get('data_coverage')}"
            )
        except Exception as exc:
            st.error(f"Ask failed: {exc}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">5) Auto dashboard generation</div>', unsafe_allow_html=True)
if st.button("Generate dashboard spec"):
    with st.spinner("Generating dashboard spec..."):
        try:
            spec_response = _api_request("GET", f"/sessions/{dataset_id}/dashboard-spec")
            st.session_state.dashboard_spec = spec_response["dashboard_spec"]
            st.success("Dashboard spec generated.")
        except Exception as exc:
            st.error(f"Dashboard spec generation failed: {exc}")

dashboard_spec = st.session_state.get("dashboard_spec")
if dashboard_spec:
    _render_kpis(dashboard_spec.get("kpis", []))
    st.write("Dashboard spec JSON")
    st.json(dashboard_spec)

    if st.button("Load preview rows for chart rendering"):
        with st.spinner("Loading preview rows..."):
            try:
                preview_response = _api_request(
                    "GET",
                    f"/sessions/{dataset_id}/preview",
                    params={"limit": 2000},
                )
                st.session_state.preview_rows = preview_response.get("rows", [])
                st.success("Preview rows loaded.")
            except Exception as exc:
                st.error(f"Could not load preview rows: {exc}")

preview_rows = st.session_state.get("preview_rows")
if dashboard_spec and preview_rows:
    st.write("Rendered dashboard")
    preview_df = pd.DataFrame(preview_rows)
    for chart_spec in dashboard_spec.get("charts", []):
        _render_chart(preview_df, chart_spec)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">6) Auto-report export (HTML/PDF via Jobs)</div>', unsafe_allow_html=True)
if st.button("Queue report generation"):
    with st.spinner("Queueing report job..."):
        try:
            queued = _api_request("POST", f"/sessions/{dataset_id}/report", json={})
            st.session_state.report_job_id = queued.get("job_id")
            st.success(f"Report job queued: {st.session_state.report_job_id}")
        except Exception as exc:
            st.error(f"Report queue failed: {exc}")

job_id = st.session_state.get("report_job_id")
auto_refresh = st.toggle("Auto-refresh job status (every 5s)", value=False, key="auto_refresh_job")
check_clicked = st.button("Check report job status")
if job_id and (check_clicked or auto_refresh):
    with st.spinner("Checking job status..."):
        try:
            status = _api_request("GET", f"/jobs/{job_id}")
            st.write(status)
            st.caption(f"Job status: {status.get('status')} | Updated: {status.get('updated_at')}")
            if status.get("progress") is not None:
                st.progress(float(status.get("progress", 0.0)))
            if status.get("status") == "completed":
                response = requests.get(f"{API_BASE_URL}/sessions/{dataset_id}/report/html", timeout=REQUEST_TIMEOUT)
                if response.status_code >= 400:
                    raise RuntimeError(response.text)
                st.session_state.report_html = response.text
                st.success("Report ready.")
            elif auto_refresh and status.get("status") not in {"completed", "failed"}:
                time.sleep(5)
                st.rerun()
        except Exception as exc:
            st.error(f"Job status failed: {exc}")

report_html = st.session_state.get("report_html")
if report_html:
    st.download_button(
        label="Download report.html",
        data=report_html,
        file_name=f"report_{dataset_id}.html",
        mime="text/html",
    )
    if st.button("Download report.pdf"):
        with st.spinner("Fetching PDF..."):
            try:
                pdf_response = requests.get(
                    f"{API_BASE_URL}/sessions/{dataset_id}/report/pdf", timeout=REQUEST_TIMEOUT
                )
                if pdf_response.status_code >= 400:
                    raise RuntimeError(pdf_response.text)
                st.download_button(
                    label="Save report.pdf",
                    data=pdf_response.content,
                    file_name=f"report_{dataset_id}.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:
                st.error(f"PDF download failed: {exc}")
    st.components.v1.html(report_html, height=550, scrolling=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">7) Session metadata and audit log</div>', unsafe_allow_html=True)
if session_meta:
    st.json(session_meta)

if st.button("Refresh audit log"):
    try:
        audit = _api_request("GET", f"/sessions/{dataset_id}/audit")
        events = audit.get("events", [])
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to fetch audit log: {exc}")
st.markdown("</div>", unsafe_allow_html=True)
