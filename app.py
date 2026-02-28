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

st.set_page_config(page_title="Healthcare AI Analytics", layout="wide")
st.title("Healthcare AI Analytics")
st.caption(f"Backend: `{API_BASE_URL}`")


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
    headers = dict(kwargs.pop("headers", {}) or {})
    user_id = st.session_state.get("user_id", "")
    if user_id:
        headers["X-API-Key"] = user_id
        headers["X-User-Id"] = user_id
    response = requests.request(method=method, url=url, timeout=REQUEST_TIMEOUT, headers=headers, **kwargs)
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"{response.status_code}: {detail}")
    if not response.content:
        return {}
    return response.json()


def _create_session() -> str:
    payload = {"created_by": st.session_state.get("user_id") or "streamlit_user"}
    created = _api_request("POST", "/sessions", json=payload)
    dataset_id = str(created["dataset_id"])
    st.session_state["dataset_id"] = dataset_id
    _set_query_param("dataset_id", dataset_id)
    return dataset_id


def _ensure_dataset_id() -> str:
    if st.session_state.get("dataset_id"):
        return str(st.session_state["dataset_id"])
    qp_dataset = _query_param_value("dataset_id")
    if qp_dataset:
        try:
            _api_request("GET", f"/sessions/{qp_dataset}")
            st.session_state["dataset_id"] = qp_dataset
            return qp_dataset
        except Exception:
            pass
    return _create_session()


def _load_session_meta(dataset_id: str) -> dict[str, Any]:
    return _api_request("GET", f"/sessions/{dataset_id}")


def _refresh_cached_artifacts() -> None:
    dataset_id = st.session_state.get("dataset_id")
    if not dataset_id:
        return
    try:
        st.session_state["session_meta"] = _load_session_meta(str(dataset_id))
    except Exception:
        return


if "user_id" not in st.session_state:
    st.session_state["user_id"] = "streamlit_user"

if "last_question" not in st.session_state:
    st.session_state["last_question"] = "Show trend for the main metric over time"

dataset_id = _ensure_dataset_id()
_refresh_cached_artifacts()


def _status_rail(meta: dict[str, Any]) -> None:
    uploaded = bool(meta.get("file"))
    profiled = bool(meta.get("artifacts", {}).get("profile") or st.session_state.get("profile_data"))
    facts_running = bool(st.session_state.get("facts_job_id"))
    facts_ready = bool(meta.get("artifacts", {}).get("facts") or st.session_state.get("facts_bundle"))
    dashboard_ready = bool(meta.get("artifacts", {}).get("dashboard_spec") or st.session_state.get("dashboard_spec"))
    report_ready = bool(meta.get("artifacts", {}).get("report_pdf") or st.session_state.get("report_ready"))

    labels = [
        ("Uploaded", uploaded),
        ("Profiled", profiled),
        ("Facts Running", facts_running),
        ("Facts Ready", facts_ready),
        ("Dashboard Ready", dashboard_ready),
        ("Report Ready", report_ready),
    ]
    cols = st.columns(len(labels))
    for col, (label, done) in zip(cols, labels):
        col.metric(label, "Yes" if done else "No")


with st.sidebar:
    st.subheader("Session")
    st.session_state["user_id"] = st.text_input("User ID", value=st.session_state.get("user_id", "streamlit_user"))
    st.code(dataset_id)

    existing = st.text_input("Load dataset_id", value="")
    if st.button("Load Session") and existing.strip():
        try:
            _api_request("GET", f"/sessions/{existing.strip()}")
            st.session_state["dataset_id"] = existing.strip()
            _set_query_param("dataset_id", existing.strip())
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.button("Create Session"):
        _create_session()
        st.rerun()

    st.divider()
    st.subheader("Upload")
    upload_file = st.file_uploader("CSV/XLSX", type=["csv", "xlsx"])
    sheet_name: str | None = None
    file_bytes: bytes | None = None
    if upload_file is not None:
        file_bytes = upload_file.getvalue()
        if upload_file.name.lower().endswith(".xlsx"):
            try:
                workbook = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
                if workbook.sheet_names:
                    sheet_name = st.selectbox("Sheet", workbook.sheet_names)
            except Exception as exc:
                st.warning(f"Workbook parse warning: {exc}")

    if st.button("Upload File", disabled=upload_file is None):
        try:
            params: dict[str, Any] = {}
            if sheet_name:
                params["sheet_name"] = sheet_name
            files = {
                "file": (
                    upload_file.name,  # type: ignore[union-attr]
                    file_bytes,
                    upload_file.type or "application/octet-stream",  # type: ignore[union-attr]
                )
            }
            data = {"uploaded_by": st.session_state.get("user_id", "streamlit_user")}
            _api_request("POST", f"/sessions/{dataset_id}/upload", params=params, files=files, data=data)
            for key in [
                "profile_data",
                "facts_bundle",
                "dashboard_spec",
                "ask_result",
                "report_ready",
                "facts_job_id",
                "report_job_id",
            ]:
                st.session_state.pop(key, None)
            st.success("Upload complete")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Actions")
    run_profile = st.button("Run Profiling")
    run_facts = st.button("Generate Facts")
    run_dashboard = st.button("Generate Dashboard")
    run_ask = st.button("Ask Data")
    run_report = st.button("Generate Report")

meta = st.session_state.get("session_meta", {})
_status_rail(meta)

if run_profile:
    try:
        resp = _api_request("GET", f"/sessions/{dataset_id}/profile")
        st.session_state["profile_data"] = resp.get("profile", {})
        st.success("Profile ready")
        _refresh_cached_artifacts()
    except Exception as exc:
        st.error(str(exc))

if run_facts:
    try:
        resp = _api_request("GET", f"/sessions/{dataset_id}/facts")
        if "job_id" in resp:
            st.session_state["facts_job_id"] = resp["job_id"]
            st.info(f"Facts job queued: {resp['job_id']}")
        else:
            st.session_state["facts_bundle"] = resp.get("facts_bundle", {})
            st.success("Facts ready")
        _refresh_cached_artifacts()
    except Exception as exc:
        st.error(str(exc))

if run_dashboard:
    try:
        resp = _api_request("POST", f"/sessions/{dataset_id}/dashboard-spec", json={"template": "health_core"}, params={"use_llm": "true"})
        st.session_state["dashboard_spec"] = resp.get("dashboard_spec", {})
        st.success("Dashboard spec ready")
        _refresh_cached_artifacts()
    except Exception as exc:
        st.error(str(exc))

if run_ask:
    try:
        resp = _api_request(
            "POST",
            f"/sessions/{dataset_id}/ask",
            json={"question": st.session_state.get("last_question", ""), "mode": "safe"},
        )
        st.session_state["ask_result"] = resp
        st.success("Ask result ready")
    except Exception as exc:
        st.error(str(exc))

if run_report:
    try:
        resp = _api_request("POST", f"/sessions/{dataset_id}/report", json={})
        if "job_id" in resp:
            st.session_state["report_job_id"] = resp["job_id"]
            st.info(f"Report job queued: {resp['job_id']}")
    except Exception as exc:
        st.error(str(exc))

facts_job_id = st.session_state.get("facts_job_id")
if facts_job_id:
    st.subheader("Facts Job")
    if st.button("Refresh Facts Job"):
        try:
            status = _api_request("GET", f"/jobs/{facts_job_id}")
            st.write(status)
            st.progress(float(status.get("progress", 0)) / 100.0)
            if status.get("status") == "succeeded":
                resp = _api_request("GET", f"/sessions/{dataset_id}/facts")
                st.session_state["facts_bundle"] = resp.get("facts_bundle", {})
                st.session_state.pop("facts_job_id", None)
                _refresh_cached_artifacts()
                st.success("Facts job completed")
            elif status.get("status") == "failed":
                st.error(f"Facts job failed: {status.get('error')}")
        except Exception as exc:
            st.error(str(exc))

report_job_id = st.session_state.get("report_job_id")
if report_job_id:
    st.subheader("Report Job")
    if st.button("Refresh Report Job"):
        try:
            status = _api_request("GET", f"/jobs/{report_job_id}")
            st.write(status)
            st.progress(float(status.get("progress", 0)) / 100.0)
            if status.get("status") == "succeeded":
                st.session_state["report_ready"] = True
                _refresh_cached_artifacts()
                st.success("Report job completed")
            elif status.get("status") == "failed":
                st.error(f"Report job failed: {status.get('error')}")
        except Exception as exc:
            st.error(str(exc))

coverage = (st.session_state.get("facts_bundle", {}) or {}).get("data_coverage", {})
if coverage:
    st.info(
        f"Data coverage: {coverage.get('mode')} | Rows used: {coverage.get('rows_used')} / {coverage.get('rows_total')} | "
        f"Sampling: {coverage.get('sampling_method')}"
    )

def _render_dashboard_chart(df: pd.DataFrame, chart: dict[str, Any]) -> None:
    chart_type = chart.get("type")
    x_col = chart.get("x")
    y_col = chart.get("y")
    agg = chart.get("aggregation", "sum")
    title = chart.get("title", chart_type)

    if chart_type == "line":
        if x_col not in df.columns or y_col not in df.columns:
            st.info(f"Skipping {title}: missing columns")
            return
        scoped = df[[x_col, y_col]].copy()
        scoped[x_col] = pd.to_datetime(scoped[x_col], errors="coerce")
        scoped[y_col] = pd.to_numeric(scoped[y_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            st.info(f"Skipping {title}: empty data")
            return
        grouped = scoped.set_index(x_col)[y_col].resample("MS").agg(agg if agg in {"sum", "mean", "count"} else "sum").reset_index()
        st.plotly_chart(px.line(grouped, x=x_col, y=y_col, title=title, markers=True), use_container_width=True)
        return

    if chart_type == "bar":
        if x_col not in df.columns or y_col not in df.columns:
            st.info(f"Skipping {title}: missing columns")
            return
        scoped = df[[x_col, y_col]].copy()
        scoped[y_col] = pd.to_numeric(scoped[y_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            st.info(f"Skipping {title}: empty data")
            return
        grouped = scoped.groupby(x_col, dropna=False)[y_col].agg(agg if agg in {"sum", "mean", "count"} else "mean").reset_index().head(20)
        st.plotly_chart(px.bar(grouped, x=x_col, y=y_col, title=title), use_container_width=True)
        return

    if chart_type == "hist":
        if x_col not in df.columns:
            st.info(f"Skipping {title}: missing columns")
            return
        scoped = pd.to_numeric(df[x_col], errors="coerce").dropna()
        if scoped.empty:
            st.info(f"Skipping {title}: empty data")
            return
        st.plotly_chart(px.histogram(pd.DataFrame({x_col: scoped}), x=x_col, nbins=30, title=title), use_container_width=True)
        return

    if chart_type == "scatter":
        if x_col not in df.columns or y_col not in df.columns:
            st.info(f"Skipping {title}: missing columns")
            return
        scoped = df[[x_col, y_col]].copy()
        scoped[x_col] = pd.to_numeric(scoped[x_col], errors="coerce")
        scoped[y_col] = pd.to_numeric(scoped[y_col], errors="coerce")
        scoped = scoped.dropna()
        if scoped.empty:
            st.info(f"Skipping {title}: empty data")
            return
        st.plotly_chart(px.scatter(scoped, x=x_col, y=y_col, title=title), use_container_width=True)
        return

    st.info(f"Chart type `{chart_type}` rendered as table preview")


tab_profile, tab_insights, tab_dashboard, tab_ask, tab_report = st.tabs([
    "Profile",
    "Insights",
    "Dashboard",
    "Ask",
    "Report",
])

with tab_profile:
    profile = st.session_state.get("profile_data")
    if not profile and meta.get("artifacts", {}).get("profile"):
        try:
            resp = _api_request("GET", f"/sessions/{dataset_id}/profile")
            profile = resp.get("profile")
            st.session_state["profile_data"] = profile
        except Exception:
            profile = None

    if not profile:
        st.info("Run profiling to populate this tab.")
    else:
        row_cols = st.columns(4)
        row_cols[0].metric("Rows", profile.get("shape", {}).get("rows"))
        row_cols[1].metric("Cols", profile.get("shape", {}).get("cols"))
        row_cols[2].metric("Quality", profile.get("quality_score"))
        row_cols[3].metric("Duplicates", profile.get("duplicate_rows"))
        pii = profile.get("pii_candidates", [])
        if pii:
            st.warning(f"PII candidates: {pii}")
        st.dataframe(pd.DataFrame(profile.get("columns", [])), use_container_width=True)

with tab_insights:
    facts = st.session_state.get("facts_bundle")
    if not facts and meta.get("artifacts", {}).get("facts"):
        try:
            resp = _api_request("GET", f"/sessions/{dataset_id}/facts")
            facts = resp.get("facts_bundle")
            st.session_state["facts_bundle"] = facts
        except Exception:
            facts = None

    if not facts:
        st.info("Generate facts to populate this tab.")
    else:
        for kpi in facts.get("kpis", []):
            st.write(f"**{kpi.get('name')}**: {kpi.get('value')} {kpi.get('unit', '')}")
        st.subheader("Insight Facts")
        st.dataframe(pd.DataFrame(facts.get("insight_facts", [])), use_container_width=True)
        st.subheader("Quality Issues")
        st.dataframe(pd.DataFrame(facts.get("quality", {}).get("issues", [])), use_container_width=True)

with tab_dashboard:
    spec = st.session_state.get("dashboard_spec")
    if not spec and meta.get("artifacts", {}).get("dashboard_spec"):
        try:
            resp = _api_request("GET", f"/sessions/{dataset_id}/dashboard-spec")
            spec = resp.get("dashboard_spec")
            st.session_state["dashboard_spec"] = spec
        except Exception:
            spec = None

    if not spec:
        st.info("Generate dashboard spec to populate this tab.")
    else:
        st.subheader(spec.get("title", "Dashboard"))
        if spec.get("kpis"):
            kpi_cols = st.columns(len(spec["kpis"]))
            facts = st.session_state.get("facts_bundle", {})
            fact_map = {f.get("id"): f for f in facts.get("insight_facts", [])}
            for idx, kpi in enumerate(spec["kpis"]):
                fact = fact_map.get(kpi.get("fact_id"), {})
                kpi_cols[idx].metric(kpi.get("name", "KPI"), str(fact.get("value", {}).get("value", "n/a")))

        preview = _api_request("GET", f"/sessions/{dataset_id}/preview", params={"limit": 2000})
        df_preview = pd.DataFrame(preview.get("rows", []))
        for chart in spec.get("charts", []):
            _render_dashboard_chart(df_preview, chart)

with tab_ask:
    question = st.text_input("Question", value=st.session_state.get("last_question", ""), key="ask_input")
    st.session_state["last_question"] = question
    if st.button("Run Ask Query"):
        try:
            ask = _api_request("POST", f"/sessions/{dataset_id}/ask", json={"question": question, "mode": "safe"})
            st.session_state["ask_result"] = ask
        except Exception as exc:
            st.error(str(exc))

    ask_result = st.session_state.get("ask_result")
    if ask_result:
        st.write(ask_result.get("answer"))
        st.caption(f"Facts used: {', '.join(ask_result.get('facts_used', []))}")
        st.caption(
            f"Confidence: {ask_result.get('confidence')} | "
            f"Fact coverage: {ask_result.get('fact_coverage')} | "
            f"Data coverage: {ask_result.get('data_coverage')}"
        )
        rows = ask_result.get("result_rows", [])
        if rows:
            ask_df = pd.DataFrame(rows)
            st.dataframe(ask_df, use_container_width=True)
            chart = ask_result.get("chart") or {}
            chart_type = chart.get("type", "table")
            if chart_type != "table" and len(ask_df.columns) >= 2:
                x_col = chart.get("x") or ask_df.columns[0]
                y_col = chart.get("y") or ask_df.columns[1]
                if x_col in ask_df.columns and y_col in ask_df.columns:
                    if chart_type == "line":
                        st.plotly_chart(px.line(ask_df, x=x_col, y=y_col, title="Ask Result"), use_container_width=True)
                    elif chart_type == "bar":
                        st.plotly_chart(px.bar(ask_df, x=x_col, y=y_col, title="Ask Result"), use_container_width=True)
                    elif chart_type == "hist":
                        st.plotly_chart(px.histogram(ask_df, x=x_col, title="Ask Result"), use_container_width=True)
                    elif chart_type == "scatter":
                        st.plotly_chart(px.scatter(ask_df, x=x_col, y=y_col, title="Ask Result"), use_container_width=True)

with tab_report:
    st.write("Generate report from facts + dashboard spec, then download PDF.")
    if st.button("Queue Report Job"):
        try:
            resp = _api_request("POST", f"/sessions/{dataset_id}/report", json={})
            st.session_state["report_job_id"] = resp.get("job_id")
            st.success(f"Report queued: {resp.get('job_id')}")
        except Exception as exc:
            st.error(str(exc))

    report_job_id = st.session_state.get("report_job_id")
    if report_job_id and st.button("Refresh Report Status"):
        try:
            status = _api_request("GET", f"/jobs/{report_job_id}")
            st.write(status)
            st.progress(float(status.get("progress", 0)) / 100.0)
            if status.get("status") == "succeeded":
                st.session_state["report_ready"] = True
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("report_ready") or meta.get("artifacts", {}).get("report_pdf"):
        try:
            headers = {}
            if st.session_state.get("user_id"):
                headers["X-API-Key"] = st.session_state["user_id"]
                headers["X-User-Id"] = st.session_state["user_id"]
            pdf_resp = requests.get(f"{API_BASE_URL}/sessions/{dataset_id}/export/pdf", headers=headers, timeout=REQUEST_TIMEOUT)
            if pdf_resp.status_code < 400:
                st.download_button(
                    "Download PDF",
                    data=pdf_resp.content,
                    file_name=f"report_{dataset_id}.pdf",
                    mime="application/pdf",
                )
            else:
                st.warning("PDF not ready yet.")
        except Exception as exc:
            st.error(str(exc))

if st.button("Refresh Audit Log"):
    try:
        audit = _api_request("GET", f"/sessions/{dataset_id}/audit")
        st.dataframe(pd.DataFrame(audit.get("events", [])), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
