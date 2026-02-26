# Health-Data-Analytics-AI

This README is the authoritative build specification for the AI intelligence layer, including site flow, architecture, APIs, data contracts, safety guardrails, performance, healthcare logic, roadmap, and deliverables.

**Date:** 2026-02-26

---

**SECTION A — SITE FLOW (MANDATORY)**

**Primary user flow (numbered sequence)**

1. **Landing**
   - Inputs:
     - None (public view)
   - Outputs:
     - Product overview, CTA to upload dataset, examples
   - UI components:
     - Hero CTA button `Upload Dataset`
     - Secondary CTA `View Demo`
     - Feature cards (EDA, Dashboards, Ask-Your-Data, Reports)
   - Stored in session:
     - Anonymous session token (optional)
   - API calls:
     - `GET /health` (optional)

2. **Upload**
   - Inputs:
     - CSV/XLSX file (drag-drop, file picker)
     - Optional dataset name, description, tags
   - Outputs:
     - Upload progress, success/failure
   - UI components:
     - Drag-drop area
     - File picker
     - Text fields (name, description)
     - Progress bar
     - Button `Create Session`
   - Stored in session:
     - `dataset_id`, `session_id`, `file_metadata` (name, size, type, checksum)
   - API calls:
     - `POST /sessions`
     - `POST /sessions/{id}/upload`

3. **Session Created**
   - Inputs:
     - None (auto redirect)
   - Outputs:
     - Session summary, dataset preview
   - UI components:
     - Preview table (first N rows)
     - Dataset info card
     - Button `Run Profiling`
   - Stored in session:
     - `preview_sample`
     - `storage_uri`
   - API calls:
     - `GET /sessions/{id}`
     - `GET /sessions/{id}/preview`

4. **Profiling**
   - Inputs:
     - Profiling options (sample size, infer types)
   - Outputs:
     - Schema, types, missingness, duplicates
   - UI components:
     - Schema table
     - Missingness heatmap
     - Data quality score card
     - Button `Run EDA`
   - Stored in session:
     - `profile_result`, `schema`, `type_inference`
   - API calls:
     - `GET /sessions/{id}/profile`

5. **EDA**
   - Inputs:
     - EDA scope (full vs sample)
   - Outputs:
     - Distributions, correlations, outliers, summary stats
   - UI components:
     - Summary stat cards
     - Distribution charts
     - Correlation matrix
     - Outlier table
     - Button `Generate Dashboard`
   - Stored in session:
     - `eda_result`, `facts_bundle`
   - API calls:
     - `POST /sessions/{id}/facts`

6. **Dashboard**
   - Inputs:
     - Dashboard template (general or health)
   - Outputs:
     - Interactive dashboard with filters
   - UI components:
     - Global filters (date, region, facility)
     - KPI cards
     - Trend lines
     - Bar charts
     - Map (if geo)
     - Button `Ask Data`
   - Stored in session:
     - `dashboard_spec`, `dashboard_state`
   - API calls:
     - `POST /sessions/{id}/dashboard-spec`
     - `GET /sessions/{id}/dashboard`

7. **Ask-Data**
   - Inputs:
     - Natural language question
   - Outputs:
     - Answer with chart + cited facts
   - UI components:
     - Chat input
     - Response panel (narrative + table/chart)
     - Button `Add to Report`
   - Stored in session:
     - `question_log`, `query_plan`, `answer_facts`
   - API calls:
     - `POST /sessions/{id}/ask`

8. **Report**
   - Inputs:
     - Report template, sections to include
   - Outputs:
     - Report preview (HTML)
   - UI components:
     - Report outline
     - Chart gallery (toggle include)
     - Button `Export PDF`
   - Stored in session:
     - `report_spec`, `report_html`
   - API calls:
     - `POST /sessions/{id}/report`

9. **Export/Share**
   - Inputs:
     - Export format (PDF/PNG/CSV)
     - Share settings (link, role)
   - Outputs:
     - Download link, shareable link
   - UI components:
     - Export dropdown
     - Button `Download`
     - Button `Create Share Link`
   - Stored in session:
     - `export_manifest`, `share_token`
   - API calls:
     - `GET /sessions/{id}/export/{format}`

**Health sector workflow flow**

1. **Ingestion** → 2. **Validation** → 3. **Indicator Computation** → 4. **Trend Monitoring** → 5. **Insight** → 6. **Reporting**

Mapping to product steps:
- Ingestion = Upload + Session Created
- Validation = Profiling + Data quality checks
- Indicator Computation = EDA + Facts bundle
- Trend Monitoring = Dashboard + Ask-Data
- Insight = Insight generation from facts
- Reporting = Report + Export

---

**SECTION B — AI INTELLIGENCE: REQUIRED FEATURES (MANDATORY)**

**P0 (must-have)**
- Robust file reading (CSV/XLSX, encoding issues, delimiter issues)
- Column type inference (numeric/categorical/datetime/text)
- Data quality scoring: missingness, duplicates, invalid dates, outliers, inconsistent categories
- Healthcare-sensitive checks: PII-like columns detection (name, phone, MRN), masking option
- Facts-first EDA computations (LLM never computes numbers)
- Insight generation from facts bundle (LLM summarization only)
- Auto chart selection logic (rules + learning-based optional)
- Dashboard spec generation (JSON schema)
- Render dashboard from spec
- Ask-Your-Data: natural language to safe analysis execution (no arbitrary code execution)
- Auto report generation (HTML/PDF): data quality + key metrics + charts + narrative + limitations

**P1 (should-have)**
- Template dashboards for HMIS/DHIS2-like data and common health indicators
- Drilldowns (district → facility → service area)
- Saved dashboards and versioning
- Role-based access control and audit logs

**P2 (advanced)**
- Forecasting (time series), anomaly alerts, early warning signals
- Dataset connectors (Postgres/MySQL/DHIS2 API)
- Multi-tenant org workspaces, scalable deployment

---

**SECTION C — “NO HALLUCINATION” SAFETY DESIGN (MANDATORY)**

**Facts Bundle structure (JSON)**
```json
{
  "dataset_id": "ds_123",
  "session_id": "sess_abc",
  "generated_at": "2026-02-26T12:00:00Z",
  "schema": {
    "columns": [
      {"name": "visit_date", "type": "datetime", "null_pct": 0.02},
      {"name": "facility", "type": "categorical", "cardinality": 124},
      {"name": "opd_visits", "type": "numeric", "null_pct": 0.01}
    ]
  },
  "quality": {
    "missingness": {"overall_pct": 0.03},
    "duplicates": {"row_pct": 0.01},
    "invalid_dates": {"visit_date": 12},
    "outliers": {"opd_visits": {"count": 18, "method": "IQR"}}
  },
  "metrics": {
    "row_count": 120034,
    "column_count": 28,
    "kpis": {
      "opd_total": 450234,
      "anc1_total": 80321
    }
  },
  "trends": {
    "opd_visits_monthly": {
      "time_grain": "month",
      "series": [
        {"date": "2025-01-01", "value": 32000},
        {"date": "2025-02-01", "value": 30500}
      ]
    }
  },
  "segments": {
    "by_region": [
      {"region": "North", "opd_total": 120000},
      {"region": "South", "opd_total": 98000}
    ]
  },
  "pii_flags": {
    "name": true,
    "phone": false,
    "mrn": true
  }
}
```

**Strict rules**
- The LLM can only produce statements that reference existing facts keys.
- The LLM must cite the computed values it references (include fact key path in the response metadata).
- If missing facts, the LLM must ask for user clarification or request an analysis run.

**Verification step**
- A validator checks narrative output to ensure every numeric or factual claim maps to a `facts` key.
- If any claim is unmapped, block the response and return a system message requesting analysis.

**User-facing confidence/uncertainty display**
- Each answer includes:
  - `confidence`: High/Medium/Low
  - `fact_coverage`: percent of claims mapped to facts
  - `data_coverage`: percent of dataset used (sample vs full)

---

**SECTION D — AI SYSTEM ARCHITECTURE (MANDATORY, WITH COMPONENTS)**

**Text diagram**

Client/UI → API Gateway → Services → Storage

- Client/UI
- API Gateway
- Ingestion & Parsing Service
- Profiling Service
- Analytics/EDA Service
- Visualization Recommender
- Dashboard Spec Builder
- Dashboard Renderer
- Ask-Your-Data Engine
- Report Generator
- Storage Layer
- Observability + Monitoring

**Module details**

1. **Ingestion & Parsing Service**
   - Responsibilities:
     - Validate file, detect encoding, delimiter, parse CSV/XLSX
     - Chunked read for large files
   - Input schema:
     - `multipart/form-data` file
   - Output schema:
     - `{dataset_id, session_id, storage_uri, file_metadata}`
   - Failure modes + fallbacks:
     - Encoding errors → try fallback encodings
     - Delimiter errors → auto-detect via sniffing
     - Large file → partial sample + enqueue full parse

2. **Profiling Service**
   - Responsibilities:
     - Schema inference, missingness, type inference, duplicates
   - Input:
     - `dataset_id`, profiling options
   - Output:
     - `profile_result` JSON
   - Failure modes + fallbacks:
     - Unsupported type → cast to string
     - Heavy compute → return sample profile + queue full

3. **Analytics/EDA Service**
   - Responsibilities:
     - Compute stats, distributions, correlations, outliers
     - Produce Facts Bundle
   - Input:
     - `dataset_id`, EDA options
   - Output:
     - `facts_bundle`
   - Failure modes + fallbacks:
     - High cardinality → cap top-N
     - Sparse data → note limitations

4. **Visualization Recommender**
   - Responsibilities:
     - Rule-based chart selection
     - Suggest filters and groupings
   - Input:
     - `facts_bundle`
   - Output:
     - `viz_recommendations` JSON
   - Failure modes + fallbacks:
     - No valid chart → default to table

5. **Dashboard Spec Builder**
   - Responsibilities:
     - Generate dashboard JSON spec
   - Input:
     - `facts_bundle`, `viz_recommendations`
   - Output:
     - `dashboard_spec`
   - Failure modes + fallbacks:
     - Missing facts → create minimal spec with KPI cards

6. **Dashboard Renderer**
   - Responsibilities:
     - Render dashboard from spec
   - Input:
     - `dashboard_spec`
   - Output:
     - HTML/JS render
   - Failure modes + fallbacks:
     - Invalid spec → render error panel + diagnostic

7. **Ask-Your-Data Engine**
   - Responsibilities:
     - NL → query plan → safe execution → response
   - Input:
     - `question`, `dataset_id`
   - Output:
     - `{answer, chart, facts_used}`
   - Failure modes + fallbacks:
     - Unsupported query → return clarification request

8. **Report Generator**
   - Responsibilities:
     - Generate HTML/PDF report from facts + charts
   - Input:
     - `facts_bundle`, `dashboard_spec`
   - Output:
     - `report_html`, `report_pdf_uri`
   - Failure modes + fallbacks:
     - PDF render failure → return HTML only

9. **Storage Layer**
   - Responsibilities:
     - Raw files (S3), cleaned data, metadata, results
   - Input:
     - Data artifacts
   - Output:
     - Storage URIs
   - Failure modes + fallbacks:
     - S3 outage → local cache + retry

10. **Observability + Monitoring**
   - Responsibilities:
     - Logs, metrics, tracing, audit trails
   - Input:
     - Service telemetry
   - Output:
     - Dashboards + alerts
   - Failure modes + fallbacks:
     - Telemetry drop → local buffer

---

**SECTION E — API DESIGN (MANDATORY)**

All endpoints are versioned under `/api/v1` (omitted below for clarity).

1. `POST /sessions`
   - Request body:
     ```json
     {"name": "Q1 HMIS", "description": "2025 Q1", "tags": ["hmis"]}
     ```
   - Response:
     ```json
     {"session_id": "sess_abc", "dataset_id": "ds_123", "created_at": "2026-02-26T12:00:00Z"}
     ```
   - Latency target: 200 ms
   - Caching: none
   - Idempotency: optional idempotency key

2. `POST /sessions/{id}/upload`
   - Request: multipart file upload
   - Response:
     ```json
     {"status": "uploaded", "storage_uri": "s3://bucket/ds_123.csv", "file_metadata": {"rows": 120034}}
     ```
   - Latency target: streaming upload
   - Caching: none
   - Idempotency: checksum-based

3. `GET /sessions/{id}/profile`
   - Response:
     ```json
     {"schema": [...], "missingness": {...}, "duplicates": {...}}
     ```
   - Latency target: 1-3s for small data
   - Caching: cache per session_id
   - Idempotency: GET

4. `POST /sessions/{id}/clean`
   - Request:
     ```json
     {"actions": ["drop_empty_cols", "normalize_dates"], "pii_mask": true}
     ```
   - Response:
     ```json
     {"status": "queued", "job_id": "job_123"}
     ```
   - Latency: async
   - Caching: n/a
   - Idempotency: job dedupe

5. `GET /sessions/{id}/facts`
   - Response:
     ```json
     {"facts_bundle": {...}}
     ```
   - Latency target: 2-6s (sample), async for full
   - Caching: cache per session_id + params
   - Idempotency: GET

6. `POST /sessions/{id}/dashboard-spec`
   - Request:
     ```json
     {"template": "health_core", "include": ["kpi", "trend", "map"]}
     ```
   - Response:
     ```json
     {"dashboard_spec": {...}}
     ```
   - Latency target: 1-2s
   - Caching: cache per session_id
   - Idempotency: yes

7. `GET /sessions/{id}/dashboard`
   - Response:
     ```json
     {"dashboard_html": "<div>...</div>", "spec_version": "v1"}
     ```
   - Latency target: 200-500 ms (spec already built)
   - Caching: CDN by spec_version
   - Idempotency: GET

8. `POST /sessions/{id}/ask`
   - Request:
     ```json
     {"question": "How did OPD visits change by month in 2025?", "mode": "safe"}
     ```
   - Response:
     ```json
     {"answer": "OPD visits decreased 4.7% from Jan to Feb 2025.", "facts_used": ["trends.opd_visits_monthly"], "chart": {...}}
     ```
   - Latency target: 2-5s
   - Caching: per question hash + dataset_id
   - Idempotency: yes

9. `POST /sessions/{id}/report`
   - Request:
     ```json
     {"template": "health_report", "sections": ["quality", "kpis", "trends", "limitations"]}
     ```
   - Response:
     ```json
     {"report_html": "<html>...</html>", "report_pdf_uri": "s3://bucket/report_123.pdf"}
     ```
   - Latency target: 5-15s
   - Caching: cache per session_id + template
   - Idempotency: yes

10. `GET /sessions/{id}/export/{format}`
    - Response:
      - PDF/PNG/CSV stream
    - Latency target: 1-5s
    - Caching: CDN (PDF/PNG) by version
    - Idempotency: GET

---

**SECTION F — PERFORMANCE & SCALABILITY (MANDATORY)**

**Dataset size definitions**
- Small: ≤ 100k rows, ≤ 50 columns
- Mid-sized: 100k–5M rows, ≤ 200 columns
- Large: > 5M rows or > 200 columns

**Strategies**
- Sample for preview + async full computation
- Use DuckDB/Polars for EDA speed
- Cache computed results per `session_id`
- Chunked reading for large CSVs
- Background jobs (queue) for heavy tasks
- Limits + user messaging:
  - If >5M rows, run sampling and show banner “Full compute queued.”
  - If >20M rows, require filter or user confirmation

---

**SECTION G — HEALTHCARE ANALYTICS LOGIC (MANDATORY)**

**Detection rules**
- Time columns: detect `date`, `visit_date`, `month`, `week`, parse with strict formats
- Geography: detect `region`, `district`, `facility`
- Service categories: detect `OPD`, `ANC`, `Immunization`, `NCD`, `TB`, `HIV`

**Compute common indicators**
- Counts and totals by time and geography
- Rates and proportions where denominator exists (e.g., ANC1/ANC4)
- Rolling averages (3-month) for trend smoothing

**Trend narratives**
- Use cautious language: “may indicate”, “is associated with”
- Avoid causal claims

**Reports include limitations**
- Missing data impact
- Sampling vs full dataset
- Potential data entry biases

---

**SECTION H — IMPLEMENTATION ROADMAP (MANDATORY, STEP-BY-STEP)**

**Phase 1: AI Core MVP (2–6 weeks)**

Tasks in order:
1. Build Facts Engine (EDA + data quality) using Polars/DuckDB
2. Build Chart Recommender (rule-based)
3. Build Dashboard Spec Builder (JSON schema)
4. Build Ask-Your-Data (safe query plan + execution)
5. Build Report Generator (HTML + PDF)

Definition of Done:
- Facts bundle generated for any dataset
- Dashboard spec rendered with charts
- Ask-Data answers reference facts only
- Reports export to PDF

Recommended tests:
- Unit tests for type inference and quality checks
- Integration tests for facts → dashboard
- Data validation tests on sample healthcare datasets

Risks + mitigations:
- Large dataset performance → sampling + async jobs
- Hallucinated numbers → strict facts validator

**Phase 2: Healthcare templates + governance**

Tasks:
1. HMIS/DHIS2 templates
2. Drilldowns (geo hierarchy)
3. PII detection + masking
4. RBAC + audit logs

Definition of Done:
- Templates usable on standard HMIS datasets
- PII flagged and masked
- Audit log entries created for all actions

Tests:
- PII detection accuracy tests
- RBAC integration tests

Risks + mitigations:
- False positives in PII → allow user override

**Phase 3: Predictive analytics**

Tasks:
1. Forecasting (ARIMA/Prophet-like)
2. Anomaly detection
3. Monitoring and alerts

Definition of Done:
- Forecasts available for time series
- Anomaly flags displayed on dashboard

Tests:
- Forecast accuracy tests on backtests
- Alert threshold tests

Risks + mitigations:
- Overconfidence in forecasts → always show uncertainty bands

---

**SECTION I — TOOLING STACK (MANDATORY)**

**Recommended stack**
- Python: Polars (fast), DuckDB (SQL), pandas (compat)
- Visualization: Plotly
- ML/Stats: scikit-learn, statsmodels
- Backend: FastAPI
- Frontend: Streamlit (MVP), migrate to React later
- Storage: Postgres (metadata) + S3-compatible (raw files)
- Queues: Celery/RQ + Redis
- Observability: OpenTelemetry + logs/metrics

**Low-budget alternatives**
- SQLite instead of Postgres
- Local filesystem instead of S3
- Background tasks via FastAPI `BackgroundTasks`

---

**SECTION J — DELIVERABLES (MANDATORY)**

**One-page architecture summary**
- The system ingests CSV/XLSX datasets into sessions, profiles them, computes a facts bundle, and uses that to drive dashboards, ask-your-data queries, and reports. The LLM never computes numbers; it only summarizes facts. A validator enforces that every claim references a fact key. The backend is FastAPI with Polars/DuckDB for analytics, Redis for job queueing, and Postgres/S3 for storage. Dashboards are generated from JSON specs and rendered in the UI. Observability provides traceability and audit logs for healthcare governance.

**Prioritized backlog**
1. Facts engine + data quality scoring
2. Chart recommender (rule-based)
3. Dashboard spec schema and renderer
4. Ask-Your-Data safe query engine
5. Report generator (HTML/PDF)
6. PII detection + masking
7. HMIS/DHIS2 templates
8. RBAC + audit logs
9. Forecasting + anomaly detection

**Minimal DashboardSpec JSON example**
```json
{
  "title": "Health KPI Overview",
  "filters": [
    {"field": "month", "type": "date", "default": "2025-01"},
    {"field": "region", "type": "category"}
  ],
  "components": [
    {"type": "kpi", "title": "OPD Visits", "value_key": "metrics.kpis.opd_total"},
    {"type": "line", "title": "OPD Trend", "series_key": "trends.opd_visits_monthly"}
  ]
}
```

**Minimal FactsBundle JSON example**
```json
{
  "dataset_id": "ds_123",
  "metrics": {"row_count": 120034, "column_count": 28},
  "quality": {"missingness": {"overall_pct": 0.03}},
  "trends": {
    "opd_visits_monthly": {
      "time_grain": "month",
      "series": [
        {"date": "2025-01-01", "value": 32000},
        {"date": "2025-02-01", "value": 30500}
      ]
    }
  }
}
```

**Exact next step to implement first (with reasons)**
- Implement the **Facts Engine** (EDA + data quality + Facts Bundle JSON).
- Reason: every other AI feature (charts, dashboards, Ask-Data, reports, safety) depends on reliable computed facts. This is the foundation that enforces no-hallucination safety and performance.

