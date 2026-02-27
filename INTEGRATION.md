# AI Analytics Service — Integration Guide

## What Was Built

Three files implement the complete AI analytics intelligence layer for the Health-Data-Analytics-AI system:

```
ai_analytics_service.py     ← Core AI Engine (9 modules)
api.py                      ← FastAPI REST API (10 endpoints)
AIAnalyticsDashboard.jsx    ← React Frontend Demo
requirements-ai-service.txt ← Python dependencies
```

---

## Architecture Map

```
CSV/XLSX Upload
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  ai_analytics_service.py                                │
│                                                         │
│  1. ingest_file()           ← robust CSV/XLSX reader    │
│  2. profile_dataset()       ← schema, missingness, PII  │
│  3. build_facts_bundle()    ← EDA + computed metrics    │
│           │                                             │
│           ├── compute_metrics()   (totals, mean, etc.)  │
│           ├── compute_trends()    (time series)         │
│           ├── compute_segments()  (group-by breakdowns) │
│           ├── detect_outliers()   (IQR method)          │
│           ├── compute_correlations()                    │
│           └── detect_health_context() (OPD/ANC/geo)    │
│                                                         │
│  4. recommend_charts()      ← rule-based chart select   │
│  5. build_dashboard_spec()  ← JSON dashboard blueprint  │
│                                                         │
│  ── CLAUDE AI BOUNDARY ──────────────────────────────── │
│     LLM only receives facts, NEVER raw data             │
│                                                         │
│  6. generate_insights()     ← Claude summarizes facts   │
│  7. ask_data()              ← NL Q&A over facts         │
│  8. generate_report()       ← Claude writes HTML report │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────┐
│    api.py       │   FastAPI REST layer
│  10 endpoints   │
└─────────────────┘
```

---

## Drop-in Integration

### Backend (Python FastAPI)

1. Copy `ai_analytics_service.py` and `api.py` into `/backend/`

2. Install dependencies:
   ```bash
   pip install -r requirements-ai-service.txt
   ```

3. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

4. Run:
   ```bash
   uvicorn backend.api:app --reload --port 8000
   ```

---

### Frontend (Next.js / React)

Copy `AIAnalyticsDashboard.jsx` into your `app/` directory.

It calls the Claude API directly (no proxy needed for the demo).
For production, route requests through your backend `/api/v1/sessions/{id}/ask`.

---

## API Endpoints Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/sessions` | Create a new analysis session |
| POST | `/api/v1/sessions/{id}/upload` | Upload CSV/XLSX file |
| GET | `/api/v1/sessions/{id}/preview` | Preview first N rows |
| GET | `/api/v1/sessions/{id}/profile` | Data quality profile |
| GET | `/api/v1/sessions/{id}/facts` | Full facts bundle (EDA) |
| POST | `/api/v1/sessions/{id}/dashboard-spec` | Generate dashboard JSON spec |
| POST | `/api/v1/sessions/{id}/insights` | AI-generated insights |
| POST | `/api/v1/sessions/{id}/ask` | Ask-Your-Data (NL question) |
| POST | `/api/v1/sessions/{id}/report` | Generate HTML report |
| POST | `/api/v1/sessions/{id}/run-full-pipeline` | Run all steps at once |

---

## No-Hallucination Safety Design

The system enforces Section C of the spec:

1. **Facts Engine computes all numbers** — Claude never sees raw data
2. **Claude only summarizes** the pre-computed facts bundle
3. **Validator checks** that every numeric claim maps to a facts key
4. **Every response includes** `confidence` + `facts_used` fields
5. **Cautious language enforced** via system prompts ("may indicate", "appears to")

---

## Quick Test (Python)

```python
from ai_analytics_service import AIAnalyticsSession

session = AIAnalyticsSession()
results = session.run(
    "your_health_data.csv",
    question="Which district had the highest OPD visits?"
)

# Save all outputs
paths = session.save("./outputs")
print(paths)
# {
#   "facts_bundle":   "./outputs/facts_bundle_sess_abc123.json",
#   "dashboard_spec": "./outputs/dashboard_spec_sess_abc123.json",
#   "insights":       "./outputs/insights_sess_abc123.json",
#   "report_html":    "./outputs/report_sess_abc123.html"
# }
```

---

## Extending the System

| Goal | Where to edit |
|------|--------------|
| Add new health indicator detection | `_detect_health_context()` in `ai_analytics_service.py` |
| Add new chart types | `recommend_charts()` |
| Change dashboard layout | `build_dashboard_spec()` |
| Tune insight quality | `_INSIGHT_SYSTEM` prompt |
| Add forecasting | New `forecast_trends()` function |
| Add PII masking | Extend `profile_dataset()` |
| Add DHIS2 template | New `build_dhis2_dashboard_spec()` |

---

## Production Checklist

- [ ] Replace in-memory `_SESSIONS` dict with Redis
- [ ] Replace `tempfile` with S3-compatible storage
- [ ] Add Celery for large-file async jobs
- [ ] Add PostgreSQL for session metadata
- [ ] Add authentication middleware
- [ ] Add rate limiting per session
- [ ] Enable OpenTelemetry tracing
