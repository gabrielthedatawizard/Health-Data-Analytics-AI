# Python MVP: FastAPI + Streamlit

This repo now includes a Python MVP stack in parallel to the existing React app:

- Backend API: `backend/main.py`
- Streamlit client: `app.py`
- Local data/session store: `data_store/`

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-mvp.txt
```

## 2) Run backend

```bash
uvicorn backend.main:app --reload --port 8000
```

## 3) Run Streamlit UI

```bash
streamlit run app.py
```

If backend is not on `http://localhost:8000`, set:

```bash
export BACKEND_API_URL="http://localhost:8000"
```

## Endpoint summary

- `POST /sessions` create deterministic dataset session
- `GET /sessions/{dataset_id}` get session metadata
- `POST /sessions/{dataset_id}/upload` upload CSV/XLSX (+ Excel sheet selection)
- `POST /sessions/{dataset_id}/masking` enable/disable PII masking
- `GET /sessions/{dataset_id}/profile` run and persist data profile
- `GET /sessions/{dataset_id}/facts` compute and persist deterministic facts bundle
- `GET /sessions/{dataset_id}/dashboard-spec` generate and persist dashboard JSON spec
- `GET /sessions/{dataset_id}/preview` preview rows for chart rendering
- `GET /sessions/{dataset_id}/report` generate and persist report HTML
- `GET /sessions/{dataset_id}/report/html` read generated report HTML
- `GET /sessions/{dataset_id}/audit` read audit log

## Notes

- Supported upload types in MVP: `.csv`, `.xlsx`
- Upload size limit: 30 MB
- Artifacts are stored per dataset session under `data_store/<dataset_id>/`
- Insights are generated from computed facts only (no free-form metric generation)
