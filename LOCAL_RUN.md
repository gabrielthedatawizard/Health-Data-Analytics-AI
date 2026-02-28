# Local Run Instructions

## 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-mvp.txt
```

## 2) Start Redis

```bash
docker run --name health-ai-redis -p 6379:6379 -d redis:7
```

Or use the repo compose file:

```bash
docker compose up redis -d
```

## 3) Start FastAPI backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4) Start Celery worker

```bash
celery -A backend.celery_app.celery_app worker --loglevel=info -Q celery
```

## 5) Start Streamlit UI

```bash
streamlit run app.py
```

## Environment variables

### Required/typical

```bash
export BACKEND_API_URL=http://localhost:8000
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0
export REDIS_URL=redis://localhost:6379/0
```

### Optional LLM (Azure preferred, OpenAI fallback)

```bash
export AI_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT=...

# Fallback
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

If no LLM variables are present, the app still works with deterministic rules-based dashboard/query planning.

## Run tests

```bash
pytest -q
```

## API flow (MVP)

1. `POST /sessions`
2. `POST /sessions/{dataset_id}/upload`
3. `GET /sessions/{dataset_id}/profile`
4. `GET /sessions/{dataset_id}/facts`
5. `POST /sessions/{dataset_id}/dashboard-spec`
6. `POST /sessions/{dataset_id}/ask`
7. `POST /sessions/{dataset_id}/report`
8. `GET /sessions/{dataset_id}/export/pdf`
9. `GET /sessions/{dataset_id}/audit`
