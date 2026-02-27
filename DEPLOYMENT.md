# Vercel Deployment

This repository is configured for Vercel with the frontend in `app/`.

## Configuration Included

- `vercel.json` at repo root:
  - `installCommand`: `cd app && npm ci --include=dev`
  - `buildCommand`: `cd app && npm run build`
  - `outputDirectory`: `app/dist`
  - SPA rewrite to `index.html`
- `app/vercel.json` (fallback when Vercel Project Root Directory is set to `app/`):
  - `installCommand`: `npm ci --include=dev`
  - `buildCommand`: `npm run build`
  - `outputDirectory`: `dist`
  - SPA rewrite to `index.html`
- `.nvmrc` set to `22.12.0` (compatible with Vite 7)
- `.vercelignore` excludes non-runtime assets and large local reference files
- `app/package.json` includes Node engine `>=20.19.0`

## Deploy From Vercel Dashboard

1. Import this GitHub repository into Vercel.
2. Use either project root:
   - Repository root (uses root `vercel.json`)
   - `app/` root (uses `app/vercel.json`)
3. In Vercel Project Settings, clear any manual Build/Install command overrides so config files are honored.
4. Deploy.

## Deploy With Vercel CLI

```bash
npm i -g vercel
vercel
vercel --prod
```

## Local Docker Compose (API + Worker + Redis + Streamlit)

This stack runs the Python MVP end-to-end with one command.

### 1) Configure environment

```bash
cp .env.example .env
```

Set your LLM credentials in `.env`:
- Azure primary: `AI_PROVIDER=azure` plus `AZURE_OPENAI_*`
- OpenAI fallback: `OPENAI_API_KEY` (and optional `OPENAI_MODEL`)

### 2) Start all services

```bash
docker compose up --build
```

Services:
- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- Redis: `localhost:6379`

### 3) Stop services

```bash
docker compose down
```

### Notes

- Persisted artifacts are mounted in `./data_store`.
- Streamlit sends `X-API-Key` using `APP_API_KEY` for audit attribution.
- Celery worker concurrency is set to `2` for both facts/report jobs.
