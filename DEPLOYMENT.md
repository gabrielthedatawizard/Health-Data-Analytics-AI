# Deployment Guide

## Recommended topology

Use two deploy targets:

1. Frontend on Vercel from the repo root
2. FastAPI backend on a Python-capable service such as Render, Railway, Fly.io, or your own container platform

This repo now supports a same-origin Vercel proxy:

- Browser calls `/api/...`
- Vercel proxy forwards that request to `BACKEND_API_URL`
- FastAPI handles `/sessions`, `/jobs`, `/auth/me`, and the rest of the governed API

## Important

Deploy the repository root on Vercel, not the `app/` folder by itself.

The root project includes:

- the Vite frontend build in `app/`
- the Vercel API proxy in `api/proxy.js`
- the rewrite rules in `vercel.json`

## Vercel environment variables

Set these in the Vercel project:

```bash
BACKEND_API_URL=https://your-fastapi-service.example.com
```

Optional:

```bash
VITE_API_BASE_URL=
```

Leave `VITE_API_BASE_URL` empty if you want the frontend to use the same-origin `/api` proxy by default.

## Backend deployment

The backend expects the dependencies in `requirements-mvp.txt` and serves the governed API from:

- `backend/main.py`

Typical start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## What this fixes

Without `BACKEND_API_URL`, a deployed frontend can fall back to calling its own domain and fail on routes like:

```text
POST /sessions
```

With this setup:

- frontend calls `/api/sessions`
- Vercel proxy forwards to `${BACKEND_API_URL}/sessions`
- the governed backend receives the request correctly
