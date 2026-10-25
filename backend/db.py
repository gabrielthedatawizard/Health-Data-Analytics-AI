from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("healthai")


def _database_url() -> str | None:
    url = os.getenv("DATABASE_URL", "").strip()
    return url or None


def _psycopg():
    try:
        import psycopg  # type: ignore

        return psycopg
    except Exception:
        return None


def pg_available() -> bool:
    return bool(_database_url() and _psycopg())


def init_pg_schema() -> bool:
    url = _database_url()
    mod = _psycopg()
    if not url or mod is None:
        return False
    try:
        with mod.connect(url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        dataset_id TEXT PRIMARY KEY,
                        created_by TEXT NOT NULL DEFAULT 'anonymous',
                        status TEXT NOT NULL DEFAULT 'created',
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        meta JSONB NOT NULL DEFAULT '{}'::jsonb
                    );
                    CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions (created_by);
                    """
                )
        return True
    except Exception as exc:
        logger.info(json.dumps({"event": "pg_init_failed", "error": str(exc)[:200]}))
        return False


def mirror_session_meta(dataset_id: str, meta: dict[str, Any]) -> None:
    url = _database_url()
    mod = _psycopg()
    if not url or mod is None:
        return
    try:
        with mod.connect(url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (dataset_id, created_by, status, updated_at, meta)
                    VALUES (%s, %s, %s, NOW(), %s::jsonb)
                    ON CONFLICT (dataset_id) DO UPDATE SET
                        created_by = EXCLUDED.created_by,
                        status = EXCLUDED.status,
                        updated_at = NOW(),
                        meta = EXCLUDED.meta;
                    """,
                    (
                        dataset_id,
                        str(meta.get("created_by") or meta.get("user_id") or "anonymous"),
                        str(meta.get("status") or "created"),
                        json.dumps(meta, ensure_ascii=False, default=str),
                    ),
                )
    except Exception as exc:
        logger.info(json.dumps({"event": "pg_mirror_failed", "dataset_id": dataset_id, "error": str(exc)[:200]}))


def pg_session_ids_for_actor(actor: str, read_all: bool = False, limit: int = 500) -> list[str] | None:
    url = _database_url()
    mod = _psycopg()
    if not url or mod is None:
        return None
    try:
        with mod.connect(url) as conn:
            with conn.cursor() as cur:
                if read_all:
                    cur.execute("SELECT dataset_id FROM sessions ORDER BY updated_at DESC LIMIT %s;", (limit,))
                else:
                    cur.execute("SELECT dataset_id FROM sessions WHERE created_by = %s ORDER BY updated_at DESC LIMIT %s;", (actor, limit))
                rows = cur.fetchall()
                return [str(r[0]) for r in rows]
    except Exception as exc:
        logger.info(json.dumps({"event": "pg_list_failed", "error": str(exc)[:200]}))
        return None


def pg_status() -> dict[str, Any]:
    url = _database_url()
    mod = _psycopg()
    if not url:
        return {"configured": False, "driver": bool(mod), "reachable": False}
    if mod is None:
        return {"configured": True, "driver": False, "reachable": False, "hint": "pip install psycopg[binary]"}
    try:
        with mod.connect(url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM sessions;")
                count = cur.fetchone()[0]
        return {"configured": True, "driver": True, "reachable": True, "sessions": int(count)}
    except Exception as exc:
        return {"configured": True, "driver": True, "reachable": False, "error": str(exc)[:200]}
