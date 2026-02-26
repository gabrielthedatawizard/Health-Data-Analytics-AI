from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

CACHE_DIR = Path("data_store/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class CacheManager:
    def __init__(self) -> None:
        self._redis = None
        redis_url = os.getenv("REDIS_URL") or os.getenv("CELERY_RESULT_BACKEND")
        if redis and redis_url:
            try:
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def _file_path(self, key: str) -> Path:
        safe = sha256(key.encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{safe}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self._redis:
            value = self._redis.get(key)
            if value:
                return json.loads(value)
        path = self._file_path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 86400) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        if self._redis:
            self._redis.setex(key, ttl_seconds, payload)
        path = self._file_path(key)
        path.write_text(payload, encoding="utf-8")

    def invalidate_prefix(self, prefix: str) -> None:
        if self._redis:
            for key in self._redis.scan_iter(match=f"{prefix}*"):
                self._redis.delete(key)
        for path in CACHE_DIR.glob("*.json"):
            # file cache uses hashed keys; best-effort cleanup by clearing all
            path.unlink(missing_ok=True)
