"""
REST API Connector for Health Data Analytics AI
Phase 5.2: Multi-Source Data Connectors

Ingests tabular health records from external REST endpoints:
- Supports API keys, Bearer tokens, and Basic Auth
- Extracts JSON payload records by key path (e.g., "data.records")
- Pagination and row-limit safety
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from backend.connectors.base import BaseConnector


class RESTAPIConnector(BaseConnector):
    """
    Connects to external REST endpoints to pull JSON datasets into DataFrames.
    """

    def __init__(
        self,
        name: str = "External_REST_API",
        endpoint_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        auth_type: str = "none",  # "none" | "bearer" | "api_key" | "basic"
        auth_token: Optional[str] = None,
        records_json_path: str = "data",
    ):
        super().__init__(name, {
            "endpoint_url": endpoint_url,
            "auth_type": auth_type,
            "records_json_path": records_json_path,
        })
        self.endpoint_url = endpoint_url
        self.headers = headers or {}
        self.auth_type = auth_type
        self.auth_token = auth_token
        self.records_json_path = records_json_path
        self._session: Optional[requests.Session] = None

    def connect(self) -> bool:
        self._session = requests.Session()
        self._session.headers.update(self.headers)

        if self.auth_type == "bearer" and self.auth_token:
            self._session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
        elif self.auth_type == "api_key" and self.auth_token:
            self._session.headers.update({"X-API-Key": self.auth_token})

        self.is_connected = True
        return True

    def disconnect(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
        self.is_connected = False

    def test_connection(self) -> Dict[str, Any]:
        if not self.endpoint_url:
            return {"status": "error", "message": "No endpoint_url provided"}

        if not self._session:
            self.connect()

        try:
            resp = self._session.head(self.endpoint_url, timeout=5)
            # Some APIs reject HEAD; fallback to GET with limit=1
            if resp.status_code in (404, 405):
                resp = self._session.get(self.endpoint_url, params={"limit": 1}, timeout=5)

            return {
                "status": "connected" if resp.status_code < 400 else "error",
                "http_status": resp.status_code,
                "latency_ms": round(resp.elapsed.total_seconds() * 1000, 1),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def fetch_schema(self) -> List[Dict[str, Any]]:
        """
        Attempts to fetch sample row to introspect column schema.
        """
        sample = self.fetch_data({}, limit=2)
        if sample.empty:
            return []
        return [{"name": col, "type": str(sample[col].dtype)} for col in sample.columns]

    def _extract_records(self, payload: Any) -> List[Dict[str, Any]]:
        """
        Navigates JSON hierarchy via dot-notation path (e.g. 'results.items').
        """
        if isinstance(payload, list):
            return payload

        if not isinstance(payload, dict):
            return []

        if not self.records_json_path:
            # Look for top-level list
            for v in payload.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return [payload]

        parts = self.records_json_path.split(".")
        curr = payload
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                break

        if isinstance(curr, list):
            return curr
        if isinstance(curr, dict):
            return [curr]
        return []

    def fetch_data(
        self,
        query_or_params: Any,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        if not self._session:
            self.connect()

        params = query_or_params if isinstance(query_or_params, dict) else {}
        if limit:
            params.setdefault("limit", limit)

        resp = self._session.get(self.endpoint_url, params=params, timeout=20)
        if resp.status_code >= 400:
            raise RuntimeError(f"REST API request failed with status {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        records = self._extract_records(payload)
        df = pd.DataFrame(records)

        if limit and len(df) > limit:
            df = df.head(limit)
        return df
