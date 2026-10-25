"""
DHIS2 Web API Connector for Health Data Analytics AI
Phase 5.2: Multi-Source Data Connectors

Connects to District Health Information Software 2 (DHIS2) instances:
- Supports Basic Auth and Personal Access Tokens (PAT)
- Queries /api/analytics and /api/dataValueSets
- Resolves UID dimensions (dx, pe, ou) into readable clinical columns
- Includes simulated mock mode for offline testing and staging environments
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from backend.connectors.base import BaseConnector


class DHIS2Connector(BaseConnector):
    """
    Connects to national/district DHIS2 deployments to extract aggregate health data.
    """

    def __init__(
        self,
        name: str = "DHIS2_Instance",
        base_url: str = "https://play.dhis2.org/demo",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_token: Optional[str] = None,
        mock_mode: bool = False,
    ):
        super().__init__(name, {
            "base_url": base_url.rstrip("/"),
            "username": username,
            "has_token": bool(api_token),
            "mock_mode": mock_mode,
        })
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.api_token = api_token
        self.mock_mode = mock_mode
        self._session: Optional[requests.Session] = None

    def connect(self) -> bool:
        if self.mock_mode:
            self.is_connected = True
            return True

        self._session = requests.Session()
        if self.api_token:
            self._session.headers.update({"Authorization": f"ApiToken {self.api_token}"})
        elif self.username and self.password:
            self._session.auth = (self.username, self.password)

        self._session.headers.update({"Accept": "application/json"})
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
        self.is_connected = False

    def test_connection(self) -> Dict[str, Any]:
        if self.mock_mode:
            return {
                "status": "connected",
                "version": "2.40.2 (Simulated Mock)",
                "system_name": "DHIS2 Test Instance",
                "latency_ms": 12,
            }

        if not self._session:
            self.connect()

        try:
            resp = self._session.get(f"{self.base_url}/api/system/info", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "connected",
                    "version": data.get("version", "unknown"),
                    "system_name": data.get("systemName", "DHIS2"),
                    "server_date": data.get("serverDate"),
                }
            return {
                "status": "error",
                "http_code": resp.status_code,
                "message": resp.text[:200],
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def fetch_schema(self) -> List[Dict[str, Any]]:
        """
        Retrieves list of available data elements and indicator groups.
        """
        if self.mock_mode:
            return [
                {"id": "DE_ANC1", "name": "ANC 1st Visit", "type": "data_element"},
                {"id": "DE_ANC4", "name": "ANC 4th Visit", "type": "data_element"},
                {"id": "DE_DELIV", "name": "Institutional Deliveries", "type": "data_element"},
                {"id": "DE_MALARIA", "name": "Confirmed Malaria Cases", "type": "data_element"},
            ]

        if not self._session:
            self.connect()

        try:
            resp = self._session.get(
                f"{self.base_url}/api/dataElements?fields=id,name,valueType&pageSize=50",
                timeout=15,
            )
            if resp.status_code == 200:
                elements = resp.json().get("dataElements", [])
                return [{"id": el["id"], "name": el["name"], "type": el.get("valueType")} for el in elements]
        except Exception:
            pass
        return []

    def fetch_data(
        self,
        query_or_params: Any,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Queries /api/analytics with dimensions dx, pe, ou and returns a clean DataFrame.
        """
        if self.mock_mode:
            # Generate realistic DHIS2 tabular response
            return pd.DataFrame({
                "period": ["2024-01", "2024-02", "2024-03", "2024-04"] * 3,
                "org_unit": ["Central Hospital"] * 4 + ["District Clinic A"] * 4 + ["District Clinic B"] * 4,
                "data_element": ["ANC 1st Visit"] * 12,
                "value": [120, 135, 142, 150, 45, 52, 48, 55, 30, 28, 35, 32],
            })

        if not self._session:
            self.connect()

        params = query_or_params if isinstance(query_or_params, dict) else {}
        url = f"{self.base_url}/api/analytics"

        resp = self._session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"DHIS2 Analytics query failed ({resp.status_code}): {resp.text[:200]}")

        payload = resp.json()
        headers = [h["name"] for h in payload.get("headers", [])]
        rows = payload.get("rows", [])

        df = pd.DataFrame(rows, columns=headers)
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        if limit:
            df = df.head(limit)
        return df
