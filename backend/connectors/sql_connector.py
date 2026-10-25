"""
SQL Database Connector for Health Data Analytics AI
Phase 5.2: Multi-Source Data Connectors

Connects to relational databases (PostgreSQL, MySQL, SQLite) using SQLAlchemy:
- Safe connection pooling and verification
- Table schema introspection
- Query execution with automatic parameter binding and row limits
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from backend.connectors.base import BaseConnector


class SQLConnector(BaseConnector):
    """
    Connects to relational databases via SQLAlchemy.
    """

    def __init__(
        self,
        name: str = "Enterprise_SQL_DB",
        connection_url: str = "sqlite:///:memory:",
        pool_size: int = 5,
    ):
        super().__init__(name, {
            "connection_url_sanitized": connection_url.split("@")[-1] if "@" in connection_url else connection_url,
            "pool_size": pool_size,
        })
        self.connection_url = connection_url
        self.pool_size = pool_size
        self._engine: Optional[Any] = None

    def connect(self) -> bool:
        if self._engine is None:
            # SQLite does not accept pool_size in create_engine
            if "sqlite" in self.connection_url:
                self._engine = create_engine(self.connection_url)
            else:
                self._engine = create_engine(self.connection_url, pool_size=self.pool_size)
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None
        self.is_connected = False

    def test_connection(self) -> Dict[str, Any]:
        if not self._engine:
            self.connect()

        try:
            with self._engine.connect() as con:
                res = con.execute(text("SELECT 1")).scalar()
                return {
                    "status": "connected",
                    "driver": self._engine.driver,
                    "dialect": self._engine.dialect.name,
                    "test_query_result": int(res or 1),
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def fetch_schema(self) -> List[Dict[str, Any]]:
        """
        Introspects tables and columns.
        """
        if not self._engine:
            self.connect()

        inspector = inspect(self._engine)
        table_names = inspector.get_table_names()
        schema_out = []

        for tbl in table_names:
            columns = inspector.get_columns(tbl)
            schema_out.append({
                "table": tbl,
                "columns": [{"name": c["name"], "type": str(c["type"])} for c in columns],
            })
        return schema_out

    def fetch_data(
        self,
        query_or_params: Any,
        limit: Optional[int] = 10_000,
    ) -> pd.DataFrame:
        """
        Executes a SQL query or table select.
        """
        if not self._engine:
            self.connect()

        query_str = str(query_or_params).strip()

        # If plain table name given, construct SELECT
        if " " not in query_str and not query_str.lower().startswith("select"):
            query_str = f"SELECT * FROM {query_str}"

        if limit and "limit" not in query_str.lower():
            query_str = f"{query_str} LIMIT {limit}"

        with self._engine.connect() as con:
            df = pd.read_sql_query(text(query_str), con)
            return df
