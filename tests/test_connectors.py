"""
Unit Tests for Phase 5.2 Multi-Source Data Connectors
"""

import pandas as pd
import pytest
from sqlalchemy import text
from backend.connectors import get_connector
from backend.connectors.dhis2_connector import DHIS2Connector
from backend.connectors.sql_connector import SQLConnector
from backend.connectors.rest_connector import RESTAPIConnector


def test_sql_connector_lifecycle_sqlite():
    sql_conn = SQLConnector(connection_url="sqlite:///:memory:")
    assert sql_conn.connect() is True

    # Test connection
    test_res = sql_conn.test_connection()
    assert test_res["status"] == "connected"
    assert test_res["dialect"] == "sqlite"

    # Seed in-memory table
    with sql_conn._engine.connect() as con:
        con.execute(text("CREATE TABLE patients (id INTEGER, name TEXT, age INTEGER)"))
        con.execute(text("INSERT INTO patients VALUES (1, 'Alice', 30), (2, 'Bob', 45)"))
        con.commit()

    # Introspect schema
    schema = sql_conn.fetch_schema()
    assert len(schema) >= 1
    table_names = [s["table"] for s in schema]
    assert "patients" in table_names

    # Fetch data
    df = sql_conn.fetch_data("SELECT * FROM patients WHERE age > 35")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Bob"

    sql_conn.disconnect()
    assert sql_conn.is_connected is False


def test_dhis2_connector_mock_lifecycle():
    dhis2 = DHIS2Connector(
        name="Kenya_DHIS2_Mock",
        mock_mode=True,
    )
    assert dhis2.connect() is True

    # Test connection
    test_res = dhis2.test_connection()
    assert test_res["status"] == "connected"
    assert "Mock" in test_res["version"]

    # Fetch schema
    schema = dhis2.fetch_schema()
    assert len(schema) >= 4
    names = [s["name"] for s in schema]
    assert "ANC 1st Visit" in names

    # Fetch data
    df = dhis2.fetch_data({"dx": "DE_ANC1", "pe": "LAST_4_MONTHS"})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert "org_unit" in df.columns
    assert "value" in df.columns

    dhis2.disconnect()
    assert dhis2.is_connected is False


def test_get_connector_factory():
    c_sql = get_connector("sql", connection_url="sqlite:///:memory:")
    assert isinstance(c_sql, SQLConnector)

    c_dhis2 = get_connector("dhis2", mock_mode=True)
    assert isinstance(c_dhis2, DHIS2Connector)

    c_rest = get_connector("rest", endpoint_url="https://api.example.org/health")
    assert isinstance(c_rest, RESTAPIConnector)

    with pytest.raises(ValueError):
        get_connector("unsupported_backend")
