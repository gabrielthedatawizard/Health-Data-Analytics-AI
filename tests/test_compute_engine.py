"""
Unit Tests for Phase 5.1 High-Performance Compute Engine
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from backend.compute_engine import ComputeEngine, DUCKDB_AVAILABLE


@pytest.fixture
def compute_engine():
    return ComputeEngine()


def test_auto_select_backend(compute_engine):
    # Small DataFrame defaults to pandas for simple operations
    assert compute_engine.auto_select_backend(100, operation="aggregate") == "pandas"

    if DUCKDB_AVAILABLE:
        # Large row counts select duckdb
        assert compute_engine.auto_select_backend(50_000, operation="aggregate") == "duckdb"
        # Complex SQL or joins select duckdb
        assert compute_engine.auto_select_backend(500, operation="sql") == "duckdb"


def test_execute_sql_duckdb(compute_engine):
    if not DUCKDB_AVAILABLE:
        pytest.skip("DuckDB not available")

    df_facilities = pd.DataFrame({
        "facility_id": [1, 2, 3],
        "facility_name": ["Hospital A", "Clinic B", "Health Center C"],
        "region": ["Central", "North", "Central"],
    })
    df_admissions = pd.DataFrame({
        "facility_id": [1, 1, 2, 3],
        "admissions": [120, 150, 45, 80],
        "month": ["2024-01", "2024-02", "2024-01", "2024-01"],
    })

    # Join and aggregate directly over in-memory DataFrames
    sql = """
    SELECT f.region, SUM(a.admissions) AS total_admissions, AVG(a.admissions) AS avg_admissions
    FROM a JOIN f ON a.facility_id = f.facility_id
    GROUP BY f.region
    ORDER BY total_admissions DESC
    """
    res = compute_engine.execute_sql(sql, tables={"f": df_facilities, "a": df_admissions})

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert res.iloc[0]["region"] == "Central"
    assert res.iloc[0]["total_admissions"] == 350


def test_compute_aggregations(compute_engine):
    df = pd.DataFrame({
        "district": ["North", "North", "South", "South"],
        "cases": [10, 20, 30, 40],
        "tests": [100, 200, 300, 400],
    })

    agg = compute_engine.compute_aggregations(
        df,
        group_by=["district"],
        metrics=["cases", "tests"],
        operations=["sum", "mean"],
    )

    assert isinstance(agg, pd.DataFrame)
    assert len(agg) == 2
    col_names = [str(c).lower() for c in agg.columns]
    assert any("cases_sum" in c or ("cases" in c and "sum" in c) for c in col_names)


def test_fast_profile_summary(compute_engine):
    np.random.seed(42)
    df = pd.DataFrame({
        "metric_a": np.random.normal(100, 10, 1000),
        "metric_b": np.random.uniform(0, 50, 1000),
    })
    df.loc[0:49, "metric_a"] = np.nan

    summary = compute_engine.fast_profile_summary(df)

    assert summary["row_count"] == 1000
    assert summary["column_count"] == 2
    assert "metric_a" in summary["columns"]
    assert summary["columns"]["metric_a"]["missing_count"] == 50
    assert summary["columns"]["metric_a"]["missing_pct"] == 5.0


def test_streaming_csv_summary(compute_engine):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Generate temporary 500-row CSV
        f.write("facility,visits,deliveries\n")
        for i in range(500):
            f.write(f"Facility_{i % 5},{100 + i},{20 + (i % 10)}\n")
        tmp_csv = f.name

    try:
        res = compute_engine.process_large_csv_summary(
            tmp_csv,
            numeric_cols=["visits", "deliveries"],
            chunk_size=100,
        )
        assert res["total_rows_streamed"] == 500
        assert "visits" in res["kpis"]
        assert res["kpis"]["visits"]["count"] == 500
        assert res["kpis"]["visits"]["min"] == 100
        assert res["kpis"]["visits"]["max"] == 599
    finally:
        Path(tmp_csv).unlink(missing_ok=True)
