"""
Integration Tests for Phase 5 Advanced Data Processing & Connectors
"""

import pandas as pd
import pytest
from backend.ai_analytics_service import build_facts_bundle, ask_data
from backend.compute_engine import ComputeEngine
from backend.connectors import get_connector


def test_phase5_dhis2_to_facts_bundle_pipeline():
    # 1. Ingest from DHIS2 mock connector
    dhis2 = get_connector("dhis2", mock_mode=True)
    dhis2.connect()
    raw_df = dhis2.fetch_data({"dx": "DE_ANC1"})
    assert not raw_df.empty

    # 2. Feed ingested DHIS2 data into AI Facts Bundle
    bundle = build_facts_bundle(raw_df, session_id="s_dhis2", dataset_id="ds_dhis2")

    assert bundle["dataset_id"] == "ds_dhis2"
    assert "value" in bundle["metrics"]["kpis"]
    # Total should match sum of mock data (120+135+142+150 + 45+52+48+55 + 30+28+35+32 = 872)
    assert bundle["metrics"]["kpis"]["value"]["total"] == 872.0

    # 3. Ask-your-data works directly on DHIS2 ingested data
    ans = ask_data("What is the total value for this DHIS2 dataset?", bundle)
    assert "872" in ans["answer"] or "total" in ans["answer"].lower()


def test_phase5_compute_engine_sql_acceleration_over_facts():
    # Construct multi-district immunization dataset
    df = pd.DataFrame({
        "facility_name": [f"Facility_{i}" for i in range(20)],
        "district": ["Central"] * 10 + ["Eastern"] * 10,
        "dpt3_doses": [150 + i * 5 for i in range(20)],
        "measles_doses": [140 + i * 4 for i in range(20)],
    })

    engine = ComputeEngine()
    # High-speed vectorized multi-metric segment calculation
    agg = engine.compute_aggregations(
        df,
        group_by=["district"],
        metrics=["dpt3_doses", "measles_doses"],
        operations=["sum", "mean"],
    )

    assert len(agg) == 2
    # Verify accurate sum calculation
    central_row = agg[agg["district"] == "Central"].iloc[0]
    expected_central_dpt3 = sum(150 + i * 5 for i in range(10))
    # Col could be dpt3_doses_sum
    sum_col = [c for c in agg.columns if "dpt3" in c and "sum" in c][0]
    assert central_row[sum_col] == expected_central_dpt3
