"""
Phase 1 Integration Test: Deep Data Understanding Engine
Verifies that schema_intelligence, data_graph, and quality_intelligence
seamlessly enrich the core profiling and facts pipelines.
"""

import pytest
import pandas as pd
import numpy as np
from backend.ai_analytics_service import profile_dataset, build_facts_bundle


@pytest.fixture
def hmis_surveillance_df():
    """Generates multi-facility, multi-period health surveillance dataset."""
    np.random.seed(42)
    records = []
    facilities = [
        ("Coast", "Mombasa", "FAC_001", "Coast General Hospital"),
        ("Coast", "Kilifi", "FAC_002", "Kilifi County Hospital"),
        ("Rift Valley", "Nakuru", "FAC_003", "Nakuru Level 5 Hospital"),
    ]
    periods = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]

    for reg, dist, fcode, fname in facilities:
        for period in periods:
            for rep in range(3):
                records.append({
                    "region": reg,
                    "district": dist,
                    "facility_code": fcode,
                    "facility_name": fname,
                    "reporting_month": period,
                    "opd_attendances": np.random.randint(200, 800),
                    "malaria_cases": np.random.randint(10, 60),
                    "measles_vaccine_coverage_pct": np.random.uniform(70.0, 98.0),
                    "bed_occupancy_rate": np.random.uniform(60.0, 90.0),
                    "patient_contact_phone": "+254712345678",
                })
    return pd.DataFrame(records)


def test_phase1_profile_dataset_enrichment(hmis_surveillance_df):
    """Verify profile_dataset produces all Phase 1 intelligence components."""
    profile = profile_dataset(hmis_surveillance_df)

    # 1. Backward compatibility keys exist
    assert "schema" in profile
    assert "missingness" in profile
    assert "duplicates" in profile
    assert "quality_score" in profile

    # 2. Phase 1 Schema Intelligence exists
    assert "schema_intelligence" in profile
    schema_intel = profile["schema_intelligence"]
    assert "data_dictionary" in schema_intel
    assert "column_intelligence" in schema_intel
    col_intel = schema_intel["column_intelligence"]
    assert "measles_vaccine_coverage_pct" in col_intel
    assert col_intel["measles_vaccine_coverage_pct"]["unit"]["symbol"] == "%"

    # 3. Phase 1 Relationship Graph exists
    assert "relationship_graph" in profile
    rel_graph = profile["relationship_graph"]
    assert "star_schema" in rel_graph
    assert "topology" in rel_graph
    assert len(rel_graph["topology"]["nodes"]) == len(hmis_surveillance_df.columns)

    # 4. Phase 1 Quality Intelligence exists
    assert "quality_intelligence" in profile
    quality_intel = profile["quality_intelligence"]
    assert "report_card" in quality_intel
    assert quality_intel["report_card"]["summary"]["grade"] in ("A", "B", "C")


def test_phase1_facts_bundle_enrichment(hmis_surveillance_df):
    """Verify build_facts_bundle contains deep understanding representations."""
    profile = profile_dataset(hmis_surveillance_df)
    facts = build_facts_bundle(
        df=hmis_surveillance_df,
        dataset_id="test_ds_001",
        session_id="test_sess_001",
        profile=profile,
    )

    # Core Facts
    assert facts["dataset_id"] == "test_ds_001"
    assert "metrics" in facts
    assert "trends" in facts
    assert "health_context" in facts

    # Phase 1 Injected Intelligence
    assert "data_dictionary" in facts
    assert facts["data_dictionary"] is not None
    assert "star_schema" in facts
    assert facts["star_schema"] is not None
    assert "topology" in facts
    assert facts["topology"] is not None
    assert "quality_report_card" in facts
    assert facts["quality_report_card"] is not None
