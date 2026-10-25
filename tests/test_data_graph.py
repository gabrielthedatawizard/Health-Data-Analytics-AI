"""
Tests for Data Relationship Graph Engine (Phase 1.2)
"""

import pytest
import pandas as pd
import numpy as np
from backend.data_graph import (
    DataRelationshipGraph,
    build_data_relationship_graph,
)


@pytest.fixture
def health_facility_df():
    """Generates a realistic hierarchical health facility dataframe."""
    np.random.seed(42)
    regions = ["Coast", "Rift Valley"]
    districts = {
        "Coast": ["Mombasa", "Kilifi"],
        "Rift Valley": ["Nakuru", "Kericho"],
    }
    rows = []
    pid = 1000
    for reg in regions:
        for dist in districts[reg]:
            for fac_num in range(3):
                fac_code = f"FAC_{dist[:3].upper()}_{fac_num}"
                fac_name = f"{dist} Health Center {fac_num+1}"
                for month in ["2024-01-01", "2024-02-01", "2024-03-01"]:
                    for pat in range(4):
                        pid += 1
                        rows.append({
                            "encounter_id": f"ENC_{pid}",
                            "region": reg,
                            "district": dist,
                            "facility_code": fac_code,
                            "facility_name": fac_name,
                            "reporting_date": month,
                            "opd_visits": np.random.randint(100, 500),
                            "bed_occupancy_rate": np.random.uniform(55.0, 95.0),
                            "stock_out_days": np.random.randint(0, 15),
                        })
    return pd.DataFrame(rows)


def test_graph_node_modeling(health_facility_df):
    """Verify nodes are properly created and typed."""
    graph = DataRelationshipGraph(health_facility_df, dataset_name="facility_hmis")
    nodes = graph.nodes

    assert "encounter_id" in nodes
    assert nodes["encounter_id"].node_type == "identifier"
    assert nodes["encounter_id"].is_primary_key_candidate is True

    assert "region" in nodes
    assert nodes["region"].node_type in ("geo_hierarchy", "dimension")

    assert "opd_visits" in nodes
    assert nodes["opd_visits"].node_type == "measure"


def test_measure_dimension_split(health_facility_df):
    """Verify split into measures, dimensions, identifiers, and temporal."""
    graph = DataRelationshipGraph(health_facility_df)
    split = graph.get_measure_dimension_split()

    assert "opd_visits" in split["measures"]
    assert "bed_occupancy_rate" in split["measures"]
    assert "stock_out_days" in split["measures"]
    assert "encounter_id" in split["identifiers"]
    assert "reporting_date" in split["temporal"]
    assert "facility_name" in split["dimensions"] or "facility_name" in split["identifiers"]


def test_hierarchies_and_fd_edges(health_facility_df):
    """Verify hierarchy edges (region -> district) and functional dependencies (facility_code -> facility_name)."""
    graph = DataRelationshipGraph(health_facility_df)
    edges = graph.edges

    relation_types = set(e.relation_type for e in edges)
    assert "hierarchy_of" in relation_types or "determines" in relation_types

    edge_tuples = [(e.source, e.target, e.relation_type) for e in edges]
    # Check facility code determines facility name
    has_fd = any(src == "facility_code" and tgt == "facility_name" for src, tgt, _ in edge_tuples)
    assert has_fd is True


def test_star_schema_suggestion(health_facility_df):
    """Verify fact table design and dimension clustering."""
    graph = DataRelationshipGraph(health_facility_df, dataset_name="facility_hmis")
    star = graph.suggest_star_schema()

    fact = star["fact_table"]
    assert fact["table_name"] == "fact_facility_hmis"
    assert "opd_visits" in fact["measures"]
    assert len(star["dimension_tables"]) >= 1


def test_cross_dataset_join_keys(health_facility_df):
    """Verify join key discovery across two datasets with matching facility codes."""
    other_df = pd.DataFrame({
        "facility_code": ["FAC_MOM_0", "FAC_MOM_1", "FAC_NAK_0"],
        "facility_type": ["Hospital", "Health Center", "Dispensary"],
        "total_nurses": [25, 8, 4],
    })

    graph = DataRelationshipGraph(health_facility_df)
    candidates = graph.find_candidate_join_keys(other_df)

    assert len(candidates) >= 1
    best_candidate = candidates[0]
    assert best_candidate["left_column"] == "facility_code"
    assert best_candidate["right_column"] == "facility_code"
    assert best_candidate["overlap_count"] == 3


def test_topology_export_format(health_facility_df):
    """Verify UI graph format with nodes and edges."""
    res = build_data_relationship_graph(health_facility_df, dataset_name="hmis_test")
    topology = res["topology"]

    assert "nodes" in topology
    assert "edges" in topology
    assert len(topology["nodes"]) == len(health_facility_df.columns)
    assert len(topology["edges"]) > 0

    first_node = topology["nodes"][0]
    assert "id" in first_node
    assert "data" in first_node
    assert "node_type" in first_node["data"]

    first_edge = topology["edges"][0]
    assert "source" in first_edge
    assert "target" in first_edge
    assert "relation_type" in first_edge["data"]
