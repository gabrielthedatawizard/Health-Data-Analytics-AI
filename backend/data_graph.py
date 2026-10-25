"""
Data Relationship Graph Engine for Health Data Analytics AI
Phase 1.2: Deep Data Understanding Engine

Constructs an entity-relationship and semantic topology graph from data:
- Node modeling: Entities, Dimensions, Measures, Temporal Keys, Identifiers
- Edge modeling: Hierarchies (Parent-Child), Functional Dependencies (1:1, N:1),
  Temporal Sequencing, Derived Metrics
- Automated Star Schema Formulation (Fact Table candidates vs. Dimension Tables)
- Multi-dataset join key discovery
- Graph export format ready for frontend visualization (Cytoscape / ReactFlow / D3)
"""

from __future__ import annotations

import math
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from backend.schema_intelligence import SchemaIntelligence, ColumnIntelligence


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str  # "dimension", "measure", "identifier", "temporal", "geo_hierarchy"
    semantic_type: str
    category: str
    unit: Optional[str] = None
    cardinality: int = 0
    null_pct: float = 0.0
    is_primary_key_candidate: bool = False
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation_type: str  # "hierarchy_of", "determines", "interval_paired_with", "derived_from", "aggregates"
    weight: float = 1.0
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DataRelationshipGraph:
    """
    Constructs and traverses the semantic and relational graph of a dataset.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        self.df = df
        self.dataset_name = dataset_name
        self.rows = len(df)
        self.schema_engine = SchemaIntelligence(df)
        self.col_profiles = self.schema_engine.analyze_columns()
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._build_graph()

    def _build_graph(self) -> None:
        """Constructs nodes and relationships from column intelligence."""
        # 1. Create Nodes
        for col, prof in self.col_profiles.items():
            sem = prof["semantic"]
            gran = prof["granularity"]
            temp = prof["temporal"]
            unit = prof["unit"]

            # Classify node type
            if gran.get("is_candidate_key") and not pd.api.types.is_float_dtype(self.df[col]):
                node_type = "identifier"
            elif temp and temp.get("is_temporal"):
                node_type = "temporal"
            elif sem["category"] == "geography" or "hierarchy" in sem["category"]:
                node_type = "geo_hierarchy"
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                # Numerics are measures unless low-cardinality flags or IDs
                if sem["category"] in ("demographics_id",) or gran["granularity"] == "categorical_dimension":
                    node_type = "dimension"
                else:
                    node_type = "measure"
            else:
                node_type = "dimension"

            node = GraphNode(
                id=col,
                label=col.replace("_", " ").title(),
                node_type=node_type,
                semantic_type=sem["semantic_type"],
                category=sem["category"],
                unit=unit["symbol"] if unit else None,
                cardinality=prof["unique_count"],
                null_pct=prof["null_pct"],
                is_primary_key_candidate=gran.get("is_candidate_key", False),
                properties={
                    "granularity": gran["granularity"],
                    "confidence": sem["confidence"],
                    "temporal_pattern": temp["pattern"] if temp else None,
                },
            )
            self.nodes[col] = node

        # 2. Hierarchical Edges
        hierarchies = self.schema_engine.detect_hierarchies()
        for h in hierarchies:
            self.edges.append(
                GraphEdge(
                    source=h["parent"],
                    target=h["child"],
                    relation_type="hierarchy_of",
                    weight=h["consistency_rate"],
                    label=f"Hierarchy ({int(h['consistency_rate']*100)}%)",
                    metadata={"parent_unique": h["parent_unique"], "child_unique": h["child_unique"]},
                )
            )

        # 3. Functional Dependency Edges
        fds = self.schema_engine.detect_functional_dependencies()
        for fd in fds:
            # Avoid duplicate if already covered by hierarchy
            existing = any(
                e.source == fd["determinant"] and e.target == fd["dependent"]
                for e in self.edges
            )
            if not existing:
                self.edges.append(
                    GraphEdge(
                        source=fd["determinant"],
                        target=fd["dependent"],
                        relation_type="determines",
                        weight=fd["fidelity"],
                        label=f"Determines ({int(fd['fidelity']*100)}%)",
                        metadata={"fidelity": fd["fidelity"]},
                    )
                )

        # 4. Temporal Interval Edges
        intervals = self.schema_engine.detect_interval_pairs()
        for inv in intervals:
            self.edges.append(
                GraphEdge(
                    source=inv["start_column"],
                    target=inv["end_column"],
                    relation_type="interval_paired_with",
                    weight=inv["consistency_rate"],
                    label=f"Duration ({inv['median_duration_days']}d)",
                    metadata={
                        "median_duration_days": inv["median_duration_days"],
                        "suggested_feature": inv["suggested_duration_feature"],
                    },
                )
            )

        # 5. Measure to Dimension Associative Edges
        self._detect_dimension_measure_links()

    def _detect_dimension_measure_links(self) -> None:
        """
        Links high-impact dimensions to measures when group variance explains
        a substantial fraction of the measure's total variation (ANOVA/eta-squared).
        """
        measures = [nid for nid, node in self.nodes.items() if node.node_type == "measure"]
        dimensions = [
            nid for nid, node in self.nodes.items()
            if node.node_type in ("dimension", "geo_hierarchy") and 2 <= node.cardinality <= 50
        ]

        if not measures or not dimensions or self.rows < 20:
            return

        for dim in dimensions[:8]:  # Limit checks for performance
            for meas in measures[:12]:
                try:
                    df_sub = self.df[[dim, meas]].dropna()
                    if len(df_sub) < 15:
                        continue
                    # Compute between-group vs total variance ratio (eta-squared)
                    grand_mean = df_sub[meas].mean()
                    total_ss = ((df_sub[meas] - grand_mean) ** 2).sum()
                    if total_ss <= 0:
                        continue
                    group_means = df_sub.groupby(dim)[meas].mean()
                    group_counts = df_sub.groupby(dim)[meas].count()
                    between_ss = (group_counts * ((group_means - grand_mean) ** 2)).sum()
                    eta_sq = between_ss / total_ss

                    if eta_sq >= 0.25:  # At least 25% variance explained
                        self.edges.append(
                            GraphEdge(
                                source=dim,
                                target=meas,
                                relation_type="explains_variation",
                                weight=round(float(eta_sq), 3),
                                label=f"R²={round(float(eta_sq), 2)}",
                                metadata={"variance_explained_eta_sq": round(float(eta_sq), 3)},
                            )
                        )
                except Exception:
                    continue

    def get_measure_dimension_split(self) -> dict[str, list[str]]:
        """
        Categorizes all columns into analytical measures vs dimensions.
        """
        measures = []
        dimensions = []
        identifiers = []
        temporal = []

        for nid, node in self.nodes.items():
            if node.node_type == "measure":
                measures.append(nid)
            elif node.node_type == "identifier":
                identifiers.append(nid)
            elif node.node_type == "temporal":
                temporal.append(nid)
            else:
                dimensions.append(nid)

        return {
            "measures": measures,
            "dimensions": dimensions,
            "identifiers": identifiers,
            "temporal": temporal,
        }

    def suggest_star_schema(self) -> dict[str, Any]:
        """
        Infers an optimal Dimensional / Star Schema model:
        - Fact table definition (foreign keys + measures + additive grain)
        - Dimension tables (entities grouped by functional dependency clusters)
        """
        split = self.get_measure_dimension_split()
        measures = split["measures"]
        dimensions = split["dimensions"]
        temporal = split["temporal"]
        identifiers = split["identifiers"]

        # Group dimensions by hierarchy / functional dependency clusters
        dim_clusters: list[dict[str, Any]] = []
        visited = set()

        # 1. Geo/Admin cluster
        geo_nodes = [nid for nid in dimensions if self.nodes[nid].node_type == "geo_hierarchy"]
        if geo_nodes:
            dim_clusters.append({
                "table_name": "dim_geography",
                "cluster_type": "geography_hierarchy",
                "columns": geo_nodes,
                "primary_key": geo_nodes[-1] if geo_nodes else None,
            })
            visited.update(geo_nodes)

        # 2. FD-based entity clusters
        for edge in self.edges:
            if edge.relation_type == "determines":
                det = edge.source
                dep = edge.target
                if det not in visited or dep not in visited:
                    # Find or create cluster
                    cluster_name = f"dim_{det.replace('_id', '').replace('_code', '')}"
                    dim_clusters.append({
                        "table_name": cluster_name,
                        "cluster_type": "entity",
                        "columns": list(set([det, dep])),
                        "primary_key": det,
                    })
                    visited.update([det, dep])

        # 3. Remaining unclustered dimensions
        remaining = [d for d in dimensions if d not in visited]
        if remaining:
            dim_clusters.append({
                "table_name": "dim_attributes",
                "cluster_type": "context_attributes",
                "columns": remaining,
                "primary_key": None,
            })

        # Fact table design
        fact_keys = identifiers + temporal
        fact_table = {
            "table_name": f"fact_{self.dataset_name}",
            "grain_keys": fact_keys,
            "measures": measures,
            "foreign_keys": [c["primary_key"] for c in dim_clusters if c["primary_key"]],
        }

        return {
            "fact_table": fact_table,
            "dimension_tables": dim_clusters,
            "total_dimensions": len(dimensions),
            "total_measures": len(measures),
        }

    def find_candidate_join_keys(self, other_df: Optional[pd.DataFrame] = None) -> list[dict[str, Any]]:
        """
        Identifies columns that serve as high-quality join keys
        (either internally as unique anchors or across to another dataframe).
        """
        candidate_keys = []

        if other_df is None:
            # Self candidate primary keys
            for nid, node in self.nodes.items():
                if node.is_primary_key_candidate and node.null_pct == 0.0:
                    candidate_keys.append({
                        "column": nid,
                        "key_type": "primary_key",
                        "cardinality": node.cardinality,
                        "confidence": 0.95,
                    })
        else:
            # Cross-dataframe foreign key matching
            for col_a in self.df.columns:
                prof_a = self.col_profiles.get(col_a)
                if not prof_a:
                    continue

                for col_b in other_df.columns:
                    # Check exact name or strong semantic alignment
                    name_match = col_a.lower() == col_b.lower()
                    sem_a = prof_a["semantic"]["semantic_type"]

                    if name_match or (sem_a in ("facility_code", "patient_id", "district_county")):
                        set_a = set(self.df[col_a].dropna().unique())
                        set_b = set(other_df[col_b].dropna().unique())
                        overlap = len(set_a.intersection(set_b))
                        union = len(set_a.union(set_b))
                        jaccard = overlap / max(union, 1)

                        if overlap > 0 and (jaccard > 0.1 or overlap >= 5):
                            candidate_keys.append({
                                "left_column": col_a,
                                "right_column": col_b,
                                "overlap_count": overlap,
                                "jaccard_similarity": round(jaccard, 3),
                                "semantic_type": sem_a,
                                "confidence": round(0.5 + 0.5 * jaccard, 2),
                            })

        return candidate_keys

    def to_graph_dict(self) -> dict[str, Any]:
        """
        Exports graph structure in a standardized format for UI graph rendering
        (Cytoscape / ReactFlow compatible nodes and edges).
        """
        ui_nodes = []
        for nid, node in self.nodes.items():
            ui_nodes.append({
                "id": node.id,
                "data": {
                    "label": node.label,
                    "node_type": node.node_type,
                    "semantic_type": node.semantic_type,
                    "category": node.category,
                    "unit": node.unit,
                    "cardinality": node.cardinality,
                    "null_pct": node.null_pct,
                    "is_primary_key": node.is_primary_key_candidate,
                    **node.properties,
                },
                "classes": f"node-{node.node_type}",
            })

        ui_edges = []
        for idx, edge in enumerate(self.edges):
            ui_edges.append({
                "id": f"e_{edge.source}_{edge.target}_{idx}",
                "source": edge.source,
                "target": edge.target,
                "data": {
                    "relation_type": edge.relation_type,
                    "weight": edge.weight,
                    "label": edge.label,
                    **edge.metadata,
                },
                "classes": f"edge-{edge.relation_type}",
            })

        return {
            "nodes": ui_nodes,
            "edges": ui_edges,
            "meta": {
                "dataset_name": self.dataset_name,
                "total_nodes": len(ui_nodes),
                "total_edges": len(ui_edges),
            },
        }


def build_data_relationship_graph(df: pd.DataFrame, dataset_name: str = "dataset") -> dict[str, Any]:
    """
    Main API for generating complete relationship graph, star schema, and UI topology.
    """
    graph = DataRelationshipGraph(df, dataset_name=dataset_name)
    split = graph.get_measure_dimension_split()
    star_schema = graph.suggest_star_schema()
    topology = graph.to_graph_dict()
    join_keys = graph.find_candidate_join_keys()

    return {
        "measure_dimension_split": split,
        "star_schema": star_schema,
        "topology": topology,
        "candidate_join_keys": join_keys,
        "total_nodes": topology["meta"]["total_nodes"],
        "total_edges": topology["meta"]["total_edges"],
    }
