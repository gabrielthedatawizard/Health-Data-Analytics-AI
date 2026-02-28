import pytest

from backend.llm_gate import FactsGroundingError, SchemaValidationError, validate_facts_references, validate_schema
from backend.llm_schemas import DASHBOARD_SPEC_SCHEMA


def test_schema_validation_rejects_invalid_dashboard_spec() -> None:
    invalid_spec = {
        "title": "Spec",
        "kpis": [],
        "filters": [],
        "charts": [],
        # missing required insight_cards
    }
    with pytest.raises(SchemaValidationError):
        validate_schema(invalid_spec, DASHBOARD_SPEC_SCHEMA)


def test_facts_grounding_rejects_unknown_keys() -> None:
    output = {
        "title": "Spec",
        "kpis": [{"name": "Rows", "fact_id": "fact_missing"}],
        "filters": [],
        "charts": [],
        "insight_cards": [{"title": "Insight", "text": "Text", "fact_ids": ["fact_missing"]}],
    }
    facts_bundle = {
        "dataset_id": "ds_test",
        "dataset_hash": "abc",
        "created_at": "2026-01-01T00:00:00Z",
        "data_coverage": {
            "mode": "full",
            "rows_total": 10,
            "rows_used": 10,
            "sampling_method": "uniform",
            "seed": None,
            "bias_notes": "test",
        },
        "profiling": {"shape": {}, "dtypes": {}, "missing_percent": {}, "pii_candidates": []},
        "quality": {"score": 90, "issues": []},
        "kpis": [{"id": "kpi_rows", "name": "Rows", "value": 10, "unit": "rows", "facts_refs": ["fact_001"]}],
        "insight_facts": [{"id": "fact_001", "type": "comparison", "value": {"metric": "row_count", "value": 10}, "evidence": {}}],
        "chart_candidates": [],
    }
    with pytest.raises(FactsGroundingError):
        validate_facts_references(output, facts_bundle)
