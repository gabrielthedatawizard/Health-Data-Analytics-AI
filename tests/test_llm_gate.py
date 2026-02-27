import pytest

from backend.llm_gate import FactsGroundingError, SchemaValidationError, validate_facts_references, validate_schema
from backend.llm_schemas import DASHBOARD_SPEC_SCHEMA


def test_schema_validation_rejects_invalid_dashboard_spec() -> None:
    invalid_spec = {
        "title": "Spec",
        "template": "health_core",
        "filters": [],
        "kpis": [],
        "charts": [],
        "components": [],
        # missing required facts_used
    }
    with pytest.raises(SchemaValidationError):
        validate_schema(invalid_spec, DASHBOARD_SPEC_SCHEMA)


def test_facts_grounding_rejects_unknown_keys() -> None:
    output = {
        "title": "Spec",
        "template": "health_core",
        "filters": [],
        "kpis": [],
        "charts": [],
        "components": [{"type": "kpi", "title": "Rows", "citation": "facts.unknown_metric"}],
        "facts_used": ["facts.unknown_metric"],
    }
    facts_bundle = {
        "facts": [{"id": "F001", "metric": "row_count", "value": 10}],
        "facts_index": {"row_count": {"id": "F001", "value": 10}},
    }
    with pytest.raises(FactsGroundingError):
        validate_facts_references(output, facts_bundle)
