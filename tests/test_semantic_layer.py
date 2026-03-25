import pytest

from backend.semantic_layer import QueryPlanValidationError, build_semantic_layer, validate_and_resolve_query_plan


def _profile() -> dict:
    return {
        "dtypes": {
            "visit_date": "datetime64[ns]",
            "district": "object",
            "payer": "object",
            "opd_visits": "int64",
            "patient_name": "object",
        },
        "numeric_cols": ["opd_visits"],
        "categorical_cols": ["district", "payer", "patient_name"],
        "datetime_cols": ["visit_date"],
        "pii_candidates": ["patient_name"],
    }


def test_build_semantic_layer_excludes_pii_and_exposes_metrics() -> None:
    semantic_layer = build_semantic_layer(_profile())

    metric_ids = {metric["id"] for metric in semantic_layer["metrics"]}
    dimension_fields = {dimension["field"] for dimension in semantic_layer["dimensions"]}

    assert "sum_opd_visits" in metric_ids
    assert "patient_name" not in dimension_fields
    assert semantic_layer["default_time_field"] == "visit_date"
    assert "patient_name" in semantic_layer["pii_blocked_fields"]


def test_validate_and_resolve_query_plan_with_metric_id() -> None:
    profile = _profile()
    semantic_layer = build_semantic_layer(profile)
    plan = {
        "intent": "compare",
        "metrics": [{"metric_id": "sum_opd_visits"}],
        "group_by": ["district"],
        "filters": [{"field": "payer", "op": "in", "value": ["Medicare"]}],
        "time": {"field": "visit_date", "grain": "month", "start": None, "end": None},
        "limit": 100,
        "chart_hint": "bar",
    }

    resolved = validate_and_resolve_query_plan(plan, profile, semantic_layer)

    assert resolved["metrics"] == [{"metric_id": "sum_opd_visits", "field": "opd_visits", "op": "sum"}]
    assert resolved["group_by"] == ["district"]
    assert resolved["filters"][0]["field"] == "payer"


def test_validate_and_resolve_query_plan_blocks_pii_fields() -> None:
    profile = _profile()
    semantic_layer = build_semantic_layer(profile)
    plan = {
        "intent": "aggregate",
        "metrics": [{"op": "count", "field": "patient_name"}],
        "group_by": [],
        "filters": [],
        "time": {"field": None, "grain": None, "start": None, "end": None},
        "limit": 25,
        "chart_hint": "table",
    }

    with pytest.raises(QueryPlanValidationError):
        validate_and_resolve_query_plan(plan, profile, semantic_layer)
