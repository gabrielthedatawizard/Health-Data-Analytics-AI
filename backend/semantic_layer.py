from __future__ import annotations

import re
from typing import Any

NUMERIC_OPS = ("sum", "mean", "min", "max", "count")
NON_NUMERIC_OPS = ("count",)
FILTER_OPS_BY_KIND = {
    "numeric": {"eq", "neq", "gt", "gte", "lt", "lte", "between", "in"},
    "categorical": {"eq", "neq", "in"},
    "date": {"eq", "neq", "gt", "gte", "lt", "lte", "between"},
}
PREFERRED_DIMENSION_KEYWORDS = (
    "facility",
    "payer",
    "district",
    "region",
    "diagnosis",
    "service",
    "department",
    "provider",
    "sex",
    "gender",
    "age",
)
PREFERRED_TIME_KEYWORDS = ("date", "month", "week", "year", "period", "admission", "discharge")
STRICT_TIME_BLOCK_KEYWORDS = ("birth", "dob")


class QueryPlanValidationError(ValueError):
    pass


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "field"


def _as_strings(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    if not values:
        return []
    return [str(value) for value in values]


def _field_kind(field: str, numeric_fields: set[str], categorical_fields: set[str], datetime_fields: set[str]) -> str | None:
    if field in numeric_fields:
        return "numeric"
    if field in datetime_fields:
        return "date"
    if field in categorical_fields:
        return "categorical"
    return None


def _preferred_field(fields: list[str], keywords: tuple[str, ...]) -> str | None:
    if not fields:
        return None
    lowered = [(field, field.lower()) for field in fields]
    for keyword in keywords:
        for field, lowered_field in lowered:
            if keyword in lowered_field:
                return field
    return fields[0]


def build_semantic_layer(profile: dict[str, Any]) -> dict[str, Any]:
    numeric_fields = _as_strings(profile.get("numeric_cols"))
    categorical_fields = _as_strings(profile.get("categorical_cols"))
    datetime_fields = _as_strings(profile.get("datetime_cols"))
    pii_fields = {str(field) for field in profile.get("pii_candidates", [])}
    restricted_time_fields = {
        field
        for field in datetime_fields
        if field in pii_fields and any(keyword in field.lower() for keyword in STRICT_TIME_BLOCK_KEYWORDS)
    }
    analytics_datetime_fields = [field for field in datetime_fields if field not in restricted_time_fields]
    blocked_fields = (pii_fields - set(analytics_datetime_fields)) | restricted_time_fields

    safe_numeric = [field for field in numeric_fields if field not in blocked_fields]
    safe_categorical = [field for field in categorical_fields if field not in blocked_fields]
    safe_datetime = [field for field in analytics_datetime_fields if field not in blocked_fields]

    metrics: list[dict[str, Any]] = []
    for field in safe_numeric:
        slug = _slugify(field)
        metrics.extend(
            {
                "id": f"{op}_{slug}",
                "label": f"{op.upper()} {field}",
                "field": field,
                "op": op,
                "kind": "numeric",
                "aliases": [f"{slug}_{op}"],
            }
            for op in NUMERIC_OPS
        )

    # Count is still useful when no numeric measure is present.
    if not metrics:
        for field in safe_categorical + safe_datetime:
            slug = _slugify(field)
            metrics.append(
                {
                    "id": f"count_{slug}",
                    "label": f"COUNT {field}",
                    "field": field,
                    "op": "count",
                    "kind": "categorical" if field in safe_categorical else "date",
                    "aliases": [f"{slug}_count"],
                }
            )

    dimensions = [
        {
            "field": field,
            "kind": "categorical" if field in safe_categorical else "date",
            "allowed_ops": list(FILTER_OPS_BY_KIND["categorical" if field in safe_categorical else "date"]),
        }
        for field in safe_categorical + safe_datetime
    ]
    filters = dimensions + [
        {
            "field": field,
            "kind": "numeric",
            "allowed_ops": list(FILTER_OPS_BY_KIND["numeric"]),
        }
        for field in safe_numeric
    ]

    return {
        "metrics": metrics,
        "dimensions": dimensions,
        "filters": filters,
        "pii_blocked_fields": sorted(blocked_fields),
        "restricted_time_fields": sorted(restricted_time_fields),
        "default_time_field": _preferred_field(safe_datetime, PREFERRED_TIME_KEYWORDS),
        "preferred_dimension": _preferred_field([item["field"] for item in dimensions], PREFERRED_DIMENSION_KEYWORDS),
    }


def semantic_prompt_context(semantic_layer: dict[str, Any], *, max_metrics: int = 24, max_dimensions: int = 24) -> dict[str, Any]:
    metrics = semantic_layer.get("metrics", [])[:max_metrics]
    dimensions = semantic_layer.get("dimensions", [])[:max_dimensions]
    filters = semantic_layer.get("filters", [])[:max_dimensions]
    return {
        "metrics": [
            {
                "metric_id": item["id"],
                "label": item["label"],
                "field": item["field"],
                "op": item["op"],
            }
            for item in metrics
        ],
        "dimensions": [{"field": item["field"], "kind": item["kind"]} for item in dimensions],
        "filters": [{"field": item["field"], "kind": item["kind"], "allowed_ops": item["allowed_ops"]} for item in filters],
        "default_time_field": semantic_layer.get("default_time_field"),
        "preferred_dimension": semantic_layer.get("preferred_dimension"),
        "pii_blocked_fields": semantic_layer.get("pii_blocked_fields", []),
    }


def validate_and_resolve_query_plan(
    plan: dict[str, Any],
    profile: dict[str, Any],
    semantic_layer: dict[str, Any],
) -> dict[str, Any]:
    numeric_fields = set(_as_strings(profile.get("numeric_cols")))
    categorical_fields = set(_as_strings(profile.get("categorical_cols")))
    datetime_fields = set(_as_strings(profile.get("datetime_cols")))
    blocked_fields = set(semantic_layer.get("pii_blocked_fields", []))
    dimension_fields = {item["field"] for item in semantic_layer.get("dimensions", [])}
    filter_index = {item["field"]: item for item in semantic_layer.get("filters", [])}

    metric_index: dict[str, dict[str, Any]] = {}
    metric_lookup_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in semantic_layer.get("metrics", []):
        metric_index[str(metric["id"])] = metric
        for alias in metric.get("aliases", []):
            metric_index[str(alias)] = metric
        metric_lookup_by_pair[(str(metric["field"]), str(metric["op"]))] = metric

    resolved_metrics: list[dict[str, Any]] = []
    for metric in plan.get("metrics", []):
        metric_id = metric.get("metric_id")
        if isinstance(metric_id, str) and metric_id:
            definition = metric_index.get(metric_id)
            if not definition:
                raise QueryPlanValidationError(f"Unknown metric_id '{metric_id}'.")
            resolved_metrics.append(
                {
                    "metric_id": str(definition["id"]),
                    "field": str(definition["field"]),
                    "op": str(definition["op"]),
                }
            )
            continue

        field = str(metric.get("field", ""))
        op = str(metric.get("op", ""))
        if not field:
            raise QueryPlanValidationError("Metric field is required.")
        if field in blocked_fields:
            raise QueryPlanValidationError(f"Field '{field}' is blocked because it is PII-sensitive.")
        field_kind = _field_kind(field, numeric_fields, categorical_fields, datetime_fields)
        if field_kind is None:
            raise QueryPlanValidationError(f"Unknown metric field '{field}'.")
        allowed_ops = NUMERIC_OPS if field_kind == "numeric" else NON_NUMERIC_OPS
        if op not in allowed_ops:
            raise QueryPlanValidationError(f"Operation '{op}' is not allowed for {field_kind} field '{field}'.")
        resolved_metric: dict[str, Any] = {"field": field, "op": op}
        definition = metric_lookup_by_pair.get((field, op))
        if definition:
            resolved_metric["metric_id"] = str(definition["id"])
        resolved_metrics.append(resolved_metric)

    resolved_group_by: list[str] = []
    for field in plan.get("group_by", []):
        name = str(field)
        if name in blocked_fields:
            raise QueryPlanValidationError(f"Field '{name}' is blocked because it is PII-sensitive.")
        if name not in dimension_fields:
            raise QueryPlanValidationError(f"Field '{name}' is not an approved grouping dimension.")
        resolved_group_by.append(name)

    resolved_filters: list[dict[str, Any]] = []
    for flt in plan.get("filters", []):
        field = str(flt.get("field", ""))
        op = str(flt.get("op", ""))
        if not field:
            raise QueryPlanValidationError("Filter field is required.")
        if field in blocked_fields:
            raise QueryPlanValidationError(f"Field '{field}' is blocked because it is PII-sensitive.")
        definition = filter_index.get(field)
        if not definition:
            raise QueryPlanValidationError(f"Field '{field}' is not an approved filter field.")
        if op not in set(definition.get("allowed_ops", [])):
            raise QueryPlanValidationError(f"Operation '{op}' is not allowed for filter field '{field}'.")
        resolved_filters.append({"field": field, "op": op, "value": flt.get("value")})

    time_filter = dict(plan.get("time") or {})
    time_field = time_filter.get("field")
    if isinstance(time_field, str) and time_field:
        if time_field in blocked_fields:
            raise QueryPlanValidationError(f"Field '{time_field}' is blocked because it is PII-sensitive.")
        if time_field not in datetime_fields:
            raise QueryPlanValidationError(f"Field '{time_field}' is not an approved time field.")
    elif time_filter.get("grain") and semantic_layer.get("default_time_field"):
        time_filter["field"] = semantic_layer["default_time_field"]

    return {
        "intent": str(plan.get("intent", "aggregate")),
        "metrics": resolved_metrics,
        "group_by": resolved_group_by,
        "filters": resolved_filters,
        "time": {
            "field": time_filter.get("field"),
            "grain": time_filter.get("grain"),
            "start": time_filter.get("start"),
            "end": time_filter.get("end"),
        },
        "limit": int(plan.get("limit", 1000)),
        "chart_hint": str(plan.get("chart_hint", "table")),
    }
