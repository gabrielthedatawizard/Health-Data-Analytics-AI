from __future__ import annotations

FACTS_BUNDLE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "dataset_id",
        "dataset_hash",
        "created_at",
        "data_coverage",
        "profiling",
        "quality",
        "kpis",
        "insight_facts",
        "chart_candidates",
    ],
    "properties": {
        "dataset_id": {"type": "string", "minLength": 1},
        "dataset_hash": {"type": "string", "minLength": 1},
        "created_at": {"type": "string", "minLength": 1},
        "data_coverage": {
            "type": "object",
            "additionalProperties": False,
            "required": ["mode", "rows_total", "rows_used", "sampling_method", "seed", "bias_notes"],
            "properties": {
                "mode": {"type": "string", "enum": ["sample", "full"]},
                "rows_total": {"type": "integer", "minimum": 0},
                "rows_used": {"type": "integer", "minimum": 0},
                "sampling_method": {"type": "string", "enum": ["time_stratified", "stratified", "uniform"]},
                "seed": {"type": ["integer", "null"]},
                "bias_notes": {"type": "string"},
            },
        },
        "profiling": {
            "type": "object",
            "additionalProperties": True,
            "required": ["shape", "dtypes", "missing_percent", "pii_candidates"],
            "properties": {
                "shape": {"type": "object"},
                "dtypes": {"type": "object"},
                "missing_percent": {"type": "object"},
                "pii_candidates": {"type": "array", "items": {"type": "string"}},
            },
        },
        "quality": {
            "type": "object",
            "additionalProperties": False,
            "required": ["score", "issues"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["code", "severity", "message", "columns"],
                        "properties": {
                            "code": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "message": {"type": "string"},
                            "columns": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "kpis": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "name", "value", "unit", "facts_refs"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "value": {},
                    "unit": {"type": "string"},
                    "facts_refs": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "insight_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "type", "value", "evidence"],
                "properties": {
                    "id": {"type": "string"},
                    "citation": {"type": "string"},
                    "value": {},
                    "evidence": {"type": "object"},
                    "type": {"type": "string", "enum": ["trend", "outlier", "distribution", "comparison"]},
                },
            },
        },
        "chart_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "chart_type", "x", "y", "group_by", "filters", "score"],
                "properties": {
                    "id": {"type": "string"},
                    "chart_type": {"type": "string"},
                    "x": {"type": ["string", "null"]},
                    "y": {"type": ["string", "null"]},
                    "group_by": {"type": ["string", "null"]},
                    "filters": {"type": "array", "items": {"type": "object"}},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


DASHBOARD_SPEC_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "kpis", "filters", "charts", "insight_cards"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "kpis": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "fact_id"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "fact_id": {"type": "string", "minLength": 1},
                },
            },
        },
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "type"],
                "properties": {
                    "field": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "enum": ["categorical", "date", "numeric"]},
                },
            },
        },
        "charts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title",
                    "type",
                    "x",
                    "y",
                    "group_by",
                    "aggregation",
                    "fact_ids",
                    "layout",
                ],
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "enum": ["line", "bar", "hist", "scatter", "heatmap", "table"]},
                    "x": {"type": ["string", "null"]},
                    "y": {"type": ["string", "null"]},
                    "group_by": {"type": ["string", "null"]},
                    "aggregation": {"type": "string", "enum": ["sum", "mean", "count"]},
                    "fact_ids": {"type": "array", "items": {"type": "string"}},
                    "layout": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["row", "col", "w", "h"],
                        "properties": {
                            "row": {"type": "integer"},
                            "col": {"type": "integer"},
                            "w": {"type": "integer"},
                            "h": {"type": "integer"},
                        },
                    },
                },
            },
        },
        "insight_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "text", "fact_ids"],
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "fact_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


QUERY_PLAN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "metrics", "group_by", "filters", "time", "limit", "chart_hint"],
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["aggregate", "trend", "compare", "distribution", "find_outliers"],
        },
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "field"],
                "properties": {
                    "op": {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]},
                    "field": {"type": "string"},
                },
            },
        },
        "group_by": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "op", "value"],
                "properties": {
                    "field": {"type": "string"},
                    "op": {"type": "string", "enum": ["eq", "neq", "gt", "gte", "lt", "lte", "in", "between"]},
                    "value": {},
                },
            },
        },
        "time": {
            "type": "object",
            "additionalProperties": False,
            "required": ["field", "grain", "start", "end"],
            "properties": {
                "field": {"type": ["string", "null"]},
                "grain": {"type": ["string", "null"], "enum": ["day", "week", "month", "year", None]},
                "start": {"type": ["string", "null"]},
                "end": {"type": ["string", "null"]},
            },
        },
        "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
        "chart_hint": {"type": "string", "enum": ["line", "bar", "table", "hist", "scatter"]},
    },
}


ASK_NARRATIVE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "fact_ids"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "fact_ids": {"type": "array", "items": {"type": "string"}},
    },
}
