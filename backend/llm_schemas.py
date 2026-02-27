from __future__ import annotations

DASHBOARD_SPEC_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "template", "filters", "kpis", "charts", "components", "facts_used"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "template": {"type": "string", "minLength": 1},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "column", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "column": {"type": "string"},
                    "type": {"type": "string", "enum": ["date_range", "multiselect", "category", "date"]},
                },
            },
        },
        "kpis": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "value", "format", "citation"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "value": {},
                    "format": {"type": "string"},
                    "citation": {"type": "string"},
                },
            },
        },
        "charts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "type", "title", "source", "citation"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["line", "bar", "bar_count", "histogram", "scatter"]},
                    "title": {"type": "string"},
                    "source": {"type": "object"},
                    "citation": {"type": "string"},
                },
            },
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "title", "citation"],
                "properties": {
                    "type": {"type": "string"},
                    "title": {"type": "string"},
                    "citation": {"type": "string"},
                    "value": {},
                    "source": {"type": "object"},
                },
            },
        },
        "facts_used": {"type": "array", "items": {"type": "string"}},
    },
}


QUERY_PLAN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question", "operations", "requested_facts"],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["op"],
                "properties": {
                    "op": {"type": "string", "enum": ["filter", "groupby", "aggregate"]},
                    "field": {"type": "string"},
                    "range": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "equals": {},
                    "func": {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]},
                },
            },
        },
        "requested_facts": {"type": "array", "items": {"type": "string"}},
    },
}


ASK_NARRATIVE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "facts_used"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "facts_used": {"type": "array", "items": {"type": "string"}},
    },
}
