from __future__ import annotations

from typing import Any

from jsonschema import ValidationError, validate


class SchemaValidationError(ValueError):
    pass


class FactsGroundingError(ValueError):
    pass


def validate_schema(output: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        validate(instance=output, schema=schema)
    except ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc


def _collect_dot_paths(obj: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys.update(_collect_dot_paths(value, path))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_dot_paths(item, prefix))
    return keys


def _extract_citations(output: Any) -> set[str]:
    citations: set[str] = set()
    if isinstance(output, dict):
        for key, value in output.items():
            if key in {"citation", "citation_key"} and isinstance(value, str):
                citations.add(value)
            elif key in {"facts_used", "requested_facts", "citations"} and isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        citations.add(item)
            else:
                citations.update(_extract_citations(value))
    elif isinstance(output, list):
        for item in output:
            citations.update(_extract_citations(item))
    return citations


def _allowed_fact_keys(facts_bundle: dict[str, Any]) -> set[str]:
    keys = _collect_dot_paths(facts_bundle)
    for fact in facts_bundle.get("facts", []):
        fact_id = fact.get("id")
        metric = fact.get("metric")
        if isinstance(fact_id, str):
            keys.add(fact_id)
        if isinstance(metric, str):
            keys.add(metric)
            keys.add(f"facts.{metric}")
    for key in facts_bundle.get("facts_index", {}).keys():
        if isinstance(key, str):
            keys.add(key)
            keys.add(f"facts.{key}")
    return keys


def validate_facts_references(output: dict[str, Any], facts_bundle: dict[str, Any]) -> None:
    citations = _extract_citations(output)
    if not citations:
        raise FactsGroundingError("Missing facts citations in LLM output.")

    allowed = _allowed_fact_keys(facts_bundle)
    missing = sorted(citation for citation in citations if citation not in allowed)
    if missing:
        raise FactsGroundingError(f"Unknown facts references: {missing}")
