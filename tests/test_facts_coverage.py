from backend.main import _build_facts_bundle


def test_facts_bundle_has_coverage() -> None:
    profile = {
        "shape": {"rows": 10, "cols": 1},
        "quality_score": 90,
        "duplicate_rows": 0,
        "duplicate_percent": 0.0,
        "missing_percent": {"a": 0.0},
        "columns": [],
        "datetime_cols": [],
        "numeric_cols": [],
        "categorical_cols": [],
        "pii_candidates": [],
        "health_signals": {},
        "data_coverage": {
            "mode": "sample",
            "rows_total": 10,
            "rows_used": 5,
            "sampling_method": "seeded_sample",
            "seed": 42,
            "bias_notes": "test",
        },
    }
    facts = _build_facts_bundle(df=None, profile=profile)  # type: ignore[arg-type]
    assert "data_coverage" in facts
