from backend.main import _render_html_report


def test_report_template_contains_limitations() -> None:
    dataset_id = "ds_test"
    meta = {"created_by": "tester", "pii_masking_enabled": False}
    profile = {
        "shape": {"rows": 10, "cols": 2},
        "quality_score": 90,
        "duplicate_rows": 0,
        "missing_percent": {"a": 0.0},
    }
    facts_bundle = {
        "facts": [{"metric": "row_count", "value": 10}],
        "insights": [],
    }
    spec = {"template": "health_report", "charts": []}

    html = _render_html_report(dataset_id, meta, profile, facts_bundle, spec)
    assert "Methods & Limitations" in html
