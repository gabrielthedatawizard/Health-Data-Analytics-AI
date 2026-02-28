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
        "insight_facts": [{"id": "fact_001", "type": "comparison", "value": {"metric": "row_count", "value": 10}, "evidence": {}}],
    }
    spec = {"title": "health_report", "charts": [], "insight_cards": []}

    html = _render_html_report(dataset_id, meta, profile, facts_bundle, spec)
    assert "Methods & Limitations" in html
