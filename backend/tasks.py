from __future__ import annotations

import json
from pathlib import Path

from weasyprint import HTML

from backend.celery_app import celery_app
from backend.jobs import update_job
from backend.main import (
    _build_dashboard_spec,
    _build_facts_bundle,
    _build_profile,
    _deterministic_sample,
    _chart_images_base64,
    _dashboard_spec_path,
    _facts_path,
    _load_json,
    _load_meta,
    _mask_recursive,
    _profile_path,
    _read_uploaded_file,
    _render_html_report,
    _report_path,
    _save_json,
    _save_meta,
    _apply_profile_masking,
    MID_COL_MAX,
    MID_ROW_MAX,
    MID_ROW_THRESHOLD,
    SAMPLE_MAX_ROWS,
)


def _report_pdf_path(dataset_id: str) -> Path:
    return _report_path(dataset_id).with_suffix(".pdf")


@celery_app.task(name="generate_report_task")
def generate_report_task(job_id: str, dataset_id: str, template: str | None, sections: list[str] | None) -> dict:
    try:
        meta = _load_meta(dataset_id)
        if not meta.get("file"):
            raise ValueError("No file uploaded for this session.")

        update_job(job_id, status="running", progress=0.1)

        profile_path = _profile_path(dataset_id)
        facts_path = _facts_path(dataset_id)
        spec_path = _dashboard_spec_path(dataset_id)

        if profile_path.exists():
            profile = _load_json(profile_path)
        else:
            df = _read_uploaded_file(meta)
            profile = _build_profile(df)
            _save_json(profile_path, profile)

        if "df" not in locals():
            df = _read_uploaded_file(meta)

        update_job(job_id, status="running", progress=0.35)

        if facts_path.exists():
            facts_bundle = _load_json(facts_path)
        else:
            facts_bundle = _build_facts_bundle(df, profile)
            _save_json(facts_path, facts_bundle)

        update_job(job_id, status="running", progress=0.55)

        if spec_path.exists():
            spec = _load_json(spec_path)
        else:
            spec = _build_dashboard_spec(dataset_id, profile=profile, facts_bundle=facts_bundle)
            _save_json(spec_path, spec)

        if profile.get("pii_candidates") and not meta.get("allow_sensitive_export", False):
            aliases = {column: f"pii_field_{index + 1}" for index, column in enumerate(profile["pii_candidates"])}
            profile = _apply_profile_masking(profile, True)
            facts_bundle = json.loads(json.dumps(facts_bundle))
            facts_bundle["facts"] = [
                {**fact, "value": _mask_recursive(fact.get("value"), aliases)}
                for fact in facts_bundle.get("facts", [])
            ]

        chart_images = _chart_images_base64(df, spec.get("charts", []))
        report_html = _render_html_report(dataset_id, meta, profile, facts_bundle, spec, chart_images)
        html_path = _report_path(dataset_id)
        html_path.write_text(report_html, encoding="utf-8")

        update_job(job_id, status="running", progress=0.8)

        pdf_path = _report_pdf_path(dataset_id)
        HTML(string=report_html).write_pdf(str(pdf_path))

        meta["artifacts"]["profile"] = str(profile_path)
        meta["artifacts"]["facts"] = str(facts_path)
        meta["artifacts"]["dashboard_spec"] = str(spec_path)
        meta["artifacts"]["report_html"] = str(html_path)
        meta["artifacts"]["report_pdf"] = str(pdf_path)
        meta["status"] = "report_generated"
        _save_meta(dataset_id, meta)

        result = {"report_html_path": str(html_path), "report_pdf_path": str(pdf_path)}
        update_job(
            job_id,
            status="completed",
            progress=1.0,
            result=result,
            artifact_links={"report_html": str(html_path), "report_pdf": str(pdf_path)},
        )
        return result
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))
        raise


@celery_app.task(name="generate_facts_task")
def generate_facts_task(job_id: str, dataset_id: str, mode: str = "auto", seed: int = 42) -> dict:
    try:
        meta = _load_meta(dataset_id)
        if not meta.get("file"):
            raise ValueError("No file uploaded for this session.")

        update_job(job_id, status="running", progress=0.1)
        df = _read_uploaded_file(meta)
        rows, cols = df.shape

        if mode == "auto":
            if rows > MID_ROW_THRESHOLD or cols > MID_COL_MAX:
                mode = "sample"
            else:
                mode = "full"

        if mode == "sample":
            sampled = _deterministic_sample(df, max_rows=SAMPLE_MAX_ROWS, seed=seed)
            data_coverage = {
                "mode": "sample",
                "rows_total": rows,
                "rows_used": int(len(sampled)),
                "sampling_method": "seeded_sample",
                "seed": seed,
                "bias_notes": "Row-uniform sample; rare categories may be underrepresented.",
            }
            work_df = sampled
        else:
            data_coverage = {
                "mode": "full",
                "rows_total": rows,
                "rows_used": rows,
                "sampling_method": "none",
                "seed": None,
                "bias_notes": "Full dataset used.",
            }
            work_df = df

        update_job(job_id, status="running", progress=0.45)
        profile = _build_profile(work_df)
        profile["data_coverage"] = data_coverage

        update_job(job_id, status="running", progress=0.7)
        facts_bundle = _build_facts_bundle(work_df, profile)
        _save_json(_profile_path(dataset_id), profile)
        _save_json(_facts_path(dataset_id), facts_bundle)

        meta["artifacts"]["profile"] = str(_profile_path(dataset_id))
        meta["artifacts"]["facts"] = str(_facts_path(dataset_id))
        meta["status"] = "facts_generated"
        _save_meta(dataset_id, meta)

        result = {"facts_path": str(_facts_path(dataset_id)), "data_coverage": data_coverage}
        update_job(
            job_id,
            status="completed",
            progress=1.0,
            result=result,
            artifact_links={"facts": str(_facts_path(dataset_id))},
        )
        return result
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))
        raise
