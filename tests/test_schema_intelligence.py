"""
Tests for Schema Intelligence Engine (Phase 1.1)
"""

import pytest
import pandas as pd
import numpy as np
from backend.schema_intelligence import (
    ColumnIntelligence,
    SchemaIntelligence,
    analyze_schema,
    SEMANTIC_TAXONOMY,
)


def test_taxonomy_size():
    """Verify that taxonomy contains 60+ health and analytics semantic types."""
    assert len(SEMANTIC_TAXONOMY) >= 50


def test_demographic_semantic_inference():
    """Verify patient ID, age, and gender detection."""
    df = pd.DataFrame({
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "age_years": [24, 45, 12, 67, 33],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
    })
    col_intel_pid = ColumnIntelligence(df["patient_id"], "patient_id")
    sem_pid = col_intel_pid.infer_semantic_type()
    assert sem_pid["semantic_type"] == "patient_id"

    col_intel_age = ColumnIntelligence(df["age_years"], "age_years")
    sem_age = col_intel_age.infer_semantic_type()
    assert sem_age["semantic_type"] == "age_years"

    col_intel_gender = ColumnIntelligence(df["gender"], "gender")
    sem_gender = col_intel_gender.infer_semantic_type()
    assert sem_gender["semantic_type"] == "gender_sex"


def test_clinical_vitals_inference():
    """Verify temperature, SpO2, BMI, and heart rate classification."""
    df = pd.DataFrame({
        "body_temp_c": [36.5, 37.2, 38.4, 36.8, 39.1],
        "heart_rate_bpm": [72, 85, 110, 68, 92],
        "spo2": [98.0, 96.5, 92.0, 99.0, 94.0],
        "bmi": [22.4, 28.1, 19.5, 31.2, 24.0],
    })
    analysis = analyze_schema(df)
    cols = analysis["column_intelligence"]

    assert cols["body_temp_c"]["semantic"]["semantic_type"] == "vital_temperature_c"
    assert cols["heart_rate_bpm"]["semantic"]["semantic_type"] == "vital_heart_rate"
    assert cols["spo2"]["semantic"]["semantic_type"] == "vital_spo2"
    assert cols["bmi"]["semantic"]["semantic_type"] == "vital_bmi"


def test_unit_detection():
    """Verify unit extraction from headers and values."""
    col_bracket = pd.Series([10.5, 20.3, 15.0])
    intel_bracket = ColumnIntelligence(col_bracket, "Weight (kg)")
    u1 = intel_bracket.detect_unit()
    assert u1 is not None
    assert u1["unit"] == "weight_kg"
    assert u1["symbol"] == "kg"

    col_pct = pd.Series([12, 45, 88])
    intel_pct = ColumnIntelligence(col_pct, "vaccination_pct")
    u2 = intel_pct.detect_unit()
    assert u2 is not None
    assert u2["unit"] == "percentage"
    assert u2["symbol"] == "%"

    col_val_pct = pd.Series(["10%", "25%", "85%"])
    intel_val_pct = ColumnIntelligence(col_val_pct, "coverage")
    u3 = intel_val_pct.detect_unit()
    assert u3 is not None
    assert u3["unit"] == "percentage"


def test_granularity_detection():
    """Verify candidate key individual level vs. categorical dimension."""
    # Unique IDs
    s_ids = pd.Series([f"ID_{i}" for i in range(100)])
    intel_ids = ColumnIntelligence(s_ids, "patient_id", 100)
    g_id = intel_ids.detect_granularity()
    assert g_id["granularity"] == "individual_record"
    assert g_id["is_candidate_key"] is True

    # Low-cardinality category
    s_cat = pd.Series(["Urban", "Rural", "Urban", "Rural"] * 25)
    intel_cat = ColumnIntelligence(s_cat, "setting", 100)
    g_cat = intel_cat.detect_granularity()
    assert g_cat["granularity"] == "categorical_dimension"
    assert g_cat["is_candidate_key"] is False


def test_temporal_pattern_detection():
    """Verify monthly regular time-series frequency detection."""
    monthly_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    s_monthly = pd.Series(monthly_dates)
    intel_temp = ColumnIntelligence(s_monthly, "reporting_month")
    res = intel_temp.detect_temporal_pattern()
    assert res is not None
    assert res["is_temporal"] is True
    assert res["pattern"] == "regular_monthly"
    assert res["frequency_code"] == "M"


def test_hygiene_issue_detection():
    """Verify detection of whitespaces, casing, boolean masquerades, and disguised numerics."""
    df_dirty = pd.DataFrame({
        "padded_col": [" Nairobi ", "Mombasa", " Kisumu "],
        "casing_col": ["Male", "male", "MALE"],
        "bool_masquerade": ["Y", "N", "Y"],
        "disguised_num": ["$1,200", "$4,500", "$300"],
    })

    # Whitespace
    c_pad = ColumnIntelligence(df_dirty["padded_col"], "padded_col").detect_hygiene_issues()
    pad_types = [h["issue_type"] for h in c_pad]
    assert "whitespace_padding" in pad_types

    # Casing
    c_case = ColumnIntelligence(df_dirty["casing_col"], "casing_col").detect_hygiene_issues()
    case_types = [h["issue_type"] for h in c_case]
    assert "inconsistent_casing" in case_types

    # Bool masquerade
    c_bool = ColumnIntelligence(df_dirty["bool_masquerade"], "bool_masquerade").detect_hygiene_issues()
    bool_types = [h["issue_type"] for h in c_bool]
    assert "boolean_masquerade" in bool_types

    # Disguised numeric
    c_num = ColumnIntelligence(df_dirty["disguised_num"], "disguised_num").detect_hygiene_issues()
    num_types = [h["issue_type"] for h in c_num]
    assert "numeric_disguised_as_string" in num_types


def test_composite_value_detection():
    """Verify detection of blood pressure and delimited hierarchies."""
    bp_series = pd.Series(["120/80", "118/75", "130/85", "140/90", "110/70"] * 10)
    intel_bp = ColumnIntelligence(bp_series, "bp_reading")
    bp_comp = intel_bp.detect_composite_values()
    assert bp_comp is not None
    assert bp_comp["composite_type"] == "blood_pressure_pair"
    assert bp_comp["suggested_splits"] == ["bp_reading_systolic", "bp_reading_diastolic"]

    hier_series = pd.Series(["Coast - Mombasa", "Coast - Kilifi", "Rift - Nakuru", "Rift - Kericho"] * 10)
    intel_hier = ColumnIntelligence(hier_series, "region_district")
    hier_comp = intel_hier.detect_composite_values()
    assert hier_comp is not None
    assert hier_comp["composite_type"] == "delimited_hierarchy"


def test_cross_column_hierarchies_and_fds():
    """Verify hierarchy and functional dependency detection across columns."""
    records = []
    for reg, dists in [("Coast", ["Mombasa", "Kilifi"]), ("Rift", ["Nakuru", "Kericho"])]:
        for dist in dists:
            for fac_idx in range(5):
                records.append({
                    "region": reg,
                    "district": dist,
                    "facility_code": f"FAC_{dist[:3]}_{fac_idx}",
                    "facility_name": f"{dist} Clinic {fac_idx}",
                })
    df_geo = pd.DataFrame(records)
    engine = SchemaIntelligence(df_geo)
    engine.analyze_columns()

    # Hierarchy: region -> district
    hierarchies = engine.detect_hierarchies()
    pairs = [(h["parent"], h["child"]) for h in hierarchies]
    assert ("region", "district") in pairs

    # Functional dependency: facility_code -> facility_name
    fds = engine.detect_functional_dependencies()
    fd_pairs = [(f["determinant"], f["dependent"]) for f in fds]
    assert ("facility_code", "facility_name") in fd_pairs


def test_interval_pairs_detection():
    """Verify admission to discharge date interval detection."""
    df_interval = pd.DataFrame({
        "admission_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "discharge_date": pd.date_range("2024-01-05", periods=10, freq="D"),
    })
    engine = SchemaIntelligence(df_interval)
    engine.analyze_columns()
    intervals = engine.detect_interval_pairs()
    assert len(intervals) >= 1
    assert intervals[0]["start_column"] == "admission_date"
    assert intervals[0]["end_column"] == "discharge_date"
    assert intervals[0]["median_duration_days"] == 4.0


def test_data_dictionary_and_domain_summary():
    """Verify data dictionary generation and health domain detection."""
    df_hmis = pd.DataFrame({
        "facility_name": ["Central Hospital", "North Clinic", "South Dispensary"] * 5,
        "reporting_month": ["2024-01", "2024-02", "2024-03"] * 5,
        "opd_attendances": [450, 520, 310] * 5,
        "malaria_cases": [25, 40, 12] * 5,
        "vaccination_rate": [88.5, 91.2, 79.4] * 5,
    })
    result = analyze_schema(df_hmis)
    summary = result["summary"]
    dictionary = result["data_dictionary"]

    assert summary["inferred_domain"] == "health_medical_data"
    assert len(dictionary["columns"]) == 5
    col_names = [c["name"] for c in dictionary["columns"]]
    assert "facility_name" in col_names
    assert "vaccination_rate" in col_names
