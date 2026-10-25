"""
Schema Intelligence Engine for Health Data Analytics AI
Phase 1.1: Deep Data Understanding Engine

Provides deep, multi-signal semantic classification for health and operational data:
- 60+ specialized health and analytics semantic types
- Measurement unit detection (explicit and implicit)
- Data granularity and temporal pattern detection
- Data hygiene, encoding, and masquerade detection
- Composite value detection and split recommendations
- Cross-column relationships (hierarchies, functional dependencies, date intervals)
- Automated data dictionary generation
"""

from __future__ import annotations

import re
import math
from typing import Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


# =====================================================================
# 1. SEMANTIC TYPE TAXONOMY (60+ Types)
# =====================================================================

@dataclass
class SemanticTypeDefinition:
    type_id: str
    category: str
    description: str
    name_patterns: list[str]
    value_regex: Optional[str] = None
    expected_dtype: str = "any"  # numeric, categorical, text, datetime, boolean, any
    value_range: Optional[tuple[float, float]] = None


SEMANTIC_TAXONOMY: dict[str, SemanticTypeDefinition] = {
    # ── Identifiers & Demographics ──
    "patient_id": SemanticTypeDefinition(
        type_id="patient_id",
        category="demographics_id",
        description="Unique identifier for a patient or client",
        name_patterns=[r"\bpatient[_-]?id\b", r"\bclient[_-]?id\b", r"\bpid\b", r"\bsubject[_-]?id\b"],
        expected_dtype="any",
    ),
    "mrn": SemanticTypeDefinition(
        type_id="mrn",
        category="demographics_id",
        description="Medical Record Number / Hospital file number",
        name_patterns=[r"\bmrn\b", r"\bmed(ical)?[_-]?rec(ord)?[_-]?(no|num|id)?\b", r"\bhospital[_-]?no\b", r"\bopd[_-]?no\b", r"\bipd[_-]?no\b"],
        expected_dtype="any",
    ),
    "national_id": SemanticTypeDefinition(
        type_id="national_id",
        category="demographics_id",
        description="National citizen identification number or social security",
        name_patterns=[r"\bnational[_-]?id\b", r"\bnid\b", r"\bssn\b", r"\bcitizen[_-]?id\b", r"\bpassport[_-]?no\b"],
        expected_dtype="any",
    ),
    "facility_code": SemanticTypeDefinition(
        type_id="facility_code",
        category="demographics_id",
        description="Standardized health facility master code (e.g. MFL code, DHIS2 UID)",
        name_patterns=[r"\bfacility[_-]?code\b", r"\bmfl[_-]?code\b", r"\bdhis2[_-]?uid\b", r"\bsite[_-]?code\b", r"\bhospital[_-]?code\b"],
        expected_dtype="any",
    ),
    "facility_name": SemanticTypeDefinition(
        type_id="facility_name",
        category="demographics_id",
        description="Health facility, clinic, hospital or health center name",
        name_patterns=[r"\bfacility([_-]?name)?\b", r"\bhospital([_-]?name)?\b", r"\bhealth[_-]?center\b", r"\bclinic([_-]?name)?\b", r"\bdispensary\b"],
        expected_dtype="categorical",
    ),
    "provider_id": SemanticTypeDefinition(
        type_id="provider_id",
        category="demographics_id",
        description="Healthcare clinician, nurse, doctor or staff ID",
        name_patterns=[r"\bprovider[_-]?id\b", r"\bclinician[_-]?id\b", r"\bdoctor[_-]?id\b", r"\bnurse[_-]?id\b", r"\bstaff[_-]?id\b"],
        expected_dtype="any",
    ),
    "patient_name": SemanticTypeDefinition(
        type_id="patient_name",
        category="demographics_id",
        description="Name of patient (PII)",
        name_patterns=[r"\bpatient[_-]?name\b", r"\bclient[_-]?name\b", r"\bfirst[_-]?name\b", r"\blast[_-]?name\b", r"\bfull[_-]?name\b"],
        expected_dtype="text",
    ),
    "age_years": SemanticTypeDefinition(
        type_id="age_years",
        category="demographics",
        description="Patient age in completed years",
        name_patterns=[r"\bage\b", r"\bage[_-]?years?\b", r"\bpatient[_-]?age\b", r"\byrs?\b"],
        expected_dtype="numeric",
        value_range=(0, 125),
    ),
    "age_months": SemanticTypeDefinition(
        type_id="age_months",
        category="demographics",
        description="Infant / child age in months",
        name_patterns=[r"\bage[_-]?months?\b", r"\bmonths?[_-]?age\b", r"\bage[_-]?mo\b"],
        expected_dtype="numeric",
        value_range=(0, 60),
    ),
    "age_group": SemanticTypeDefinition(
        type_id="age_group",
        category="demographics",
        description="Categorical age cohorts (e.g. <5, 5-14, 15-24, 25-49, 50+)",
        name_patterns=[r"\bage[_-]?group\b", r"\bage[_-]?cohort\b", r"\bage[_-]?bracket\b", r"\bage[_-]?category\b", r"\bage[_-]?band\b"],
        expected_dtype="categorical",
    ),
    "gender_sex": SemanticTypeDefinition(
        type_id="gender_sex",
        category="demographics",
        description="Biological sex or gender identity",
        name_patterns=[r"\bgender\b", r"\bsex\b"],
        expected_dtype="categorical",
    ),
    "marital_status": SemanticTypeDefinition(
        type_id="marital_status",
        category="demographics",
        description="Marital or civil status (single, married, divorced, widowed)",
        name_patterns=[r"\bmarital[_-]?status\b", r"\bcivil[_-]?status\b", r"\bmarriage\b"],
        expected_dtype="categorical",
    ),
    "ethnicity_race": SemanticTypeDefinition(
        type_id="ethnicity_race",
        category="demographics",
        description="Ethnicity, race, or tribal group",
        name_patterns=[r"\bethnicity\b", r"\brace\b", r"\btribe\b"],
        expected_dtype="categorical",
    ),
    "education_level": SemanticTypeDefinition(
        type_id="education_level",
        category="demographics",
        description="Educational attainment level",
        name_patterns=[r"\beducation([_-]?level)?\b", r"\bschooling\b", r"\bliteracy\b"],
        expected_dtype="categorical",
    ),
    "phone_number": SemanticTypeDefinition(
        type_id="phone_number",
        category="demographics_pii",
        description="Contact telephone or mobile phone number",
        name_patterns=[r"\bphone\b", r"\bmobile\b", r"\bcell\b", r"\btel(ephone)?\b", r"\bcontact[_-]?no\b"],
        expected_dtype="any",
    ),
    "email_address": SemanticTypeDefinition(
        type_id="email_address",
        category="demographics_pii",
        description="Electronic mail address",
        name_patterns=[r"\bemail([_-]?address)?\b", r"\be[_-]?mail\b"],
        value_regex=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        expected_dtype="text",
    ),

    # ── Geographic & Administrative ──
    "country": SemanticTypeDefinition(
        type_id="country",
        category="geography",
        description="National administrative entity",
        name_patterns=[r"\bcountry([_-]?name)?\b", r"\bnation\b"],
        expected_dtype="categorical",
    ),
    "province_state": SemanticTypeDefinition(
        type_id="province_state",
        category="geography",
        description="First-level administrative division (Province, State)",
        name_patterns=[r"\bprovince\b", r"\bstate\b", r"\bregion\b"],
        expected_dtype="categorical",
    ),
    "district_county": SemanticTypeDefinition(
        type_id="district_county",
        category="geography",
        description="Second-level administrative division (District, County, LGA)",
        name_patterns=[r"\bdistrict\b", r"\bcounty\b", r"\blga\b", r"\bmunicipality\b", r"\bdepartment\b"],
        expected_dtype="categorical",
    ),
    "sub_county_ward": SemanticTypeDefinition(
        type_id="sub_county_ward",
        category="geography",
        description="Third-level local administrative unit (Sub-county, Ward, Sector)",
        name_patterns=[r"\bsub[_-]?county\b", r"\bward\b", r"\bsector\b", r"\bcommune\b", r"\bparish\b"],
        expected_dtype="categorical",
    ),
    "postal_code": SemanticTypeDefinition(
        type_id="postal_code",
        category="geography",
        description="Postal or ZIP code",
        name_patterns=[r"\bpostal([_-]?code)?\b", r"\bzip([_-]?code)?\b"],
        expected_dtype="any",
    ),
    "gps_latitude": SemanticTypeDefinition(
        type_id="gps_latitude",
        category="geography",
        description="Geographic latitude coordinate (-90 to +90)",
        name_patterns=[r"\blat(itude)?\b", r"\bgps[_-]?lat\b", r"\by[_-]?coord\b"],
        expected_dtype="numeric",
        value_range=(-90.0, 90.0),
    ),
    "gps_longitude": SemanticTypeDefinition(
        type_id="gps_longitude",
        category="geography",
        description="Geographic longitude coordinate (-180 to +180)",
        name_patterns=[r"\blon(gitude)?\b", r"\bgps[_-]?lon\b", r"\bx[_-]?coord\b", r"\blng\b"],
        expected_dtype="numeric",
        value_range=(-180.0, 180.0),
    ),
    "urban_rural": SemanticTypeDefinition(
        type_id="urban_rural",
        category="geography",
        description="Settlement classification (Urban, Peri-urban, Rural)",
        name_patterns=[r"\burban[_-]?rural\b", r"\bsetting\b", r"\blocality\b", r"\bsettlement\b"],
        expected_dtype="categorical",
    ),

    # ── Temporal ──
    "encounter_date": SemanticTypeDefinition(
        type_id="encounter_date",
        category="temporal",
        description="Date of patient clinical encounter or visit",
        name_patterns=[r"\bvisit[_-]?date\b", r"\bencounter[_-]?date\b", r"\bconsultation[_-]?date\b", r"\battendance[_-]?date\b"],
        expected_dtype="datetime",
    ),
    "admission_date": SemanticTypeDefinition(
        type_id="admission_date",
        category="temporal",
        description="Date of inpatient hospital admission",
        name_patterns=[r"\badmission[_-]?date\b", r"\badmit[_-]?date\b", r"\bdate[_-]?admitted\b"],
        expected_dtype="datetime",
    ),
    "discharge_date": SemanticTypeDefinition(
        type_id="discharge_date",
        category="temporal",
        description="Date of inpatient discharge or departure",
        name_patterns=[r"\bdischarge[_-]?date\b", r"\bdate[_-]?discharged\b"],
        expected_dtype="datetime",
    ),
    "birth_date": SemanticTypeDefinition(
        type_id="birth_date",
        category="temporal",
        description="Date of birth / DOB",
        name_patterns=[r"\bdob\b", r"\bbirth[_-]?date\b", r"\bdate[_-]?of[_-]?birth\b"],
        expected_dtype="datetime",
    ),
    "reporting_period": SemanticTypeDefinition(
        type_id="reporting_period",
        category="temporal",
        description="Reporting aggregation period (e.g. 2024-01, 2024Q1)",
        name_patterns=[r"\breporting[_-]?period\b", r"\bperiod\b", r"\bmonth[_-]?year\b", r"\byear[_-]?month\b"],
        expected_dtype="any",
    ),
    "reporting_year": SemanticTypeDefinition(
        type_id="reporting_year",
        category="temporal",
        description="Calendar or fiscal year (e.g. 2020-2030)",
        name_patterns=[r"\byear\b", r"\byr\b", r"\breporting[_-]?year\b", r"\bfiscal[_-]?year\b"],
        expected_dtype="numeric",
        value_range=(1900, 2100),
    ),
    "reporting_month": SemanticTypeDefinition(
        type_id="reporting_month",
        category="temporal",
        description="Calendar month (1-12 or Jan-Dec)",
        name_patterns=[r"\bmonth\b", r"\bmo\b", r"\breporting[_-]?month\b"],
        expected_dtype="any",
    ),
    "epidemiological_week": SemanticTypeDefinition(
        type_id="epidemiological_week",
        category="temporal",
        description="Epi week (1-53) for disease surveillance reporting",
        name_patterns=[r"\bepi[_-]?week\b", r"\bew\b", r"\bcdc[_-]?week\b", r"\bweek[_-]?no\b", r"\bweek[_-]?number\b"],
        expected_dtype="numeric",
        value_range=(1, 53),
    ),

    # ── Clinical & Diagnostic ──
    "icd10_code": SemanticTypeDefinition(
        type_id="icd10_code",
        category="clinical",
        description="ICD-10 clinical diagnostic code (e.g. A09, B20, J18.9)",
        name_patterns=[r"\bicd([_-]?10)?([_-]?code)?\b", r"\bdiagnosis[_-]?code\b"],
        value_regex=r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$",
        expected_dtype="any",
    ),
    "diagnosis_name": SemanticTypeDefinition(
        type_id="diagnosis_name",
        category="clinical",
        description="Clinical diagnosis, disease, or medical condition name",
        name_patterns=[r"\bdiagnosis\b", r"\bdisease\b", r"\bcondition\b", r"\bmorbidity\b", r"\bprimary[_-]?diagnosis\b"],
        expected_dtype="categorical",
    ),
    "symptom_chief_complaint": SemanticTypeDefinition(
        type_id="symptom_chief_complaint",
        category="clinical",
        description="Chief complaint, presenting symptoms, reason for visit",
        name_patterns=[r"\bcomplaint\b", r"\bchief[_-]?complaint\b", r"\bsymptom(s)?\b", r"\breason[_-]?for[_-]?visit\b"],
        expected_dtype="text",
    ),
    "triage_acuity": SemanticTypeDefinition(
        type_id="triage_acuity",
        category="clinical",
        description="Emergency triage category / acuity level (Red/Yellow/Green, ESI 1-5)",
        name_patterns=[r"\btriage([_-]?level|[_-]?acuity|[_-]?score)?\b", r"\bacuity\b", r"\besi\b"],
        expected_dtype="categorical",
    ),
    "vital_blood_pressure": SemanticTypeDefinition(
        type_id="vital_blood_pressure",
        category="clinical_vitals",
        description="Blood pressure reading (e.g. 120/80 mmHg or systolic/diastolic)",
        name_patterns=[r"\bbp\b", r"\bblood[_-]?pressure\b", r"\bsys(tolic)?\b", r"\bdia(stolic)?\b"],
        expected_dtype="any",
    ),
    "vital_heart_rate": SemanticTypeDefinition(
        type_id="vital_heart_rate",
        category="clinical_vitals",
        description="Heart rate / pulse in beats per minute (bpm)",
        name_patterns=[r"\bheart[_-]?rate\b", r"\bhr\b", r"\bpulse([_-]?rate)?\b", r"\bbpm\b"],
        expected_dtype="numeric",
        value_range=(20, 260),
    ),
    "vital_temperature_c": SemanticTypeDefinition(
        type_id="vital_temperature_c",
        category="clinical_vitals",
        description="Body temperature in degrees Celsius",
        name_patterns=[
            r"\btemp(erature)?\b",
            r"\btemp[_-]?c\b",
            r"\bbody[_-]?temp(erature)?([_-]?c)?\b",
            r"\bpatient[_-]?temp\b",
        ],
        expected_dtype="numeric",
        value_range=(30.0, 45.0),
    ),
    "vital_spo2": SemanticTypeDefinition(
        type_id="vital_spo2",
        category="clinical_vitals",
        description="Blood oxygen saturation percentage (SpO2)",
        name_patterns=[r"\bspo2\b", r"\boxygen[_-]?sat(uration)?\b", r"\bo2[_-]?sat\b"],
        expected_dtype="numeric",
        value_range=(50.0, 100.0),
    ),
    "vital_bmi": SemanticTypeDefinition(
        type_id="vital_bmi",
        category="clinical_vitals",
        description="Body Mass Index (BMI in kg/m2)",
        name_patterns=[r"\bbmi\b", r"\bbody[_-]?mass[_-]?index\b"],
        expected_dtype="numeric",
        value_range=(8.0, 80.0),
    ),
    "lab_test_name": SemanticTypeDefinition(
        type_id="lab_test_name",
        category="clinical_lab",
        description="Laboratory investigation name (e.g. Hemoglobin, Malaria RDT, HbA1c)",
        name_patterns=[r"\blab[_-]?test\b", r"\binvestigation\b", r"\bassid\b", r"\btest[_-]?name\b"],
        expected_dtype="categorical",
    ),
    "lab_numeric_result": SemanticTypeDefinition(
        type_id="lab_numeric_result",
        category="clinical_lab",
        description="Quantitative laboratory measurement",
        name_patterns=[r"\blab[_-]?result\b", r"\btest[_-]?result\b", r"\bquantitative[_-]?result\b", r"\btiter\b"],
        expected_dtype="numeric",
    ),
    "lab_qualitative_result": SemanticTypeDefinition(
        type_id="lab_qualitative_result",
        category="clinical_lab",
        description="Qualitative lab test result (Positive, Negative, Reactive, Indeterminate)",
        name_patterns=[r"\bresult[_-]?status\b", r"\btest[_-]?outcome\b", r"\bqualitative[_-]?result\b"],
        expected_dtype="categorical",
    ),
    "medication_name": SemanticTypeDefinition(
        type_id="medication_name",
        category="clinical_rx",
        description="Prescribed drug or pharmaceutical product name",
        name_patterns=[r"\bdrug([_-]?name)?\b", r"\bmedication([_-]?name)?\b", r"\bmedicine\b", r"\bpharmaceutical\b", r"\brx\b"],
        expected_dtype="categorical",
    ),
    "medication_dosage": SemanticTypeDefinition(
        type_id="medication_dosage",
        category="clinical_rx",
        description="Medication dosage or strength (e.g. 500mg, 10ml, 2x daily)",
        name_patterns=[r"\bdose\b", r"\bdosage\b", r"\bstrength\b", r"\bfrequency\b"],
        expected_dtype="any",
    ),
    "treatment_regimen": SemanticTypeDefinition(
        type_id="treatment_regimen",
        category="clinical_rx",
        description="Clinical regimen line (e.g. 1st-line ART, TB Cat-1, ACT)",
        name_patterns=[r"\bregimen\b", r"\btreatment[_-]?line\b", r"\bart[_-]?regimen\b", r"\btb[_-]?regimen\b"],
        expected_dtype="categorical",
    ),
    "outcome_status": SemanticTypeDefinition(
        type_id="outcome_status",
        category="clinical_outcome",
        description="Patient clinical outcome (discharged, transferred, recovered, died, default)",
        name_patterns=[r"\boutcome\b", r"\bdischarge[_-]?status\b", r"\bexit[_-]?status\b", r"\bdisposition\b"],
        expected_dtype="categorical",
    ),
    "mortality_flag": SemanticTypeDefinition(
        type_id="mortality_flag",
        category="clinical_outcome",
        description="Binary indicator or flag for patient death / deceased status",
        name_patterns=[r"\bdied\b", r"\bdeath\b", r"\bdeceased\b", r"\bmortality\b", r"\bfatal(ity)?\b"],
        expected_dtype="any",
    ),

    # ── Public Health & Surveillance ──
    "vaccine_name": SemanticTypeDefinition(
        type_id="vaccine_name",
        category="public_health",
        description="Vaccine antigen / immunization product (e.g. BCG, OPV, Penta, Measles, COVID-19)",
        name_patterns=[r"\bvaccine([_-]?name)?\b", r"\bantigen\b", r"\bimmunization\b", r"\bvac\b"],
        expected_dtype="categorical",
    ),
    "vaccine_dose": SemanticTypeDefinition(
        type_id="vaccine_dose",
        category="public_health",
        description="Dose sequence in vaccination series (Dose 1, Dose 2, Booster)",
        name_patterns=[r"\bdose[_-]?no\b", r"\bdose[_-]?number\b", r"\bvaccine[_-]?dose\b"],
        expected_dtype="any",
    ),
    "target_population": SemanticTypeDefinition(
        type_id="target_population",
        category="public_health",
        description="Denominator or target catchment population size",
        name_patterns=[r"\btarget[_-]?pop(ulation)?\b", r"\bcatchment[_-]?pop(ulation)?\b", r"\bdenominator\b"],
        expected_dtype="numeric",
        value_range=(0, 1e9),
    ),
    "coverage_rate": SemanticTypeDefinition(
        type_id="coverage_rate",
        category="public_health",
        description="Public health coverage percentage (e.g. vaccination rate, ANC coverage)",
        name_patterns=[r"\bcoverage([_-]?(pct|rate|percent))?\b", r"\bvaccination[_-]?rate\b", r"\buptake[_-]?rate\b"],
        expected_dtype="numeric",
        value_range=(0.0, 150.0),  # Some coverage rates exceed 100% due to mobile populations
    ),
    "case_status": SemanticTypeDefinition(
        type_id="case_status",
        category="public_health",
        description="Epidemiological case classification (Suspected, Probable, Confirmed, Discarded)",
        name_patterns=[r"\bcase[_-]?status\b", r"\bcase[_-]?class(ification)?\b", r"\bcase[_-]?def(inition)?\b"],
        expected_dtype="categorical",
    ),
    "malaria_rdt_result": SemanticTypeDefinition(
        type_id="malaria_rdt_result",
        category="public_health",
        description="Malaria Rapid Diagnostic Test result (Pf, Pan, Negative, Positive)",
        name_patterns=[r"\brdt([_-]?result)?\b", r"\bmalaria[_-]?test\b", r"\bmrdt\b", r"\bbs[_-]?for[_-]?mps\b"],
        expected_dtype="categorical",
    ),
    "hiv_viral_load": SemanticTypeDefinition(
        type_id="hiv_viral_load",
        category="public_health",
        description="HIV Viral Load count (copies/mL) or suppression status",
        name_patterns=[r"\bviral[_-]?load\b", r"\bvl\b", r"\bhiv[_-]?vl\b", r"\bcopies\b"],
        expected_dtype="any",
    ),
    "cd4_count": SemanticTypeDefinition(
        type_id="cd4_count",
        category="public_health",
        description="CD4 cell count (cells/uL)",
        name_patterns=[r"\bcd4([_-]?count)?\b", r"\bcd4[_-]?cells\b"],
        expected_dtype="numeric",
        value_range=(0, 3000),
    ),
    "anc_visit_number": SemanticTypeDefinition(
        type_id="anc_visit_number",
        category="public_health",
        description="Antenatal Care visit contact number (ANC 1, ANC 4+, ANC 8)",
        name_patterns=[r"\banc[_-]?visit\b", r"\banc[_-]?contact\b", r"\banc1\b", r"\banc4\b"],
        expected_dtype="any",
    ),
    "delivery_mode": SemanticTypeDefinition(
        type_id="delivery_mode",
        category="public_health",
        description="Obstetric delivery mode (SVD, C-section, Assisted breech)",
        name_patterns=[r"\bdelivery[_-]?mode\b", r"\bmode[_-]?of[_-]?delivery\b", r"\bc[_-]?section\b", r"\bsvd\b"],
        expected_dtype="categorical",
    ),
    "family_planning_method": SemanticTypeDefinition(
        type_id="family_planning_method",
        category="public_health",
        description="Contraceptive or family planning commodity (Implants, Injectable, OCP, IUCD, Condoms)",
        name_patterns=[r"\bfp[_-]?method\b", r"\bcontraceptive\b", r"\bfamily[_-]?planning\b"],
        expected_dtype="categorical",
    ),

    # ── Operations & Health Systems ──
    "service_volume_count": SemanticTypeDefinition(
        type_id="service_volume_count",
        category="operations",
        description="Count of services rendered or patients seen (OPD attendances, tests performed)",
        name_patterns=[r"\battendance(s)?\b", r"\bvolume\b", r"\bheadcount\b", r"\bvisits?\b", r"\bconsultations?\b", r"\btotal[_-]?clients\b"],
        expected_dtype="numeric",
    ),
    "bed_occupancy_rate": SemanticTypeDefinition(
        type_id="bed_occupancy_rate",
        category="operations",
        description="Hospital bed occupancy percentage (BOR)",
        name_patterns=[r"\bbed[_-]?occupancy([_-]?rate)?\b", r"\bbor\b", r"\boccupancy[_-]?rate\b"],
        expected_dtype="numeric",
        value_range=(0.0, 200.0),
    ),
    "stock_out_days": SemanticTypeDefinition(
        type_id="stock_out_days",
        category="operations",
        description="Number of stock-out days for essential medicines or commodities",
        name_patterns=[r"\bstock[_-]?out([_-]?days)?\b", r"\bdays[_-]?out[_-]?of[_-]?stock\b"],
        expected_dtype="numeric",
        value_range=(0, 366),
    ),
    "staff_category": SemanticTypeDefinition(
        type_id="staff_category",
        category="operations",
        description="Health workforce cadre or professional category",
        name_patterns=[r"\bcadre\b", r"\bstaff[_-]?category\b", r"\bjob[_-]?title\b", r"\bdesignation\b", r"\brole\b"],
        expected_dtype="categorical",
    ),
    "financial_currency": SemanticTypeDefinition(
        type_id="financial_currency",
        category="operations_finance",
        description="Monetary cost, charge, expenditure, revenue or billing amount",
        name_patterns=[r"\bcost\b", r"\bfee\b", r"\bprice\b", r"\brevenue\b", r"\bexpenditure\b", r"\bbill(ing)?\b", r"\bamount\b", r"\btariff\b"],
        expected_dtype="numeric",
    ),
    "insurance_status": SemanticTypeDefinition(
        type_id="insurance_status",
        category="operations_finance",
        description="Health insurance coverage or payer scheme (NHIF, Medicare, Out-of-pocket)",
        name_patterns=[r"\binsurance\b", r"\bpayer\b", r"\bnhif\b", r"\bcoverage[_-]?type\b", r"\bfinancial[_-]?scheme\b"],
        expected_dtype="categorical",
    ),
    "data_completeness_rate": SemanticTypeDefinition(
        type_id="data_completeness_rate",
        category="data_quality",
        description="Proportion of expected reports received or forms completed",
        name_patterns=[r"\bcompleteness\b", r"\btimeliness\b", r"\breporting[_-]?rate\b"],
        expected_dtype="numeric",
        value_range=(0.0, 100.0),
    ),

    # ── General Fallback Semantics ──
    "percentage": SemanticTypeDefinition(
        type_id="percentage",
        category="general_metrics",
        description="Proportion or rate expressed as a percentage (0 - 100)",
        name_patterns=[r"\bpct\b", r"\bpercent(age)?\b", r"\bratio\b", r"\%"],
        expected_dtype="numeric",
        value_range=(0.0, 100.0),
    ),
    "generic_count": SemanticTypeDefinition(
        type_id="generic_count",
        category="general_metrics",
        description="Discrete non-negative integer count",
        name_patterns=[r"\bcount\b", r"\btotal\b", r"\bnumber\b", r"\bnum\b", r"\bqty\b", r"\bquantity\b"],
        expected_dtype="numeric",
    ),
    "generic_category": SemanticTypeDefinition(
        type_id="generic_category",
        category="general",
        description="General discrete categorical dimension",
        name_patterns=[r"\bcategory\b", r"\btype\b", r"\bstatus\b", r"\bclass\b", r"\bgroup\b"],
        expected_dtype="categorical",
    ),
    "boolean_flag": SemanticTypeDefinition(
        type_id="boolean_flag",
        category="general",
        description="Binary true/false indicator flag",
        name_patterns=[r"\bis[_-]", r"\bhas[_-]", r"\bflag\b", r"\benabled\b", r"\bactive\b"],
        expected_dtype="boolean",
    ),
}


# =====================================================================
# 2. COLUMN INTELLIGENCE ANALYZER
# =====================================================================

class ColumnIntelligence:
    """
    Performs deep multi-signal inspection of a pandas Series.
    Extracts semantic types, units, temporal frequencies, granularity,
    encoding/hygiene bugs, and composite value split hints.
    """

    def __init__(self, series: pd.Series, col_name: str, total_dataset_rows: int = 0):
        self.raw_series = series
        self.col_name = str(col_name).strip()
        self.non_null_series = series.dropna()
        self.total_rows = total_dataset_rows or len(series)
        self.non_null_count = len(self.non_null_series)
        self.unique_count = self.non_null_series.nunique()
        self.pandas_dtype = str(series.dtype)

    # ─────────────────────────────────────────────────────────────
    # Unit Detection
    # ─────────────────────────────────────────────────────────────
    def detect_unit(self) -> Optional[dict[str, Any]]:
        """
        Detects unit of measurement from column name suffixes/parentheses
        and content value patterns.
        """
        name_lower = self.col_name.lower()

        # 1. Look for explicit units in column name brackets/parentheses, e.g. "Weight (kg)"
        bracket_match = re.search(r"[\(\[\{]([^\)\]\}]+)[\)\]\}]", self.col_name)
        if bracket_match:
            candidate = bracket_match.group(1).strip().lower()
            unit_map = {
                "%": {"unit": "percentage", "symbol": "%"},
                "pct": {"unit": "percentage", "symbol": "%"},
                "percent": {"unit": "percentage", "symbol": "%"},
                "usd": {"unit": "currency_usd", "symbol": "$"},
                "$": {"unit": "currency_usd", "symbol": "$"},
                "ksh": {"unit": "currency_kes", "symbol": "KSh"},
                "kes": {"unit": "currency_kes", "symbol": "KSh"},
                "eur": {"unit": "currency_eur", "symbol": "€"},
                "€": {"unit": "currency_eur", "symbol": "€"},
                "kg": {"unit": "weight_kg", "symbol": "kg"},
                "g": {"unit": "weight_g", "symbol": "g"},
                "mg": {"unit": "dosage_mg", "symbol": "mg"},
                "ml": {"unit": "volume_ml", "symbol": "ml"},
                "l": {"unit": "volume_l", "symbol": "L"},
                "c": {"unit": "celsius", "symbol": "°C"},
                "celsius": {"unit": "celsius", "symbol": "°C"},
                "f": {"unit": "fahrenheit", "symbol": "°F"},
                "days": {"unit": "duration_days", "symbol": "days"},
                "hours": {"unit": "duration_hours", "symbol": "hours"},
                "hrs": {"unit": "duration_hours", "symbol": "hrs"},
                "weeks": {"unit": "duration_weeks", "symbol": "wks"},
                "months": {"unit": "duration_months", "symbol": "mo"},
                "years": {"unit": "duration_years", "symbol": "yrs"},
                "bpm": {"unit": "heart_rate_bpm", "symbol": "bpm"},
                "mmhg": {"unit": "blood_pressure_mmhg", "symbol": "mmHg"},
                "copies/ml": {"unit": "copies_per_ml", "symbol": "copies/mL"},
                "cells/ul": {"unit": "cells_per_ul", "symbol": "cells/μL"},
            }
            if candidate in unit_map:
                res = unit_map[candidate].copy()
                res["source"] = "column_header_bracket"
                res["confidence"] = 0.98
                return res

        # 2. Look for name suffixes / keywords
        suffix_rules = [
            (r"(_usd|\$)", "currency_usd", "$", 0.95),
            (r"(_kes|_ksh)", "currency_kes", "KSh", 0.95),
            (r"(_eur|€)", "currency_eur", "€", 0.95),
            (r"(_pct|_percent|_percentage)\b", "percentage", "%", 0.95),
            (r"(_rate|_ratio)\b", "ratio", "ratio", 0.85),
            (r"(_kg|_kilograms?)\b", "weight_kg", "kg", 0.95),
            (r"(_mg|_milligrams?)\b", "dosage_mg", "mg", 0.95),
            (r"(_days?)\b", "duration_days", "days", 0.90),
            (r"(_hours?|_hrs?)\b", "duration_hours", "hours", 0.90),
            (r"(_weeks?)\b", "duration_weeks", "weeks", 0.90),
            (r"(_months?)\b", "duration_months", "months", 0.90),
            (r"(_years?|_yrs?)\b", "duration_years", "years", 0.90),
            (r"(_celsius|_deg_c)\b", "celsius", "°C", 0.95),
            (r"(_bpm)\b", "heart_rate_bpm", "bpm", 0.95),
            (r"(_mmhg)\b", "blood_pressure_mmhg", "mmHg", 0.95),
            (r"(_count|_total|_num)\b", "count", "count", 0.80),
        ]
        for pattern, unit, symbol, conf in suffix_rules:
            if re.search(pattern, name_lower):
                return {"unit": unit, "symbol": symbol, "source": "column_name_suffix", "confidence": conf}

        # 3. Value inspection (first 100 non-null strings)
        if self.non_null_count > 0:
            sample = self.non_null_series.head(100).astype(str)
            pct_count = sample.str.endswith("%").sum()
            if pct_count / len(sample) >= 0.8:
                return {"unit": "percentage", "symbol": "%", "source": "value_pattern", "confidence": 0.92}

            dollar_count = sample.str.startswith("$").sum()
            if dollar_count / len(sample) >= 0.8:
                return {"unit": "currency_usd", "symbol": "$", "source": "value_pattern", "confidence": 0.92}

            celsius_count = sample.str.contains(r"°C|degC", case=False).sum()
            if celsius_count / len(sample) >= 0.8:
                return {"unit": "celsius", "symbol": "°C", "source": "value_pattern", "confidence": 0.95}

        return None

    # ─────────────────────────────────────────────────────────────
    # Granularity Detection
    # ─────────────────────────────────────────────────────────────
    def detect_granularity(self) -> dict[str, Any]:
        """
        Infers whether column operates at individual row level,
        aggregated group level, or high-level summary.
        """
        if self.non_null_count == 0:
            return {"granularity": "empty", "confidence": 1.0, "uniqueness_ratio": 0.0}

        uniqueness = self.unique_count / max(self.total_rows, 1)

        # High cardinality ID or timestamp -> individual record level
        if uniqueness > 0.90:
            is_key = True
            # Continuous float measurements (e.g. rates, temps, totals) are measures, not identifiers
            if pd.api.types.is_float_dtype(self.raw_series):
                is_key = False
            return {
                "granularity": "individual_record",
                "confidence": 0.95,
                "uniqueness_ratio": round(uniqueness, 4),
                "is_candidate_key": is_key,
            }

        # Medium cardinality -> cohort / sub-group / facility level
        if 0.05 <= uniqueness <= 0.90:
            return {
                "granularity": "subgroup_level",
                "confidence": 0.80,
                "uniqueness_ratio": round(uniqueness, 4),
                "is_candidate_key": False,
            }

        # Low cardinality (< 5% unique) -> categorical dimension
        return {
            "granularity": "categorical_dimension",
            "confidence": 0.90,
            "uniqueness_ratio": round(uniqueness, 4),
            "is_candidate_key": False,
        }

    # ─────────────────────────────────────────────────────────────
    # Temporal Pattern Detection
    # ─────────────────────────────────────────────────────────────
    def detect_temporal_pattern(self) -> Optional[dict[str, Any]]:
        """
        Tests if series is datetime or parseable as datetime and detects
        its frequency pattern (daily, weekly, monthly, quarterly, irregular).
        """
        is_dt = pd.api.types.is_datetime64_any_dtype(self.raw_series)
        parsed_dt = None

        if is_dt:
            parsed_dt = pd.to_datetime(self.non_null_series, errors="coerce").dropna()
        else:
            name_lower = self.col_name.lower()
            date_keywords = re.compile(r"date|month|week|year|period|time|admission|discharge|dob", re.I)
            if date_keywords.search(name_lower):
                try:
                    # Sample first 50 values to avoid slow parsing on massive tables
                    sample = self.non_null_series.head(50)
                    test_dt = pd.to_datetime(sample, errors="coerce")
                    if test_dt.notna().mean() >= 0.8:
                        parsed_dt = pd.to_datetime(self.non_null_series.head(500), errors="coerce").dropna()
                except Exception:
                    pass

        if parsed_dt is None or len(parsed_dt) < 2:
            return None

        # Sort and calculate intervals
        sorted_dt = parsed_dt.sort_values().drop_duplicates()
        if len(sorted_dt) < 2:
            return {
                "is_temporal": True,
                "pattern": "single_point",
                "min_date": str(sorted_dt.min()),
                "max_date": str(sorted_dt.max()),
                "confidence": 0.70,
            }

        diffs = sorted_dt.diff().dropna()
        median_days = diffs.dt.total_seconds().median() / 86400.0

        pattern = "irregular"
        freq_label = "irregular"
        conf = 0.75

        if 0.8 <= median_days <= 1.2:
            pattern = "regular_daily"
            freq_label = "D"
            conf = 0.95
        elif 6.5 <= median_days <= 7.5:
            pattern = "regular_weekly"
            freq_label = "W"
            conf = 0.92
        elif 27.0 <= median_days <= 32.0:
            pattern = "regular_monthly"
            freq_label = "M"
            conf = 0.95
        elif 85.0 <= median_days <= 95.0:
            pattern = "regular_quarterly"
            freq_label = "Q"
            conf = 0.90
        elif 350.0 <= median_days <= 370.0:
            pattern = "regular_yearly"
            freq_label = "Y"
            conf = 0.95

        return {
            "is_temporal": True,
            "pattern": pattern,
            "frequency_code": freq_label,
            "median_interval_days": round(median_days, 2),
            "min_date": str(sorted_dt.min()),
            "max_date": str(sorted_dt.max()),
            "span_days": round((sorted_dt.max() - sorted_dt.min()).total_seconds() / 86400.0, 1),
            "confidence": conf,
        }

    # ─────────────────────────────────────────────────────────────
    # Data Hygiene, Encoding & Masquerade Detection
    # ─────────────────────────────────────────────────────────────
    def detect_hygiene_issues(self) -> list[dict[str, Any]]:
        """
        Detects data cleanliness issues, hidden encodings, and masquerading types:
        - Leading/trailing whitespace
        - Boolean masquerading as strings ("Y"/"N", "Yes"/"No", "1"/"0")
        - Numeric disguised as string (e.g. "1,000", "$45.20", "20%")
        - Inconsistent casing in categorical values ("Female", "female", "FEMALE")
        - Sentinel missing values disguised as data (e.g. 999, -99, "NA", "NULL", "?")
        - Corrupted UTF-8 / Mojibake characters (e.g. "Ã©", "â€", "\ufffd")
        """
        issues: list[dict[str, Any]] = []
        if self.non_null_count == 0:
            return issues

        sample = self.non_null_series.head(500)

        # 1. Whitespace padding
        if sample.dtype == object or pd.api.types.is_string_dtype(sample):
            str_vals = sample.astype(str)
            whitespace_padded = str_vals.apply(lambda s: len(s) != len(s.strip())).sum()
            if whitespace_padded > 0:
                issues.append({
                    "issue_type": "whitespace_padding",
                    "severity": "medium",
                    "affected_sample_count": int(whitespace_padded),
                    "affected_sample_pct": round(whitespace_padded / len(sample), 3),
                    "description": "Values contain leading or trailing whitespace.",
                    "remediation": "Apply .str.strip()",
                })

            # 2. Casing inconsistency
            uniques = str_vals.unique()
            lower_uniques = set(s.lower() for s in uniques)
            if len(lower_uniques) < len(uniques):
                issues.append({
                    "issue_type": "inconsistent_casing",
                    "severity": "medium",
                    "distinct_case_variants": len(uniques) - len(lower_uniques),
                    "description": "Category values differ only by character casing (e.g. 'Male' vs 'male').",
                    "remediation": "Apply .str.title() or .str.lower() normalization",
                })

            # 3. Boolean masquerade
            bool_tokens = {"y", "n", "yes", "no", "true", "false", "t", "f", "1", "0"}
            sample_tokens = set(str_vals.str.strip().str.lower().unique())
            if sample_tokens and sample_tokens.issubset(bool_tokens) and len(sample_tokens) <= 3:
                issues.append({
                    "issue_type": "boolean_masquerade",
                    "severity": "low",
                    "detected_tokens": list(sample_tokens),
                    "description": "Boolean binary flag is stored as text.",
                    "remediation": "Map to boolean True/False",
                })

            # 4. Numeric disguised as string
            cleaned_num = str_vals.str.replace(r"[,\$%\s]", "", regex=True)
            numeric_convertible = pd.to_numeric(cleaned_num, errors="coerce").notna().sum()
            if numeric_convertible / len(sample) >= 0.85 and not pd.api.types.is_numeric_dtype(self.raw_series):
                issues.append({
                    "issue_type": "numeric_disguised_as_string",
                    "severity": "high",
                    "convertible_pct": round(numeric_convertible / len(sample), 3),
                    "description": "Numeric column formatted with currency symbols, commas, or percent signs.",
                    "remediation": "Strip formatting characters and cast to float/int",
                })

            # 5. Mojibake / Corrupted UTF-8
            mojibake_pattern = re.compile(r"Ã.|â€|\ufffd|Â")
            mojibake_count = str_vals.str.contains(mojibake_pattern).sum()
            if mojibake_count > 0:
                issues.append({
                    "issue_type": "mojibake_encoding_error",
                    "severity": "high",
                    "affected_sample_count": int(mojibake_count),
                    "description": "Contains corrupted multi-byte character sequences (mojibake).",
                    "remediation": "Re-encode using proper character set (e.g. ftfy or latin1/utf-8 roundtrip)",
                })

        # 6. Sentinel missing values disguised as numeric
        if pd.api.types.is_numeric_dtype(self.raw_series):
            sentinels = [999, 9999, -99, -999, -1]
            for s in sentinels:
                s_count = (self.non_null_series == s).sum()
                if s_count > 0 and (s_count / len(self.non_null_series)) > 0.01:
                    issues.append({
                        "issue_type": "sentinel_missing_value",
                        "severity": "high",
                        "sentinel_value": s,
                        "occurrences": int(s_count),
                        "description": f"Value {s} appears to be a sentinel missing data code, not a true measurement.",
                        "remediation": f"Replace {s} with np.nan",
                    })

        return issues

    # ─────────────────────────────────────────────────────────────
    # Composite Value Detection
    # ─────────────────────────────────────────────────────────────
    def detect_composite_values(self) -> Optional[dict[str, Any]]:
        """
        Detects if column holds composite or concatenated values,
        such as 'Nairobi - West', 'Ward 4 (Kibera)', '120/80', 'M / 42'.
        """
        if not (self.raw_series.dtype == object or pd.api.types.is_string_dtype(self.raw_series)):
            return None

        sample = self.non_null_series.head(200).astype(str).str.strip()
        if len(sample) < 10:
            return None

        # Pattern 1: Blood pressure "120/80"
        bp_match = sample.str.match(r"^\d{2,3}/\d{2,3}$").sum()
        if bp_match / len(sample) >= 0.8:
            return {
                "composite_type": "blood_pressure_pair",
                "delimiter": "/",
                "suggested_splits": [f"{self.col_name}_systolic", f"{self.col_name}_diastolic"],
                "target_dtype": "numeric",
                "confidence": 0.95,
            }

        # Pattern 2: Delimited hierarchy e.g. "Region - District" or "Province / District"
        for delim in [" - ", " / ", " > ", " : "]:
            delim_count = sample.str.contains(re.escape(delim)).sum()
            if delim_count / len(sample) >= 0.85:
                # Check parts count
                parts = sample.str.split(delim, expand=True)
                if parts.shape[1] == 2:
                    return {
                        "composite_type": "delimited_hierarchy",
                        "delimiter": delim,
                        "suggested_splits": [f"{self.col_name}_parent", f"{self.col_name}_child"],
                        "confidence": 0.88,
                    }

        # Pattern 3: Name with ID in parentheses: "St Mary's Hospital (HOSP-012)"
        code_in_bracket = sample.str.contains(r"\([A-Za-z0-9\-_]{2,}\)$").sum()
        if code_in_bracket / len(sample) >= 0.75:
            return {
                "composite_type": "entity_and_code",
                "delimiter": "parentheses",
                "suggested_splits": [f"{self.col_name}_name", f"{self.col_name}_code"],
                "confidence": 0.85,
            }

        return None

    # ─────────────────────────────────────────────────────────────
    # Semantic Type Inference
    # ─────────────────────────────────────────────────────────────
    def _match_pattern(self, pattern: str) -> bool:
        col = self.col_name
        if re.search(pattern, col, re.IGNORECASE):
            return True
        # Allow spaces or hyphens where underscores/hyphens are present
        flex_pat = pattern.replace(r"[_-]?", r"[\s_-]?").replace(r"[_-]", r"[\s_-]")
        if re.search(flex_pat, col, re.IGNORECASE):
            return True
        # Underscore normalized form
        norm_col = re.sub(r"[\s\-]+", "_", col.lower().strip())
        if re.search(pattern, norm_col, re.IGNORECASE):
            return True
        # Space-separated token form for clean \b word-boundary matching
        col_spaced = f" {re.sub(r'[_\-]+', ' ', col.lower().strip())} "
        pat_spaced = pattern.replace(r"[_-]?", r"\s?").replace(r"[_-]", r"\s")
        if re.search(pat_spaced, col_spaced, re.IGNORECASE):
            return True
        return False

    def infer_semantic_type(self, context_cols: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Scores the column against the 60+ health and analytics semantic taxonomy.
        Combines name regex, data types, value patterns, and physiological limits.
        """
        best_match_id = "generic_text"
        best_score = 0.0
        best_rationale = "Default fallback"

        # Check preliminary basic classification
        is_num = pd.api.types.is_numeric_dtype(self.raw_series)
        is_bool = pd.api.types.is_bool_dtype(self.raw_series)
        is_dt = pd.api.types.is_datetime64_any_dtype(self.raw_series)

        if is_bool:
            best_match_id = "boolean_flag"
            best_score = 0.95
            best_rationale = "Boolean dtype detected"
        elif is_num:
            best_match_id = "generic_numeric"
            best_score = 0.30
            best_rationale = "Numeric dtype detected"
        elif self.unique_count / max(self.total_rows, 1) < 0.08:
            best_match_id = "generic_category"
            best_score = 0.35
            best_rationale = "Low cardinality discrete values"

        # Check each taxonomy candidate
        for type_id, defn in SEMANTIC_TAXONOMY.items():
            score = 0.0
            rationale_parts = []

            # 1. Match column name regex
            name_matched = False
            for pat in defn.name_patterns:
                if self._match_pattern(pat):
                    name_matched = True
                    score += 0.55
                    rationale_parts.append(f"Header matched pattern '{pat}'")
                    break

            # 2. Check expected dtype compatibility
            if defn.expected_dtype == "numeric":
                if is_num:
                    score += 0.20
                else:
                    score -= 0.30
            elif defn.expected_dtype == "datetime":
                if is_dt:
                    score += 0.35
                else:
                    # check if string parses as datetime
                    temporal_info = self.detect_temporal_pattern()
                    if temporal_info and temporal_info.get("is_temporal"):
                        score += 0.30
            elif defn.expected_dtype == "boolean":
                if is_bool:
                    score += 0.30
            elif defn.expected_dtype == "categorical":
                if (not is_num) and (self.unique_count / max(self.total_rows, 1) < 0.20):
                    score += 0.20

            # 3. Value regex check if defined
            if defn.value_regex and self.non_null_count > 0:
                sample = self.non_null_series.head(50).astype(str)
                reg_matches = sample.str.match(defn.value_regex).sum()
                if reg_matches / len(sample) >= 0.70:
                    score += 0.35
                    rationale_parts.append(f"Value content matched regex {defn.value_regex}")

            # 4. Value range check if numeric and defined
            if defn.value_range and is_num and self.non_null_count > 0:
                min_val = float(self.non_null_series.min())
                max_val = float(self.non_null_series.max())
                low_b, high_b = defn.value_range
                if min_val >= low_b and max_val <= high_b:
                    score += 0.20
                    rationale_parts.append(f"Values within physiological bounds [{low_b}, {high_b}]")
                elif min_val < low_b or max_val > high_b * 1.5:
                    score -= 0.35  # Outside expected range

            # Apply bonus if column name has strong domain keyword
            if name_matched and score > best_score:
                best_score = score
                best_match_id = type_id
                best_rationale = "; ".join(rationale_parts)

        # Normalize confidence to [0.1, 0.99]
        confidence = min(0.99, max(0.15, round(best_score, 2)))
        defn = SEMANTIC_TAXONOMY.get(best_match_id)

        return {
            "semantic_type": best_match_id,
            "category": defn.category if defn else "general",
            "description": defn.description if defn else "",
            "confidence": confidence,
            "rationale": best_rationale,
        }

    # ─────────────────────────────────────────────────────────────
    # Full Column Profile Synthesis
    # ─────────────────────────────────────────────────────────────
    def analyze(self, context_cols: Optional[list[str]] = None) -> dict[str, Any]:
        """Runs full suite of intelligence detectors on the column."""
        semantic_info = self.infer_semantic_type(context_cols)
        unit_info = self.detect_unit()
        granularity_info = self.detect_granularity()
        temporal_info = self.detect_temporal_pattern()
        hygiene_issues = self.detect_hygiene_issues()
        composite_info = self.detect_composite_values()

        return {
            "column_name": self.col_name,
            "pandas_dtype": self.pandas_dtype,
            "total_rows": self.total_rows,
            "null_count": int(self.raw_series.isna().sum()),
            "null_pct": round(float(self.raw_series.isna().mean()), 4),
            "unique_count": int(self.unique_count),
            "semantic": semantic_info,
            "unit": unit_info,
            "granularity": granularity_info,
            "temporal": temporal_info,
            "hygiene_issues": hygiene_issues,
            "composite": composite_info,
        }


# =====================================================================
# 3. CROSS-COLUMN RELATIONSHIP & SCHEMA ANALYZER
# =====================================================================

class SchemaIntelligence:
    """
    Dataset-wide intelligence engine.
    Analyzes all columns and detects cross-column relationships:
    - Hierarchical parent-child relationships (e.g. Region -> District -> Facility)
    - Functional dependencies (e.g. Facility Code -> Facility Name)
    - Interval date pairs (e.g. Admission Date -> Discharge Date)
    - Unit/Dimension pairs
    - Generates standardized data dictionary
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rows = len(df)
        self.columns = list(df.columns)
        self.col_analyzers: dict[str, ColumnIntelligence] = {
            col: ColumnIntelligence(df[col], col, self.rows) for col in self.columns
        }
        self.col_profiles: dict[str, dict[str, Any]] = {}

    def analyze_columns(self) -> dict[str, dict[str, Any]]:
        """Analyzes all columns individually."""
        profiles = {}
        for col, analyzer in self.col_analyzers.items():
            profiles[col] = analyzer.analyze(self.columns)
        self.col_profiles = profiles
        return profiles

    def detect_hierarchies(self) -> list[dict[str, Any]]:
        """
        Detects administrative or geographical hierarchies where
        Parent Column -> Child Column (e.g. Region -> District -> Facility).
        Condition: Every value in Child belongs to exactly ONE value in Parent.
        """
        hierarchies = []
        if self.rows < 10:
            return hierarchies

        # Look for categorical or geographical columns
        geo_types = {"country", "province_state", "district_county", "sub_county_ward", "facility_name", "generic_category"}
        candidates = [
            col for col, prof in self.col_profiles.items()
            if prof["semantic"]["semantic_type"] in geo_types or prof["granularity"]["granularity"] == "categorical_dimension"
        ]

        # Test candidate pairs (Parent, Child) where unique(Parent) < unique(Child)
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                parent = candidates[i]
                child = candidates[j]

                n_parent = self.col_profiles[parent]["unique_count"]
                n_child = self.col_profiles[child]["unique_count"]

                # Parent must have strictly fewer unique values than child
                if 1 < n_parent < n_child and n_child <= (self.rows * 0.95):
                    # Check if each child maps to a single parent
                    grouped = self.df[[parent, child]].dropna().groupby(child)[parent].nunique()
                    if len(grouped) > 0 and (grouped == 1).mean() >= 0.95:
                        hierarchies.append({
                            "parent": parent,
                            "child": child,
                            "parent_unique": n_parent,
                            "child_unique": n_child,
                            "consistency_rate": round(float((grouped == 1).mean()), 3),
                            "relationship_type": "hierarchy_parent_child",
                        })

        return hierarchies

    def detect_functional_dependencies(self) -> list[dict[str, Any]]:
        """
        Detects Functional Dependencies where Column A -> Column B (A uniquely determines B).
        E.g. Facility Code -> Facility Name, Patient ID -> Date of Birth.
        """
        fds = []
        if self.rows < 5:
            return fds

        # Candidate determinants: high cardinality or ID types
        for col_a in self.columns:
            prof_a = self.col_profiles[col_a]
            if prof_a["unique_count"] < 2:
                continue

            for col_b in self.columns:
                if col_a == col_b:
                    continue
                prof_b = self.col_profiles[col_b]

                # Determinant must have >= unique count of dependent
                if prof_a["unique_count"] >= prof_b["unique_count"]:
                    # Group by A and check if B has exactly 1 value per A
                    grouped = self.df[[col_a, col_b]].dropna().groupby(col_a)[col_b].nunique()
                    if len(grouped) > 0 and (grouped == 1).mean() >= 0.98:
                        fds.append({
                            "determinant": col_a,
                            "dependent": col_b,
                            "determinant_type": prof_a["semantic"]["semantic_type"],
                            "dependent_type": prof_b["semantic"]["semantic_type"],
                            "fidelity": round(float((grouped == 1).mean()), 4),
                        })

        return fds

    def detect_interval_pairs(self) -> list[dict[str, Any]]:
        """
        Detects date or time pairs representing intervals,
        e.g. (Admission Date, Discharge Date), (Start Date, End Date).
        """
        intervals = []
        dt_cols = [
            col for col, prof in self.col_profiles.items()
            if prof["temporal"] is not None and prof["temporal"]["is_temporal"]
        ]

        if len(dt_cols) < 2:
            return intervals

        # Pairs of dates
        for i in range(len(dt_cols)):
            for j in range(i + 1, len(dt_cols)):
                col1 = dt_cols[i]
                col2 = dt_cols[j]

                # Convert to datetime for comparison
                dt1 = pd.to_datetime(self.df[col1], errors="coerce")
                dt2 = pd.to_datetime(self.df[col2], errors="coerce")

                valid_mask = dt1.notna() & dt2.notna()
                if valid_mask.sum() >= 5:
                    diff = (dt2[valid_mask] - dt1[valid_mask]).dt.total_seconds()
                    # Check if col2 is consistently >= col1
                    if (diff >= 0).mean() >= 0.90:
                        median_duration_days = round(float(diff.median() / 86400.0), 2)
                        intervals.append({
                            "start_column": col1,
                            "end_column": col2,
                            "consistency_rate": round(float((diff >= 0).mean()), 3),
                            "median_duration_days": median_duration_days,
                            "suggested_duration_feature": f"length_of_stay_days" if "admi" in col1.lower() else f"{col1}_{col2}_duration_days",
                        })
                    elif (diff <= 0).mean() >= 0.90:
                        median_duration_days = round(float((-diff).median() / 86400.0), 2)
                        intervals.append({
                            "start_column": col2,
                            "end_column": col1,
                            "consistency_rate": round(float((diff <= 0).mean()), 3),
                            "median_duration_days": median_duration_days,
                            "suggested_duration_feature": f"{col2}_{col1}_duration_days",
                        })

        return intervals

    def generate_data_dictionary(self) -> dict[str, Any]:
        """
        Produces a rich, structured Data Dictionary for the entire dataset,
        suitable for LLM grounding, UI inspection, and automated documentation.
        """
        if not self.col_profiles:
            self.analyze_columns()

        hierarchies = self.detect_hierarchies()
        fds = self.detect_functional_dependencies()
        intervals = self.detect_interval_pairs()

        # Classify overall dataset domain
        detected_categories = [prof["semantic"]["category"] for prof in self.col_profiles.values()]
        primary_domain = "general_tabular"
        health_cats = sum(
            1 for c in detected_categories
            if any(k in c for k in ["clinical", "demographics", "public_health", "operations", "data_quality"])
        )
        if health_cats >= 2:
            primary_domain = "health_medical_data"

        dictionary_entries = []
        for col, prof in self.col_profiles.items():
            sem = prof["semantic"]
            unit = prof["unit"]
            gran = prof["granularity"]
            temp = prof["temporal"]

            # Sample values (up to 3 distinct representation examples)
            sample_vals = self.df[col].dropna().head(3).tolist()
            sample_str = [str(x) for x in sample_vals]

            entry = {
                "name": col,
                "inferred_semantic_type": sem["semantic_type"],
                "category": sem["category"],
                "description": sem["description"],
                "confidence": sem["confidence"],
                "unit": unit["symbol"] if unit else None,
                "unit_details": unit,
                "granularity": gran["granularity"],
                "temporal_pattern": temp["pattern"] if temp else None,
                "null_pct": prof["null_pct"],
                "unique_values": prof["unique_count"],
                "sample_values": sample_str,
                "hygiene_issues_count": len(prof["hygiene_issues"]),
                "hygiene_warnings": [h["description"] for h in prof["hygiene_issues"]],
            }
            dictionary_entries.append(entry)

        return {
            "dataset_summary": {
                "total_rows": self.rows,
                "total_columns": len(self.columns),
                "inferred_domain": primary_domain,
                "detected_hierarchies_count": len(hierarchies),
                "detected_functional_deps_count": len(fds),
                "detected_interval_pairs_count": len(intervals),
            },
            "columns": dictionary_entries,
            "relationships": {
                "hierarchies": hierarchies,
                "functional_dependencies": fds,
                "intervals": intervals,
            },
        }


# =====================================================================
# 4. TOP-LEVEL CONVENIENCE API
# =====================================================================

def analyze_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Main entry point for deep schema intelligence.
    Returns:
    - column-by-column intelligence
    - cross-column relationships (hierarchies, FDs, intervals)
    - comprehensive data dictionary
    """
    engine = SchemaIntelligence(df)
    col_profiles = engine.analyze_columns()
    data_dict = engine.generate_data_dictionary()

    return {
        "column_intelligence": col_profiles,
        "data_dictionary": data_dict,
        "relationships": data_dict["relationships"],
        "summary": data_dict["dataset_summary"],
    }
