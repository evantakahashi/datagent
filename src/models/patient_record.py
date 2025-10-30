"""Patient Record Data Models"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime, date
from enum import Enum


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Ethnicity(str, Enum):
    CAUCASIAN = "caucasian"
    AFRICAN_AMERICAN = "african_american"
    HISPANIC = "hispanic"
    ASIAN = "asian"
    NATIVE_AMERICAN = "native_american"
    OTHER = "other"


class Demographics(BaseModel):
    """Patient demographic information"""
    age: int = Field(ge=0, le=120, description="Patient age in years")
    sex: Sex
    ethnicity: Ethnicity
    location: str = Field(description="Geographic location")

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError("Age must be between 0 and 120")
        return v


class VitalSigns(BaseModel):
    """Patient vital signs"""
    heart_rate: float = Field(ge=30, le=250, description="Heart rate in bpm")
    systolic_bp: float = Field(ge=60, le=250, description="Systolic BP in mmHg")
    diastolic_bp: float = Field(ge=40, le=150, description="Diastolic BP in mmHg")
    temperature: float = Field(ge=95.0, le=106.0, description="Temperature in F")
    respiratory_rate: float = Field(ge=8, le=40, description="Breaths per minute")
    oxygen_saturation: float = Field(ge=70, le=100, description="O2 sat %")

    @field_validator('systolic_bp', 'diastolic_bp')
    @classmethod
    def validate_bp(cls, v):
        return round(v, 1)


class LabResult(BaseModel):
    """Laboratory test result"""
    test_name: str
    value: float
    unit: str
    reference_range: str
    abnormal: bool = False


class Condition(BaseModel):
    """Medical condition/diagnosis"""
    name: str
    icd10_code: str
    onset_date: date
    severity: Literal["mild", "moderate", "severe"]
    status: Literal["active", "resolved", "chronic"]


class Medication(BaseModel):
    """Medication prescription"""
    name: str
    dosage: str
    frequency: str
    indication: str
    start_date: date


class PatientRecord(BaseModel):
    """Complete patient record"""
    record_id: str
    demographics: Demographics
    chief_complaint: str
    vitals: VitalSigns
    conditions: List[Condition]
    medications: List[Medication]
    lab_results: List[LabResult]
    clinical_notes: str
    visit_date: datetime

    def to_text(self) -> str:
        """Convert record to clinical text format for embedding"""
        text = f"""
Patient Record ID: {self.record_id}

DEMOGRAPHICS:
Age: {self.demographics.age} years
Sex: {self.demographics.sex.value}
Ethnicity: {self.demographics.ethnicity.value}
Location: {self.demographics.location}

CHIEF COMPLAINT:
{self.chief_complaint}

VITAL SIGNS:
Heart Rate: {self.vitals.heart_rate} bpm
Blood Pressure: {self.vitals.systolic_bp}/{self.vitals.diastolic_bp} mmHg
Temperature: {self.vitals.temperature}°F
Respiratory Rate: {self.vitals.respiratory_rate} breaths/min
Oxygen Saturation: {self.vitals.oxygen_saturation}%

CONDITIONS:
"""
        for cond in self.conditions:
            text += f"- {cond.name} ({cond.icd10_code}): {cond.severity}, {cond.status}\n"

        text += "\nMEDICATIONS:\n"
        for med in self.medications:
            text += f"- {med.name} {med.dosage} {med.frequency} for {med.indication}\n"

        text += "\nLABORATORY RESULTS:\n"
        for lab in self.lab_results:
            abnormal_flag = "ABNORMAL" if lab.abnormal else "NORMAL"
            text += f"- {lab.test_name}: {lab.value} {lab.unit} ({lab.reference_range}) [{abnormal_flag}]\n"

        text += f"\nCLINICAL NOTES:\n{self.clinical_notes}\n"

        return text.strip()
