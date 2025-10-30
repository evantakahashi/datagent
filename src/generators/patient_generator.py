"""Synthetic Patient Record Generator"""
import random
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import json

from src.models import PatientRecord, Demographics, VitalSigns, LabResult, Condition, Medication, Sex, Ethnicity
from configs.medical_knowledge import (
    MEDICAL_CONDITIONS, LAB_REFERENCE_RANGES, MEDICATIONS,
    VITAL_SIGN_RANGES, DEMOGRAPHIC_DISTRIBUTIONS
)


class PatientRecordGenerator:
    """Generate synthetic patient records with medical constraints"""

    def __init__(self, llm_client=None, seed: int = None):
        """
        Args:
            llm_client: Optional LLM client for enhanced generation
            seed: Random seed for reproducibility
        """
        self.llm_client = llm_client
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_demographics(self, constraints: Dict[str, Any] = None) -> Demographics:
        """Generate realistic demographics"""
        constraints = constraints or {}

        # Age selection with stratification
        if 'age' in constraints:
            age = constraints['age']
        else:
            age_groups = DEMOGRAPHIC_DISTRIBUTIONS['age_groups']
            selected_group = random.choices(
                age_groups,
                weights=[g['weight'] for g in age_groups]
            )[0]
            age = random.randint(*selected_group['range'])

        # Sex selection
        if 'sex' in constraints:
            sex = Sex(constraints['sex'])
        else:
            sex_dist = DEMOGRAPHIC_DISTRIBUTIONS['sex']
            sex_value = random.choices(
                [s['value'] for s in sex_dist],
                weights=[s['weight'] for s in sex_dist]
            )[0]
            sex = Sex(sex_value)

        # Ethnicity selection
        eth_dist = DEMOGRAPHIC_DISTRIBUTIONS['ethnicity']
        ethnicity_value = random.choices(
            [e['value'] for e in eth_dist],
            weights=[e['weight'] for e in eth_dist]
        )[0]
        ethnicity = Ethnicity(ethnicity_value)

        # Location
        location = random.choice(DEMOGRAPHIC_DISTRIBUTIONS['locations'])

        return Demographics(
            age=age,
            sex=sex,
            ethnicity=ethnicity,
            location=location
        )

    def generate_vitals(self, conditions: List[str], demographics: Demographics) -> VitalSigns:
        """Generate vital signs based on conditions"""

        # Start with normal ranges, adjust based on conditions
        vitals = {}

        for vital_name, ranges in VITAL_SIGN_RANGES.items():
            normal_min, normal_max = ranges['normal']
            possible_min, possible_max = ranges['possible']

            # Default: slightly vary within normal range
            base_value = random.uniform(normal_min, normal_max)

            # Adjust for conditions
            if 'hypertension' in conditions and vital_name in ['systolic_bp', 'diastolic_bp']:
                # Elevated BP
                base_value = random.uniform(normal_max, min(possible_max, normal_max * 1.4))

            elif 'copd' in conditions or 'asthma' in conditions:
                if vital_name == 'respiratory_rate':
                    base_value = random.uniform(normal_max, min(possible_max, normal_max * 1.3))
                elif vital_name == 'oxygen_saturation':
                    base_value = random.uniform(max(possible_min, 88), normal_min)

            elif 'cad' in conditions and vital_name == 'heart_rate':
                # Possible tachycardia or on beta blockers (bradycardia)
                if random.random() < 0.5:
                    base_value = random.uniform(normal_max, min(possible_max, 110))
                else:
                    base_value = random.uniform(max(possible_min, 55), normal_min)

            vitals[vital_name] = round(base_value, 1)

        return VitalSigns(
            heart_rate=vitals['heart_rate'],
            systolic_bp=vitals['systolic_bp'],
            diastolic_bp=vitals['diastolic_bp'],
            temperature=vitals['temperature'],
            respiratory_rate=vitals['respiratory_rate'],
            oxygen_saturation=vitals['oxygen_saturation']
        )

    def generate_lab_results(self, conditions: List[str]) -> List[LabResult]:
        """Generate lab results based on conditions"""
        labs = []
        required_labs = set()

        # Collect required labs from conditions
        for cond_key in conditions:
            if cond_key in MEDICAL_CONDITIONS:
                required_labs.update(MEDICAL_CONDITIONS[cond_key]['required_labs'])

        # Generate each required lab
        for lab_name in required_labs:
            if lab_name not in LAB_REFERENCE_RANGES:
                continue

            lab_info = LAB_REFERENCE_RANGES[lab_name]
            normal_min, normal_max = lab_info['normal_range']
            possible_min, possible_max = lab_info['possible_range']

            # Decide if abnormal based on condition
            is_abnormal = random.random() < 0.6  # 60% chance of abnormal

            if is_abnormal:
                # Generate abnormal value
                if random.random() < 0.5:
                    # High
                    value = random.uniform(normal_max, min(possible_max, normal_max * 1.5))
                else:
                    # Low
                    value = random.uniform(max(possible_min, normal_min * 0.7), normal_min)
            else:
                # Normal value
                value = random.uniform(normal_min, normal_max)

            labs.append(LabResult(
                test_name=lab_name.upper(),
                value=round(value, 2),
                unit=lab_info['unit'],
                reference_range=lab_info['reference_text'],
                abnormal=is_abnormal
            ))

        return labs

    def generate_conditions(self, demographics: Demographics, num_conditions: int = None) -> List[Condition]:
        """Generate medical conditions appropriate for demographics"""
        if num_conditions is None:
            num_conditions = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]

        conditions = []
        selected_keys = []

        available_conditions = list(MEDICAL_CONDITIONS.keys())
        random.shuffle(available_conditions)

        for cond_key in available_conditions:
            if len(conditions) >= num_conditions:
                break

            cond_info = MEDICAL_CONDITIONS[cond_key]
            age_min, age_max = cond_info['age_range']

            # Check age appropriateness
            if not (age_min <= demographics.age <= age_max):
                continue

            # Add condition
            onset_years_ago = random.randint(0, min(5, demographics.age - age_min))
            onset_date = date.today() - timedelta(days=onset_years_ago * 365)

            severity = random.choice(['mild', 'moderate', 'severe'])
            status = random.choice(['active', 'chronic'])

            conditions.append(Condition(
                name=cond_info['name'],
                icd10_code=cond_info['icd10'],
                onset_date=onset_date,
                severity=severity,
                status=status
            ))
            selected_keys.append(cond_key)

        return conditions, selected_keys

    def generate_medications(self, condition_keys: List[str]) -> List[Medication]:
        """Generate medications based on conditions"""
        medications = []

        for cond_key in condition_keys:
            if cond_key not in MEDICAL_CONDITIONS:
                continue

            cond_info = MEDICAL_CONDITIONS[cond_key]

            # Select 1-2 medications for this condition
            num_meds = random.randint(0, min(2, len(cond_info['common_medications'])))
            selected_meds = random.sample(cond_info['common_medications'], num_meds)

            for med_name in selected_meds:
                if med_name not in MEDICATIONS:
                    continue

                med_info = MEDICATIONS[med_name]
                start_date = date.today() - timedelta(days=random.randint(30, 1000))

                medications.append(Medication(
                    name=med_name.capitalize(),
                    dosage=random.choice(med_info['dosages']),
                    frequency=random.choice(med_info['frequencies']),
                    indication=med_info['indication'],
                    start_date=start_date
                ))

        return medications

    def generate_clinical_notes(self, demographics: Demographics, conditions: List[Condition],
                                chief_complaint: str) -> str:
        """Generate clinical notes"""
        notes = f"Patient is a {demographics.age}-year-old {demographics.sex.value} who presents with {chief_complaint}. "

        if conditions:
            cond_names = [c.name for c in conditions]
            notes += f"Past medical history significant for {', '.join(cond_names)}. "

        notes += "Physical examination and laboratory findings as documented above. "
        notes += "Plan discussed with patient. Follow-up scheduled."

        return notes

    def generate_record(self, constraints: Dict[str, Any] = None) -> PatientRecord:
        """Generate a complete patient record"""
        constraints = constraints or {}

        # Generate demographics
        demographics = self.generate_demographics(constraints)

        # Generate conditions
        conditions, condition_keys = self.generate_conditions(demographics)

        # Generate chief complaint based on first condition
        if conditions:
            first_cond_key = condition_keys[0]
            symptoms = MEDICAL_CONDITIONS[first_cond_key]['symptoms']
            chief_complaint = random.choice(symptoms)
        else:
            chief_complaint = "routine checkup"

        # Generate vitals
        vitals = self.generate_vitals(condition_keys, demographics)

        # Generate labs
        lab_results = self.generate_lab_results(condition_keys)

        # Generate medications
        medications = self.generate_medications(condition_keys)

        # Generate clinical notes
        clinical_notes = self.generate_clinical_notes(demographics, conditions, chief_complaint)

        # Create record
        record_id = f"PT{random.randint(100000, 999999)}"
        visit_date = datetime.now() - timedelta(days=random.randint(0, 90))

        return PatientRecord(
            record_id=record_id,
            demographics=demographics,
            chief_complaint=chief_complaint,
            vitals=vitals,
            conditions=conditions,
            medications=medications,
            lab_results=lab_results,
            clinical_notes=clinical_notes,
            visit_date=visit_date
        )

    def generate_batch(self, n: int, diversity_constraints: bool = True) -> List[PatientRecord]:
        """Generate a batch of patient records with diversity constraints"""
        records = []

        if diversity_constraints:
            # Ensure stratified sampling across demographics
            age_groups = DEMOGRAPHIC_DISTRIBUTIONS['age_groups']
            sex_dist = DEMOGRAPHIC_DISTRIBUTIONS['sex']

            # Calculate how many records per stratum
            records_per_age_group = n // len(age_groups)

            for age_group in age_groups:
                for _ in range(records_per_age_group):
                    age = random.randint(*age_group['range'])
                    sex = random.choice([s['value'] for s in sex_dist])

                    constraints = {'age': age, 'sex': sex}
                    record = self.generate_record(constraints)
                    records.append(record)

            # Fill remaining
            remaining = n - len(records)
            for _ in range(remaining):
                records.append(self.generate_record())
        else:
            # Simple random generation
            for _ in range(n):
                records.append(self.generate_record())

        return records
