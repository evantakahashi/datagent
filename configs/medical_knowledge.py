"""Medical Knowledge Base for Validation and Generation"""

# Common medical conditions with constraints
MEDICAL_CONDITIONS = {
    "type_2_diabetes": {
        "name": "Type 2 Diabetes Mellitus",
        "icd10": "E11.9",
        "age_range": (30, 85),
        "common_demographics": ["all"],
        "required_labs": ["glucose", "hba1c"],
        "common_medications": ["metformin", "insulin"],
        "symptoms": ["increased thirst", "frequent urination", "fatigue"],
    },
    "hypertension": {
        "name": "Essential Hypertension",
        "icd10": "I10",
        "age_range": (30, 90),
        "common_demographics": ["all"],
        "required_labs": [],
        "common_medications": ["lisinopril", "amlodipine", "hydrochlorothiazide"],
        "symptoms": ["headache", "dizziness"],
    },
    "asthma": {
        "name": "Asthma",
        "icd10": "J45.909",
        "age_range": (5, 80),
        "common_demographics": ["all"],
        "required_labs": [],
        "common_medications": ["albuterol", "fluticasone"],
        "symptoms": ["wheezing", "shortness of breath", "chest tightness"],
    },
    "copd": {
        "name": "COPD",
        "icd10": "J44.9",
        "age_range": (50, 90),
        "common_demographics": ["all"],
        "required_labs": [],
        "common_medications": ["albuterol", "tiotropium"],
        "symptoms": ["chronic cough", "dyspnea", "sputum production"],
    },
    "cad": {
        "name": "Coronary Artery Disease",
        "icd10": "I25.10",
        "age_range": (45, 90),
        "common_demographics": ["all"],
        "required_labs": ["cholesterol", "ldl", "hdl"],
        "common_medications": ["atorvastatin", "aspirin"],
        "symptoms": ["chest pain", "dyspnea"],
    },
    "ckd": {
        "name": "Chronic Kidney Disease Stage 3",
        "icd10": "N18.3",
        "age_range": (50, 90),
        "common_demographics": ["all"],
        "required_labs": ["creatinine", "egfr"],
        "common_medications": [],
        "symptoms": ["fatigue", "edema"],
    },
}

# Laboratory test reference ranges
LAB_REFERENCE_RANGES = {
    "glucose": {
        "unit": "mg/dL",
        "normal_range": (70, 100),
        "possible_range": (40, 600),
        "reference_text": "70-100 mg/dL",
    },
    "hba1c": {
        "unit": "%",
        "normal_range": (4.0, 5.6),
        "possible_range": (3.5, 15.0),
        "reference_text": "4.0-5.6%",
    },
    "creatinine": {
        "unit": "mg/dL",
        "normal_range": (0.6, 1.2),
        "possible_range": (0.3, 15.0),
        "reference_text": "0.6-1.2 mg/dL",
    },
    "egfr": {
        "unit": "mL/min/1.73m2",
        "normal_range": (90, 120),
        "possible_range": (5, 150),
        "reference_text": ">60 mL/min/1.73m2",
    },
    "cholesterol": {
        "unit": "mg/dL",
        "normal_range": (125, 200),
        "possible_range": (100, 400),
        "reference_text": "<200 mg/dL",
    },
    "ldl": {
        "unit": "mg/dL",
        "normal_range": (0, 100),
        "possible_range": (20, 300),
        "reference_text": "<100 mg/dL",
    },
    "hdl": {
        "unit": "mg/dL",
        "normal_range": (40, 80),
        "possible_range": (20, 120),
        "reference_text": ">40 mg/dL",
    },
    "tsh": {
        "unit": "mIU/L",
        "normal_range": (0.4, 4.0),
        "possible_range": (0.01, 100.0),
        "reference_text": "0.4-4.0 mIU/L",
    },
}

# Medication information
MEDICATIONS = {
    "metformin": {
        "dosages": ["500mg", "850mg", "1000mg"],
        "frequencies": ["once daily", "twice daily"],
        "indication": "Type 2 Diabetes",
    },
    "insulin": {
        "dosages": ["10 units", "20 units", "variable"],
        "frequencies": ["before meals", "twice daily"],
        "indication": "Diabetes",
    },
    "lisinopril": {
        "dosages": ["10mg", "20mg", "40mg"],
        "frequencies": ["once daily"],
        "indication": "Hypertension",
    },
    "amlodipine": {
        "dosages": ["5mg", "10mg"],
        "frequencies": ["once daily"],
        "indication": "Hypertension",
    },
    "atorvastatin": {
        "dosages": ["10mg", "20mg", "40mg", "80mg"],
        "frequencies": ["once daily at bedtime"],
        "indication": "Hyperlipidemia",
    },
    "aspirin": {
        "dosages": ["81mg"],
        "frequencies": ["once daily"],
        "indication": "Cardiovascular disease prevention",
    },
    "albuterol": {
        "dosages": ["90mcg"],
        "frequencies": ["2 puffs every 4-6 hours as needed"],
        "indication": "Asthma/COPD",
    },
}

# Vital sign ranges
VITAL_SIGN_RANGES = {
    "heart_rate": {
        "normal": (60, 100),
        "possible": (30, 250),
    },
    "systolic_bp": {
        "normal": (90, 120),
        "possible": (60, 250),
    },
    "diastolic_bp": {
        "normal": (60, 80),
        "possible": (40, 150),
    },
    "temperature": {
        "normal": (97.0, 99.0),
        "possible": (95.0, 106.0),
    },
    "respiratory_rate": {
        "normal": (12, 20),
        "possible": (8, 40),
    },
    "oxygen_saturation": {
        "normal": (95, 100),
        "possible": (70, 100),
    },
}

# Demographic distributions (for diversity)
DEMOGRAPHIC_DISTRIBUTIONS = {
    "age_groups": [
        {"range": (18, 30), "label": "young_adult", "weight": 0.15},
        {"range": (31, 50), "label": "middle_age", "weight": 0.30},
        {"range": (51, 65), "label": "older_adult", "weight": 0.30},
        {"range": (66, 85), "label": "geriatric", "weight": 0.25},
    ],
    "sex": [
        {"value": "male", "weight": 0.48},
        {"value": "female", "weight": 0.50},
        {"value": "other", "weight": 0.02},
    ],
    "ethnicity": [
        {"value": "caucasian", "weight": 0.40},
        {"value": "african_american", "weight": 0.25},
        {"value": "hispanic", "weight": 0.20},
        {"value": "asian", "weight": 0.12},
        {"value": "other", "weight": 0.03},
    ],
    "locations": [
        "New York, NY",
        "Los Angeles, CA",
        "Chicago, IL",
        "Houston, TX",
        "Phoenix, AZ",
        "Philadelphia, PA",
        "San Antonio, TX",
        "San Diego, CA",
        "Dallas, TX",
        "Atlanta, GA",
    ],
}

# Sex-specific condition exclusions
SEX_SPECIFIC_CONDITIONS = {
    "male_only": [],
    "female_only": [],
    "excluded": {
        "male": ["pregnancy", "ovarian_cancer", "cervical_cancer"],
        "female": ["prostate_cancer", "testicular_cancer"],
    },
}
