import os
import random
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

USE_HF_DATASETS = os.getenv("OPENENV_USE_HF_DATASETS", "false").lower() in ("1", "true", "yes")
DATASET_SEED = os.getenv("OPENENV_DATASET_SEED")
if DATASET_SEED is not None:
    try:
        DATASET_SEED = int(DATASET_SEED)
    except ValueError:
        DATASET_SEED = None
# LOAD REAL DATASETS FROM HUGGING FACE
# ==============================================================================

def load_medical_datasets() -> Dict:
    """Load and format real medical datasets from Hugging Face."""
    if not USE_HF_DATASETS:
        return {}

    if not _DATASETS_AVAILABLE:
        print("Warning: datasets package not installed; skipping Hugging Face dataset loading.")
        return {}

    try:
        # Load MedMCQA dataset (Medical Multiple Choice Questions)
        medmcqa = load_dataset("medmcqa", split="train")
        
        # Load BigBio MedQA dataset (Medical Question Answering)
        medqa = load_dataset("bigbio/med_qa", split="train")
        
        return {
            "medmcqa": medmcqa,
            "medqa": medqa
        }
    except Exception as e:
        print(f"Warning: Could not load datasets: {e}. No dataset cases loaded.")
        return {}

# ==============================================================================
# INNOVATIVE REWARD SYSTEM USING LLM JUDGMENT
# ==============================================================================


RELEVANT_QUESTION_KEYWORDS = {
    "easy_flu": ["fever", "cough", "ache", "fatigue", "onset", "symptom", "contact", "vaccine", "temperature"],
    "easy_uti": ["urination", "burning", "pain", "frequent", "abdominal", "bladder", "kidney"],
    "medium_pneumonia": ["cough", "fever", "breath", "chest", "pain", "smoking", "sputum", "productive"],
    "medium_appendicitis": ["abdominal", "pain", "nausea", "vomiting", "rebound", "right lower quadrant", "appetite", "fever"],
    "hard_endocarditis": ["fever", "murmur", "drug", "iv", "dental", "hemorrhage", "splinter", "heart"],
    "hard_meningitis": ["headache", "neck", "stiff", "fever", "photophobia", "confusion", "vomit", "seizure"],
}

def calculate_question_reward(case_id: str, question: str) -> float:
    keywords = RELEVANT_QUESTION_KEYWORDS.get(case_id, [])
    q_lower = question.lower()
    matches = sum(1 for kw in keywords if kw in q_lower)
    if matches >= 2: return 0.08
    if matches == 1: return 0.05
    return 0.01

TEST_NAME_ALIASES = {
    "complete blood count": "cbc",
    "cbc": "cbc",
    "urinalysis": "urinalysis",
    "urine culture": "urine_culture",
    "blood cultures": "blood_cultures",
    "echocardiogram": "echocardiogram",
    "ct head": "ct_head",
    "ct scan": "ct_head",
    "chest xray": "chest_xray",
    "chest radiograph": "chest_xray",
    "sputum culture": "sputum_culture",
    "rapid flu test": "rapid_flu_test",
    "flu test": "rapid_flu_test",
}


def normalize_test_name(test_name: Optional[str]) -> str:
    if not test_name or not isinstance(test_name, str):
        return ""
    cleaned = test_name.strip().lower()
    return TEST_NAME_ALIASES.get(cleaned, cleaned)


def calculate_test_reward(case_id: str, test_name: Optional[str]) -> float:
    """
    Calculate reward for ordering a diagnostic test.
    
    Returns higher reward for tests that are more relevant to the case.
    """
    if not test_name or not isinstance(test_name, str):
        return -0.02

    case = PATIENT_CASES.get(case_id, {})
    test_results = case.get("test_results", {})
    
    test_lower = normalize_test_name(test_name)
    
    # Check if test is available and relevant
    for test_key in test_results.keys():
        if test_key.lower() in test_lower or test_lower in test_key.lower():
            # Test is available - give reward based on relevance
            if "cbc" in test_lower or "blood" in test_lower:
                return 0.10  # Common useful test
            elif "flu" in test_lower or "influenza" in test_lower:
                return 0.10  # Specific relevant test
            else:
                return 0.05  # Somewhat useful test
    
    # Test not available or irrelevant
    return -0.02

def calculate_diagnosis_accuracy(case_id: str, submitted: str) -> float:
    case = PATIENT_CASES.get(case_id, {})
    s = submitted.lower().strip()
    true = case.get("true_diagnosis", "").lower()
    if s == true: return 1.0
    for acceptable in case.get("correct_diagnoses", []):
        if acceptable.lower() in s or s in acceptable.lower(): return 1.0
    # partial credit
    true_words = set(true.split())
    sub_words = set(s.split())
    overlap = len(true_words & sub_words) / max(len(true_words), 1)
    return round(min(overlap, 0.7), 2)

# ==============================================================================
# PATIENT CASES DATABASE (FORMATTED FROM REAL DATASETS)
# ==============================================================================

def format_medmcqa_to_case(entry: Dict) -> Dict:
    """Format a MedMCQA entry into our case structure."""
    question = entry.get("question", "")
    options = entry.get("options", {})
    correct_answer = entry.get("answer", "")
    subject = entry.get("subject_name", "")
    
    # Create presentation from question
    presentation = f"Patient presents with: {question}"
    
    # Use options as possible diagnoses
    diagnoses = list(options.values())
    true_diagnosis = options.get(correct_answer, diagnoses[0] if diagnoses else "Unknown")
    
    return {
        "case_id": f"medmcqa_{entry.get('id', random.randint(1000,9999))}",
        "difficulty": "medium",  # Default to medium
        "true_diagnosis": true_diagnosis,
        "age": random.randint(25, 75),
        "gender": random.choice(["Male", "Female"]),
        "presentation": presentation,
        "hidden_findings": {},  # Would need more processing
        "test_results": {},
        "correct_diagnoses": [true_diagnosis],
        "differential_diagnoses": diagnoses[:3],  # First 3 options
        "source": "medmcqa"
    }

def format_medqa_to_case(entry: Dict) -> Dict:
    """Format a BigBio MedQA entry into our case structure."""
    question = entry.get("question", "")
    answer = entry.get("answer", "")
    
    presentation = f"Medical question: {question}"
    
    return {
        "case_id": f"medqa_{hash(question) % 10000}",
        "difficulty": "hard",  # MedQA is more complex
        "true_diagnosis": answer,
        "age": random.randint(30, 80),
        "gender": random.choice(["Male", "Female"]),
        "presentation": presentation,
        "hidden_findings": {},
        "test_results": {},
        "correct_diagnoses": [answer],
        "differential_diagnoses": [],
        "source": "medqa"
    }

def generate_patient_cases_from_datasets() -> Dict:
    """Generate patient cases from real Hugging Face datasets."""
    cases = {}
    datasets = load_medical_datasets()
    if not datasets:
        return cases

    if DATASET_SEED is not None:
        random.seed(DATASET_SEED)

    # Generate easy cases from MedMCQA (simpler questions)
    if "medmcqa" in datasets:
        medmcqa_data = datasets["medmcqa"]
        easy_indices = random.sample(range(len(medmcqa_data)), min(3, len(medmcqa_data)))
        for i, idx in enumerate(easy_indices):
            entry = medmcqa_data[idx]
            case = format_medmcqa_to_case(entry)
            case["difficulty"] = "easy"
            cases[f"easy_real_{i}"] = case

    # Generate medium cases
    if "medmcqa" in datasets:
        medmcqa_data = datasets["medmcqa"]
        medium_indices = random.sample(range(len(medmcqa_data)), min(2, len(medmcqa_data)))
        for i, idx in enumerate(medium_indices):
            entry = medmcqa_data[idx]
            case = format_medmcqa_to_case(entry)
            case["difficulty"] = "medium"
            cases[f"medium_real_{i}"] = case

    # Generate hard cases from MedQA
    if "medqa" in datasets:
        medqa_data = datasets["medqa"]
        hard_indices = random.sample(range(len(medqa_data)), min(2, len(medqa_data)))
        for i, idx in enumerate(hard_indices):
            entry = medqa_data[idx]
            case = format_medqa_to_case(entry)
            case["difficulty"] = "hard"
            cases[f"hard_real_{i}"] = case

    return cases

STATIC_PATIENT_CASES = {
    "easy_flu": {
        "case_id": "easy_flu",
        "difficulty": "easy",
        "true_diagnosis": "Seasonal Influenza",
        "age": 28, "gender": "Female",
        "presentation": "Patient presents with sudden fever (38.9°C), body aches, headache, fatigue, and dry cough for 2 days. No shortness of breath.",
        "hidden_findings": {"fever": "38.9°C", "duration": "2 days", "onset": "sudden"},
        "test_results": {
            "rapid_flu_test": {"result": "Positive for Influenza A", "interpretation": "Positive Influenza A — confirms influenza diagnosis"},
            "cbc": {"result": "WBC 9.2, lymphocytosis", "interpretation": "Mild lymphocytosis consistent with viral infection"},
        },
        "correct_diagnoses": ["Seasonal Influenza", "Influenza A", "Flu"],
        "differential_diagnoses": ["COVID-19", "Common Cold", "Strep Throat"],
    },
    "easy_uti": {
        "case_id": "easy_uti",
        "difficulty": "easy",
        "true_diagnosis": "Urinary Tract Infection",
        "age": 35, "gender": "Female",
        "presentation": "Patient presents with frequent urination, burning sensation during urination, and lower abdominal pain for 3 days.",
        "hidden_findings": {"frequency": "frequent urination", "pain": "burning during urination", "duration": "3 days"},
        "test_results": {
            "urinalysis": {"result": "Positive for nitrites, leukocytes >10", "interpretation": "Urinalysis shows signs of bacterial infection consistent with UTI"},
            "urine_culture": {"result": "E. coli >100,000 CFU/mL", "interpretation": "Urine culture confirms E. coli urinary tract infection"},
        },
        "correct_diagnoses": ["Urinary Tract Infection", "UTI", "Bladder Infection"],
        "differential_diagnoses": ["Cystitis", "Pyelonephritis", "Vaginitis"],
    },
    "medium_pneumonia": {
        "case_id": "medium_pneumonia",
        "difficulty": "medium",
        "true_diagnosis": "Community-Acquired Pneumonia",
        "age": 45, "gender": "Male",
        "presentation": "Patient presents with productive cough, fever (39.2°C), shortness of breath, and right-sided chest pain for 5 days. Smoker with 20 pack-year history.",
        "hidden_findings": {"cough": "productive", "fever": "39.2°C", "breathing": "shortness of breath", "smoking": "20 pack-years"},
        "test_results": {
            "chest_xray": {"result": "Right lower lobe consolidation", "interpretation": "Chest X-ray shows consolidation in right lower lobe consistent with pneumonia"},
            "cbc": {"result": "WBC 14.5, neutrophilia", "interpretation": "Elevated white blood cell count with neutrophilia suggesting bacterial infection"},
            "sputum_culture": {"result": "Streptococcus pneumoniae", "interpretation": "Sputum culture positive for Streptococcus pneumoniae"},
        },
        "correct_diagnoses": ["Community-Acquired Pneumonia", "Pneumonia", "Bacterial Pneumonia"],
        "differential_diagnoses": ["Bronchitis", "Pulmonary Embolism", "Lung Cancer"],
    },
    "hard_endocarditis": {
        "case_id": "hard_endocarditis",
        "difficulty": "hard",
        "true_diagnosis": "Infective Endocarditis",
        "age": 55, "gender": "Male",
        "presentation": "Patient with history of IV drug use presents with fever (38.8°C), new heart murmur, and splinter hemorrhages. Recent dental procedure 2 weeks ago.",
        "hidden_findings": {"drug_use": "IV drug user", "murmur": "new heart murmur", "hemorrhages": "splinter hemorrhages", "dental": "recent dental procedure"},
        "test_results": {
            "blood_cultures": {"result": "Staphylococcus aureus in 3/3 bottles", "interpretation": "Blood cultures positive for Staphylococcus aureus in multiple bottles"},
            "echocardiogram": {"result": "Vegetation on aortic valve", "interpretation": "Echocardiogram shows vegetation on aortic valve consistent with endocarditis"},
            "cbc": {"result": "WBC 12.8, anemia", "interpretation": "Elevated white blood cells with anemia of chronic disease"},
        },
        "correct_diagnoses": ["Infective Endocarditis", "Endocarditis", "Bacterial Endocarditis"],
        "differential_diagnoses": ["Sepsis", "Acute Rheumatic Fever", "Myocardial Infarction"],
    },
    "medium_appendicitis": {
        "case_id": "medium_appendicitis",
        "difficulty": "medium",
        "true_diagnosis": "Acute Appendicitis",
        "age": 23, "gender": "Female",
        "presentation": "Patient presents with right lower quadrant abdominal pain, nausea, anorexia, and low-grade fever for 24 hours.",
        "hidden_findings": {"pain": "right lower quadrant", "nausea": "yes", "anorexia": "yes", "fever": "low-grade"},
        "test_results": {
            "abdominal_ultrasound": {"result": "Enlarged appendix with periappendiceal fluid", "interpretation": "Findings are consistent with acute appendicitis"},
            "cbc": {"result": "WBC 13.4, neutrophilia", "interpretation": "Elevated white blood cell count with neutrophils suggests acute inflammation"},
            "urinalysis": {"result": "Trace leukocytes", "interpretation": "Urinalysis slightly abnormal but not diagnostic"},
        },
        "correct_diagnoses": ["Acute Appendicitis", "Appendicitis"],
        "differential_diagnoses": ["Ovarian Cyst", "Ectopic Pregnancy", "Gastroenteritis"],
    },
    "hard_meningitis": {
        "case_id": "hard_meningitis",
        "difficulty": "hard",
        "true_diagnosis": "Bacterial Meningitis",
        "age": 34, "gender": "Male",
        "presentation": "Patient presents with severe headache, neck stiffness, fever, photophobia, and confusion over the last 12 hours.",
        "hidden_findings": {"headache": "severe", "neck": "stiff", "fever": "high", "photophobia": "yes"},
        "test_results": {
            "lumbar_puncture": {"result": "Cloudy CSF with neutrophil predominance", "interpretation": "CSF findings are consistent with bacterial meningitis"},
            "blood_cultures": {"result": "Gram-positive cocci in pairs", "interpretation": "Blood cultures positive for likely Streptococcus pneumoniae"},
            "ct_head": {"result": "No mass effect or hemorrhage", "interpretation": "CT head is unremarkable prior to lumbar puncture"},
        },
        "correct_diagnoses": ["Bacterial Meningitis", "Meningitis"],
        "differential_diagnoses": ["Viral Meningitis", "Migraine", "Subarachnoid Hemorrhage"],
    },
}

real_cases = generate_patient_cases_from_datasets() if USE_HF_DATASETS else {}
# Merge: static cases are always available, real cases supplement
PATIENT_CASES = {**STATIC_PATIENT_CASES, **real_cases}


# ==============================================================================================================================
# PATIENT RESPONSE GENERATION
# ==============================================================================================================================

def get_patient_response(case_id: str, question: str) -> str:
    """
    Generate a patient response to a question.

    For dataset-driven cases, responses are generic but clinically plausible.
    """
    question_lower = question.lower()
    if "pain" in question_lower:
        return "Yes, I am experiencing pain in that area."
    if "fever" in question_lower or "temperature" in question_lower:
        return "I feel warm and may have a fever."
    if "nausea" in question_lower or "vomit" in question_lower:
        return "Yes, I am nauseated and may vomit."
    if "cough" in question_lower or "breath" in question_lower:
        return "I have some coughing and breathing discomfort."
    if "symptom" in question_lower or "feel" in question_lower:
        return "I have concerning symptoms right now."
    return "I'm not sure about that. Can you ask in a different way?"
