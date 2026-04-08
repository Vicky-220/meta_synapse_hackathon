import os
import random
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from functools import lru_cache


# ==============================================================================
# LOAD REAL DATASETS FROM HUGGING FACE
# ==============================================================================

def load_medical_datasets() -> Dict:
    """Load and format real medical datasets from Hugging Face."""
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
    # ... per case
}

def calculate_question_reward(case_id: str, question: str) -> float:
    keywords = RELEVANT_QUESTION_KEYWORDS.get(case_id, [])
    q_lower = question.lower()
    matches = sum(1 for kw in keywords if kw in q_lower)
    if matches >= 2: return 0.08
    if matches == 1: return 0.05
    return 0.01

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
    },
}

real_cases = generate_patient_cases_from_datasets()
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
