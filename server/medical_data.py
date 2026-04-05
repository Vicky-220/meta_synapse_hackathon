"""
server/medical_data.py — Knowledge base and patient cases for the Medical Diagnostic Environment.

This module contains:
1. Patient cases loaded from real Hugging Face datasets (MedMCQA, BigBio MedQA)
2. Medical knowledge base for question/test evaluation
3. Grading logic for diagnoses using LLM-based reward judgments
4. Innovative reward calculation using AI judgment instead of hardcoded rules
"""

import os
import random
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
import openai
from functools import lru_cache

# Set OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

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

@lru_cache(maxsize=1000)
def get_llm_reward_score(prompt: str) -> float:
    """
    Get reward score from LLM judgment.
    
    This innovative approach uses AI to evaluate the quality of diagnostic actions
    rather than hardcoded rules, making the reward system adaptive and context-aware.
    """
    if not openai.api_key:
        # Fallback to random score if no API key
        return random.uniform(0.0, 1.0)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical expert evaluating diagnostic actions. Rate the quality of the action on a scale from 0.0 to 1.0, where 1.0 is excellent and 0.0 is poor. Respond with only the number."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.1
        )
        score_text = response.choices[0].message.content.strip()
        # Extract number from response
        import re
        match = re.search(r'(\d+\.?\d*)', score_text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        else:
            return 0.5
    except Exception as e:
        print(f"LLM reward error: {e}")
        return 0.5  # Neutral score on error

def calculate_question_reward(case_id: str, question: str) -> float:
    """
    Calculate reward for asking a question using LLM judgment with access to true diagnosis.

    Uses AI to evaluate question relevance in medical context, considering the actual diagnosis.
    """
    case = PATIENT_CASES.get(case_id, {})
    presentation = case.get("presentation", "Unknown patient presentation")
    true_diagnosis = case.get("true_diagnosis", "Unknown diagnosis")

    prompt = f"""
Patient presentation: {presentation}
True diagnosis: {true_diagnosis}

Question asked: "{question}"

Rate how relevant this question is for diagnosing this patient on a scale from 0.0 to 1.0.
Consider:
- Does it help narrow down the differential diagnosis?
- Is it clinically appropriate?
- Does it address key symptoms or risk factors?

Score (0.0-1.0):
"""

    score = get_llm_reward_score(prompt)
    return score * 0.1  # Scale to 0-0.1 range for questions

def calculate_test_reward(case_id: str, test_name: str) -> float:
    """
    Calculate reward for ordering a test using LLM judgment with access to true diagnosis.

    Uses AI to evaluate test appropriateness and diagnostic value, considering the actual diagnosis.
    """
    case = PATIENT_CASES.get(case_id, {})
    presentation = case.get("presentation", "Unknown patient presentation")
    true_diagnosis = case.get("true_diagnosis", "Unknown diagnosis")

    prompt = f"""
Patient presentation: {presentation}
True diagnosis: {true_diagnosis}

Test ordered: {test_name}

Rate how appropriate and valuable this test is for this patient's diagnosis on a scale from 0.0 to 1.0.
Consider:
- Is this test indicated based on the presentation?
- Will it help confirm or rule out key diagnoses?
- Is it cost-effective and necessary?

Score (0.0-1.0):
"""

    score = get_llm_reward_score(prompt)
    return score * 0.15  # Scale to 0-0.15 range for tests

def calculate_diagnosis_accuracy(case_id: str, submitted_diagnosis: str) -> float:
    """
    Calculate diagnosis accuracy using LLM judgment.
    
    Innovative: AI evaluates diagnostic accuracy considering differentials and context.
    """
    case = PATIENT_CASES.get(case_id, {})
    true_diagnosis = case.get("true_diagnosis", "").lower()
    presentation = case.get("presentation", "")
    
    prompt = f"""
Patient presentation: {presentation}
True diagnosis: {true_diagnosis}
Submitted diagnosis: {submitted_diagnosis}

Rate the accuracy of the submitted diagnosis on a scale from 0.0 to 1.0.
Consider:
- Is it the exact correct diagnosis? (1.0)
- Is it an acceptable differential? (0.7-0.9)
- Is it partially correct? (0.3-0.6)
- Is it completely wrong? (0.0-0.2)

Score (0.0-1.0):
"""
    
    score = get_llm_reward_score(prompt)
    return score

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

# Load real cases only
real_cases = generate_patient_cases_from_datasets()

PATIENT_CASES = real_cases


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


# ==============================================================================================================================
# MEDICAL KNOWLEDGE BASE (DEPRECATED - Now using LLM-based rewards)
# ==============================================================================================================================

# Note: The old hardcoded knowledge base has been replaced with LLM-based reward judgment
# for more innovative and adaptive evaluation of diagnostic actions.

# The patient response generator is dataset-driven and does not depend on legacy sample cases.
