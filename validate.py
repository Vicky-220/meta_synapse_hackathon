#!/usr/bin/env python3
"""
Quick validation script for Medical Diagnostic Environment

This script validates that the core environment works correctly without
requiring the server to be running or external dependencies beyond models.

Run with: python validate.py
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import DiagnosticAction, PatientObservation, ClinicalState
from server.environment import MedicalDiagnosticEnvironment
from server.medical_data import (
    PATIENT_CASES,
    calculate_question_reward,
    calculate_test_reward,
    calculate_diagnosis_accuracy,
)


class ValidationResult:
    """Result of a validation check"""
    def __init__(self, name: str, passed: bool, error: str = None):
        self.name = name
        self.passed = passed
        self.error = error

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f"{status}: {self.name}"
        if self.error:
            msg += f"\n  Error: {self.error}"
        return msg


def validate_imports() -> ValidationResult:
    """Check that all imports work"""
    try:
        from models import DiagnosticAction, PatientObservation, ClinicalState
        from server.environment import MedicalDiagnosticEnvironment
        from server.medical_data import (
            calculate_question_reward,
            calculate_test_reward,
            calculate_diagnosis_accuracy,
        )
        return ValidationResult("Imports", True)
    except Exception as e:
        return ValidationResult("Imports", False, str(e))


def validate_model_creation() -> ValidationResult:
    """Check that models can be instantiated"""
    try:
        action = DiagnosticAction(
            action_type="ask_question",
            question="Test question?"
        )
        assert action.action_type == "ask_question"
        assert action.question == "Test question?"
        return ValidationResult("Model Creation", True)
    except Exception as e:
        return ValidationResult("Model Creation", False, str(e))


def validate_environment_init() -> ValidationResult:
    """Check that environment initializes"""
    try:
        env = MedicalDiagnosticEnvironment()
        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        return ValidationResult("Environment Initialization", True)
    except Exception as e:
        return ValidationResult("Environment Initialization", False, str(e))


def validate_reset_all_difficulties() -> ValidationResult:
    """Check that reset works for all difficulties"""
    try:
        env = MedicalDiagnosticEnvironment()
        for difficulty in ["easy", "medium", "hard"]:
            obs = env.reset(difficulty=difficulty)
            assert obs is not None
            assert env.current_difficulty == difficulty
            assert env.current_case_id is not None
        return ValidationResult("Reset All Difficulties", True)
    except Exception as e:
        return ValidationResult("Reset All Difficulties", False, str(e))


def validate_question_action() -> ValidationResult:
    """Check that asking questions works"""
    try:
        env = MedicalDiagnosticEnvironment()
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="ask_question",
            question="Does the patient have symptoms?"
        )
        result = env.step(action)
        assert result is not None
        assert result.reward >= 0
        assert result.done is False  # Should not end on question
        return ValidationResult("Question Action", True)
    except Exception as e:
        return ValidationResult("Question Action", False, str(e))


def validate_test_action() -> ValidationResult:
    """Check that ordering tests works"""
    try:
        env = MedicalDiagnosticEnvironment()
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="order_test",
            test_name="Complete Blood Count"
        )
        result = env.step(action)
        assert result is not None
        assert result.reward >= 0
        assert result.done is False  # Should not end on test
        return ValidationResult("Test Action", True)
    except Exception as e:
        return ValidationResult("Test Action", False, str(e))


def validate_diagnosis_action() -> ValidationResult:
    """Check that diagnosis submission works"""
    try:
        env = MedicalDiagnosticEnvironment()
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="submit_diagnosis",
            diagnosis="Common Flu"
        )
        result = env.step(action)
        assert result is not None
        assert result.reward is not None
        assert result.done is True  # Should end on diagnosis
        return ValidationResult("Diagnosis Action", True)
    except Exception as e:
        return ValidationResult("Diagnosis Action", False, str(e))


def validate_episode_summary() -> ValidationResult:
    """Check that episode summaries are generated correctly"""
    try:
        env = MedicalDiagnosticEnvironment()
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="submit_diagnosis",
            diagnosis="Test"
        )
        env.step(action)
        summary = env.get_episode_summary()
        assert summary is not None
        assert "case_id" in summary
        assert "difficulty" in summary
        assert "accuracy" in summary
        assert "total_reward" in summary
        assert "steps" in summary
        return ValidationResult("Episode Summary", True)
    except Exception as e:
        return ValidationResult("Episode Summary", False, str(e))


def validate_reward_functions() -> ValidationResult:
    """Check that reward functions work"""
    try:
        case_id = next(iter(PATIENT_CASES))
        q_reward = calculate_question_reward(case_id, "Test question?")
        assert isinstance(q_reward, float)
        assert 0.0 <= q_reward <= 1.0
        
        t_reward = calculate_test_reward(case_id, "CBC")
        assert isinstance(t_reward, float)
        assert 0.0 <= t_reward <= 1.0
        
        true_diag = PATIENT_CASES[case_id].get("true_diagnosis", "")
        d_accuracy = calculate_diagnosis_accuracy(case_id, true_diag)
        assert isinstance(d_accuracy, float)
        assert 0.0 <= d_accuracy <= 1.0
        
        return ValidationResult("Reward Functions", True)
    except Exception as e:
        return ValidationResult("Reward Functions", False, str(e))


def validate_state_property() -> ValidationResult:
    """Check that state property works"""
    try:
        env = MedicalDiagnosticEnvironment()
        env.reset(difficulty="easy")
        state = env.state
        assert state is not None
        assert hasattr(state, "patient_id")
        assert hasattr(state, "step_count")
        assert hasattr(state, "true_diagnosis")
        return ValidationResult("State Property", True)
    except Exception as e:
        return ValidationResult("State Property", False, str(e))


def validate_concurrent_support() -> ValidationResult:
    """Check that environment supports concurrent sessions"""
    try:
        env = MedicalDiagnosticEnvironment()
        assert hasattr(env, "SUPPORTS_CONCURRENT_SESSIONS")
        assert env.SUPPORTS_CONCURRENT_SESSIONS is True
        return ValidationResult("Concurrent Sessions Support", True)
    except Exception as e:
        return ValidationResult("Concurrent Sessions Support", False, str(e))


def main():
    """Run all validation checks"""
    print("=" * 70)
    print("MEDICAL DIAGNOSTIC ENVIRONMENT - VALIDATION SUITE")
    print("=" * 70)
    print()

    validators = [
        validate_imports,
        validate_model_creation,
        validate_environment_init,
        validate_reset_all_difficulties,
        validate_question_action,
        validate_test_action,
        validate_diagnosis_action,
        validate_episode_summary,
        validate_reward_functions,
        validate_state_property,
        validate_concurrent_support,
    ]

    results: List[ValidationResult] = []
    for validator in validators:
        try:
            result = validator()
        except Exception as e:
            result = ValidationResult(
                validator.__name__,
                False,
                traceback.format_exc()
            )
        results.append(result)
        print(result)

    print()
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
