"""
server/environment.py — Core medical diagnostic environment logic.

This is the BRAIN of the Medical Diagnostic Environment. It:
1. Manages patient cases and episode state
2. Processes agent actions (questions, tests, diagnoses)
3. Calculates rewards based on diagnostic quality
4. Provides trajectory-based reward signals (not sparse)

Pure Python - no HTTP or WebSocket code here.
All logic is deterministic and reproducible.
"""

import random
import uuid
from typing import Dict, List, Optional
from openenv.core.env_server import Environment

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DiagnosticAction, PatientObservation, ClinicalState
from server.medical_data import (
    PATIENT_CASES,
    calculate_question_reward,
    calculate_test_reward,
    calculate_diagnosis_accuracy,
    get_patient_response,
)


class MedicalDiagnosticEnvironment(Environment):
    """
    Medical Diagnostic Environment for RL Training.
    
    Simulates doctor-patient interaction where an LLM agent must:
    1. Ask relevant clinical questions
    2. Order appropriate diagnostic tests
    3. Make accurate diagnoses
    
    The environment provides rich reward signals throughout the trajectory:
    - +0.05 per relevant question asked
    - +0.10 per informative test ordered
    - +1.0 for correct final diagnosis
    - Penalizes inefficient or irrelevant actions
    
    This is NOT a sparse reward environment - the agent sees meaningful progress
    at each step, which is crucial for learning.
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = True  # Allow multiple parallel training sessions
    
    # Episode configuration
    MAX_STEPS = 15  # Maximum actions per episode
    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
    
    def __init__(self):
        """Initialize environment state."""
        super().__init__()
        
        # Episode state variables
        self._episode_id: str = ""
        self._case_id: str = ""
        self._difficulty: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        
        # Patient interaction tracking
        self._questions_asked: List[str] = []
        self._tests_ordered: List[str] = []
        self._test_results: Dict = {}
        self._diagnosis_submitted: Optional[str] = None
        self._final_accuracy: float = 0.0
        
        # Episode status
        self._done: bool = False
        self._episode_reward_breakdown: Dict = {
            "question_rewards": 0.0,
            "test_rewards": 0.0,
            "diagnosis_reward": 0.0,
            "efficiency_penalty": 0.0,
        }

    @property
    def current_case_id(self) -> str:
        """Current patient case identifier."""
        return self._case_id

    @property
    def current_difficulty(self) -> str:
        """Current episode difficulty level."""
        return self._difficulty

    # ─────────────────────────────────────────────────────────────────────
    # Core API Methods
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, difficulty: str = "easy", **kwargs) -> PatientObservation:
        """
        Reset the environment for a new diagnostic episode.
        
        Args:
            difficulty: "easy", "medium", or "hard" (controls case selection)
        
        Returns:
            Initial PatientObservation for the agent to read
        """
        # Initialize episode
        self._episode_id = str(uuid.uuid4())
        self._difficulty = difficulty if difficulty in self.DIFFICULTY_LEVELS else "easy"
        self._case_id = self._select_case_by_difficulty(self._difficulty)
        self._step_count = 0
        self._total_reward = 0.0
        
        # Reset tracking
        self._questions_asked = []
        self._tests_ordered = []
        self._test_results = {}
        self._diagnosis_submitted = None
        self._final_accuracy = 0.0
        self._done = False
        self._episode_reward_breakdown = {
            "question_rewards": 0.0,
            "test_rewards": 0.0,
            "diagnosis_reward": 0.0,
            "efficiency_penalty": 0.0,
        }
        
        # Get case information
        case = PATIENT_CASES[self._case_id]
        
        # Create initial observation
        initial_message = (
            f"Patient presents with: {case['presentation']}\n"
            f"Age: {case['age']}, Gender: {case['gender']}\n"
            f"You have up to {self.MAX_STEPS} steps to diagnose this patient.\n"
            f"Please start by asking questions or ordering tests."
        )
        
        return PatientObservation(
            done=False,
            reward=0.0,
            message=initial_message,
            patient_response=None,
            test_result=None,
            questions_asked=[],
            tests_completed=[],
            patient_data_revealed={
                "age": case["age"],
                "gender": case["gender"],
                "presentation": case["presentation"],
            },
            steps_taken=0,
            max_steps=self.MAX_STEPS,
        )

    def step(self, action: DiagnosticAction, **kwargs) -> PatientObservation:
        """
        Process one diagnostic action (question, test, or diagnosis).
        
        Returns immediate reward and next observation.
        """
        if self._done:
            return self._create_done_observation(
                message="Episode already ended. Call reset() to start a new case."
            )
        
        self._step_count += 1
        step_reward = 0.0
        message = ""
        patient_response = None
        test_result = None
        
        # ── Process action based on type ──
        if action.action_type == "ask_question":
            step_reward, message, patient_response = self._handle_question(action.question)
            
        elif action.action_type == "order_test":
            step_reward, message, test_result = self._handle_test(action.test_name)
            
        elif action.action_type == "submit_diagnosis":
            step_reward, message = self._handle_diagnosis(action.diagnosis)
            self._done = True
            
        else:
            message = f"Unknown action type: {action.action_type}"
            step_reward = -0.05
        
        # Accumulate rewards
        self._total_reward += step_reward
        
        # Check if episode should end
        if self._step_count >= self.MAX_STEPS and not self._done:
            message += f"\nMax steps reached. Episode ending."
            self._done = True
        
        # Get current case for patient data revelation
        case = PATIENT_CASES[self._case_id]
        
        return PatientObservation(
            done=self._done,
            reward=step_reward,
            message=message,
            patient_response=patient_response,
            test_result=test_result,
            questions_asked=self._questions_asked.copy(),
            tests_completed=self._tests_ordered.copy(),
            patient_data_revealed=self._build_patient_data_view(case),
            steps_taken=self._step_count,
            max_steps=self.MAX_STEPS,
        )

    def state(self) -> ClinicalState:
        """
        Return complete internal state (includes hidden information).
        Used for debugging only - NEVER send to agent.
        """
        case = PATIENT_CASES.get(self._case_id, {})
        
        return ClinicalState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            true_diagnosis=case.get("true_diagnosis", ""),
            patient_case=self._case_id,
            patient_id=self._case_id,
            patient_details=case,
            difficulty=self._difficulty,
            questions_asked=self._questions_asked.copy(),
            tests_completed=self._tests_ordered.copy(),
            final_diagnosis_submitted=self._diagnosis_submitted,
            final_accuracy=self._final_accuracy,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Action Processing
    # ─────────────────────────────────────────────────────────────────────

    def _handle_question(self, question: str) -> tuple:
        """
        Process a question about the patient.
        
        Returns:
        (reward, message, patient_response)
        """
        # Calculate reward for asking this question
        reward = calculate_question_reward(self._case_id, question)
        
        # Record question
        self._questions_asked.append(question)
        self._episode_reward_breakdown["question_rewards"] += reward
        
        # Get patient response
        response = get_patient_response(self._case_id, question)
        
        message = f"Patient response: {response}"
        if reward == 0.00:
            message += " (Question may not be directly relevant)"
        elif reward == 0.01:
            message += " (Somewhat relevant question)"
        else:
            message += " (Good clinical question!)"
        
        return reward, message, response

    def _handle_test(self, test_name: str) -> tuple:
        """
        Process a test order.
        
        Returns:
        (reward, message, test_result_dict)
        """
        # Calculate reward for ordering this test
        reward = calculate_test_reward(self._case_id, test_name)
        
        # Get case data
        case = PATIENT_CASES[self._case_id]
        
        # Try to find matching test result
        test_result_data = None
        matched_test_key = None
        
        test_lower = test_name.lower()
        for test_key, result in case.get("test_results", {}).items():
            if test_key.lower() in test_lower or test_lower in test_key.lower():
                test_result_data = result
                matched_test_key = test_key
                break
        
        if test_result_data is None:
            message = f"Test '{test_name}' not available for this patient or unavailable in this setting."
            reward = -0.02
            return reward, message, None
        
        # Record test
        self._tests_ordered.append(matched_test_key)
        self._test_results[matched_test_key] = test_result_data
        self._episode_reward_breakdown["test_rewards"] += reward
        
        # Format test result for agent
        test_result_dict = {
            "test_name": matched_test_key,
            "result": str(test_result_data),
            "interpretation": test_result_data.get("interpretation", test_result_data.get("finding", ""))
        }
        
        message = f"Test result received for {matched_test_key}:\n{test_result_dict['interpretation']}"
        if reward == 0.10:
            message += " (Excellent diagnostic test!)"
        elif reward == 0.05:
            message += " (Useful supporting test)"
        else:
            message += " (Test ordered but may be less relevant)"
        
        return reward, message, test_result_dict

    def _handle_diagnosis(self, diagnosis: str) -> tuple:
        """
        Process final diagnosis submission.
        
        Returns:
        (reward, message)
        """
        # Calculate diagnostic accuracy
        accuracy = calculate_diagnosis_accuracy(self._case_id, diagnosis)
        self._final_accuracy = accuracy
        self._diagnosis_submitted = diagnosis
        
        # Create diagnosis reward (not just accuracy, but also process quality)
        case = PATIENT_CASES[self._case_id]
        true_diagnosis = case["true_diagnosis"]
        
        # Change the if/elif chain to use >= comparisons:
        if accuracy >= 0.95:
            reward = 1.0
            message = f"Correct diagnosis: {diagnosis}"
        elif accuracy >= 0.7:
            reward = accuracy
            message = f"Acceptable diagnosis: {diagnosis}. True: {true_diagnosis}"
        elif accuracy >= 0.3:
            reward = accuracy
            message = f"Partially correct. True: {true_diagnosis}"
        else:
            reward = 0.0
            message = f"Incorrect. True: {true_diagnosis}"
        
        self._episode_reward_breakdown["diagnosis_reward"] = reward
        
        # Add efficiency feedback
        if self._step_count > self.MAX_STEPS * 0.8:
            penalty = 0.1 * (self._step_count / self.MAX_STEPS - 0.8)
            self._episode_reward_breakdown["efficiency_penalty"] = penalty
            message += f"\n(Efficiency penalty: -{penalty:.2f} for taking many steps)"
        
        return reward, message

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def _select_case_by_difficulty(self, difficulty: str) -> str:
        matching_keys = [k for k, v in PATIENT_CASES.items() if v["difficulty"] == difficulty]
        if not matching_keys:
            matching_keys = list(PATIENT_CASES.keys())
        return random.choice(matching_keys)

    def _build_patient_data_view(self, case: Dict) -> Dict:
        """
        Build what the agent has learned about the patient so far.
        Only includes information revealed through questions/tests.
        """
        revealed = {
            "age": case.get("age"),
            "gender": case.get("gender"),
            "presentation": case.get("presentation"),
        }
        
        # Add findings based on questions asked
        findings = case.get("hidden_findings", {})
        for question in self._questions_asked:
            q_lower = question.lower()
            for finding, value in findings.items():
                if finding.lower().replace("_", " ") in q_lower:
                    revealed[f"finding_{finding}"] = value
        
        # Add test results
        if self._test_results:
            revealed["test_results"] = self._test_results
        
        return revealed

    def _create_done_observation(self, message: str) -> PatientObservation:
        """Create a terminal observation."""
        return PatientObservation(
            done=True,
            reward=0.0,
            message=message,
            patient_response=None,
            test_result=None,
            questions_asked=self._questions_asked.copy(),
            tests_completed=self._tests_ordered.copy(),
            patient_data_revealed={},
            steps_taken=self._step_count,
            max_steps=self.MAX_STEPS,
        )

    def get_episode_summary(self) -> Dict:
        """
        Return a summary of the episode for logging/evaluation.
        """
        case = PATIENT_CASES.get(self._case_id, {})
        return {
            "episode_id": self._episode_id,
            "case_id": self._case_id,
            "difficulty": self._difficulty,
            "true_diagnosis": case.get("true_diagnosis", ""),
            "submitted_diagnosis": self._diagnosis_submitted,
            "accuracy": self._final_accuracy,
            "diagnostic_accuracy": self._final_accuracy,
            "total_reward": self._total_reward,
            "steps": self._step_count,
            "steps_taken": self._step_count,
            "max_steps": self.MAX_STEPS,
            "questions_asked": len(self._questions_asked),
            "tests_ordered": len(self._tests_ordered),
            "reward_breakdown": self._episode_reward_breakdown,
        }
