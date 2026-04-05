"""
Unit tests for Medical Diagnostic Environment

Run with: python -m pytest tests/test_environment.py -v
"""

import pytest
from server.environment import MedicalDiagnosticEnvironment
from server.medical_data import (
    PATIENT_CASES,
    calculate_question_reward,
    calculate_test_reward,
    calculate_diagnosis_accuracy,
)
from models import DiagnosticAction


class TestMedicalDiagnosticEnvironment:
    """Test suite for MedicalDiagnosticEnvironment"""

    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test"""
        return MedicalDiagnosticEnvironment()

    def test_environment_initialization(self, env):
        """Test that environment initializes correctly"""
        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "state")

    def test_reset_easy(self, env):
        """Test reset with easy difficulty"""
        observation = env.reset(difficulty="easy")
        assert observation is not None
        assert hasattr(observation, "message")
        assert "presentation" in observation.message.lower() or "patient" in observation.message.lower()
        assert env.current_case_id is not None
        assert env.current_difficulty == "easy"

    def test_reset_medium(self, env):
        """Test reset with medium difficulty"""
        observation = env.reset(difficulty="medium")
        assert observation is not None
        assert env.current_difficulty == "medium"

    def test_reset_hard(self, env):
        """Test reset with hard difficulty"""
        observation = env.reset(difficulty="hard")
        assert observation is not None
        assert env.current_difficulty == "hard"

    def test_ask_question_action(self, env):
        """Test asking a question"""
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="ask_question",
            question="Does the patient have a fever?"
        )
        result = env.step(action)
        assert result is not None
        assert hasattr(result, "reward")
        assert result.reward >= 0  # Questions give non-negative reward

    def test_order_test_action(self, env):
        """Test ordering a test"""
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="order_test",
            test_name="Complete Blood Count"
        )
        result = env.step(action)
        assert result is not None
        assert hasattr(result, "reward")
        assert result.reward >= 0  # Tests give non-negative reward

    def test_submit_diagnosis_action(self, env):
        """Test submitting a diagnosis"""
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="submit_diagnosis",
            diagnosis="Common Flu"
        )
        result = env.step(action)
        assert result is not None
        assert hasattr(result, "reward")
        assert result.done is True  # Episode should end on diagnosis

    def test_max_steps_enforcement(self, env):
        """Test that episodes end after max steps"""
        env.reset(difficulty="easy")
        for _ in range(15):  # Max 15 steps
            action = DiagnosticAction(
                action_type="ask_question",
                question="Test question"
            )
            result = env.step(action)
            if result.done:
                break
        assert result.done is True

    def test_episode_summary(self, env):
        """Test episode summary generation"""
        env.reset(difficulty="easy")
        action = DiagnosticAction(
            action_type="submit_diagnosis",
            diagnosis="Test Diagnosis"
        )
        env.step(action)
        summary = env.get_episode_summary()
        assert summary is not None
        assert "case_id" in summary
        assert "difficulty" in summary
        assert "accuracy" in summary
        assert "total_reward" in summary
        assert "steps" in summary

    def test_state_property(self, env):
        """Test the state property"""
        env.reset(difficulty="easy")
        state = env.state
        assert state is not None
        assert hasattr(state, "patient_id")
        assert hasattr(state, "step_count")
        assert hasattr(state, "true_diagnosis")

    def test_concurrent_sessions(self):
        """Test that environment supports concurrent sessions"""
        env = MedicalDiagnosticEnvironment()
        assert env.SUPPORTS_CONCURRENT_SESSIONS is True

    def test_multiple_episodes(self, env):
        """Test running multiple episodes"""
        for difficulty in ["easy", "medium", "hard"]:
            observation = env.reset(difficulty=difficulty)
            assert observation is not None
            assert env.current_difficulty == difficulty


class TestMedicalData:
    """Test suite for medical data functions"""

    def test_question_reward_calculation(self):
        """Test question reward calculation"""
        # This is case-specific, so we just verify the function works
        case_id = next(iter(PATIENT_CASES))
        reward = calculate_question_reward(
            case_id=case_id,
            question="Does the patient have a fever?"
        )
        assert 0.0 <= reward <= 1.0

    def test_test_reward_calculation(self):
        """Test test reward calculation"""
        case_id = next(iter(PATIENT_CASES))
        reward = calculate_test_reward(
            case_id=case_id,
            test_name="CBC"
        )
        assert 0.0 <= reward <= 1.0

    def test_diagnosis_accuracy_exact_match(self):
        """Test exact diagnosis match"""
        case_id = next(iter(PATIENT_CASES))
        accuracy = calculate_diagnosis_accuracy(
            case_id=case_id,
            submitted_diagnosis=PATIENT_CASES[case_id].get("true_diagnosis", "")
        )
        assert accuracy == 1.0

    def test_diagnosis_accuracy_partial(self):
        """Test partial diagnosis accuracy"""
        case_id = next(iter(PATIENT_CASES))
        accuracy = calculate_diagnosis_accuracy(
            case_id=case_id,
            submitted_diagnosis="Pneumonia"
        )
        assert 0.0 <= accuracy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
