"""
models.py — Type-safe contracts for the Medical Diagnostic Environment.

These Pydantic models define the interface between the LLM agent and the environment:
- DiagnosticAction: What the agent sends (questions, tests, diagnoses)
- PatientObservation: What the agent receives (feedback, test results, progress)
- ClinicalState: Full episode state (for debugging, not sent to agent)

Using type-safe models enables:
✓ IDE autocomplete for agents
✓ Automatic JSON validation
✓ Clear API documentation
✓ Type-checking at runtime
"""

from typing import Optional, List, Dict
from openenv.core.env_server import Action, Observation, State


class DiagnosticAction(Action):
    """
    Actions the LLM agent can take during diagnosis.
    
    The agent must choose one action per step:
    1. ask_question: Gather patient history
    2. order_test: Request diagnostic test results
    3. submit_diagnosis: Make final diagnosis (ends episode)
    """
    
    action_type: str  # "ask_question", "order_test", "submit_diagnosis"
    question: Optional[str] = None  # Used when action_type="ask_question"
    test_name: Optional[str] = None  # Used when action_type="order_test"
    diagnosis: Optional[str] = None  # Used when action_type="submit_diagnosis"


class PatientObservation(Observation):
    """
    What the agent observes after taking an action.
    
    Inherits from Observation:
    - done: bool → Is the episode over?
    - reward: Optional[float] → Reward signal
    
    Adds medical-specific fields:
    - message: Human-readable feedback
    - patient_response: Answer to question (if applicable)
    - test_result: Test outcome with interpretation
    - questions_asked: History of all questions
    - tests_completed: History of all completed tests
    - patient_data_revealed: What the agent has discovered so far
    """
    
    message: str  # Feedback from environment
    patient_response: Optional[str] = None  # Answer to a question asked
    test_result: Optional[Dict] = None  # {"test_name": "X", "result": "...", "interpretation": "..."}
    questions_asked: List[str] = None  # Questions asked so far
    tests_completed: List[str] = None  # Tests ordered so far
    patient_data_revealed: Dict = None  # Current knowledge about patient
    steps_taken: int = 0  # How many actions so far
    max_steps: int = 15  # Maximum steps allowed
    
    def __post_init__(self):
        if self.questions_asked is None:
            self.questions_asked = []
        if self.tests_completed is None:
            self.tests_completed = []
        if self.patient_data_revealed is None:
            self.patient_data_revealed = {}


class ClinicalState(State):
    """
    Complete internal state snapshot. Contains hidden information (diagnosis, true findings).
    Use for debugging only - NEVER send to agent.
    
    Inherits from State:
    - episode_id: str → Unique episode identifier
    - step_count: int → Current step number
    
    Adds clinical fields:
    - true_diagnosis: The correct diagnosis (hidden from agent)
    - patient_case: Case identifier
    - patient_details: Full patient information (hidden)
    - difficulty: ease|medium|hard
    """
    
    true_diagnosis: str = ""
    patient_case: str = ""
    patient_id: str = ""
    patient_details: Dict = None
    difficulty: str = "easy"
    questions_asked: List[str] = None
    tests_completed: List[str] = None
    final_diagnosis_submitted: Optional[str] = None
    final_accuracy: float = 0.0
    
    def __post_init__(self):
        if self.patient_details is None:
            self.patient_details = {}
        if self.questions_asked is None:
            self.questions_asked = []
        if self.tests_completed is None:
            self.tests_completed = []
