"""
client.py — OpenEnv client for the Medical Diagnostic Environment.

This client enables training code to interact with the environment via WebSocket.
Provides both async and sync interfaces for flexibility.
"""

from typing import Optional
import asyncio
import json

try:
    import websockets
except ImportError:
    websockets = None

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DiagnosticAction, PatientObservation, ClinicalState


class DiagnosticEnv(EnvClient[DiagnosticAction, PatientObservation, ClinicalState]):
    """
    Client for interacting with the Medical Diagnostic Environment.
    
    Supports both async and sync usage:
    
    Async (recommended for training):
        async with DiagnosticEnv(base_url="...") as env:
            obs = await env.reset()
            obs = await env.step(DiagnosticAction(...))
    
    Sync (for notebooks/simple scripts):
        with DiagnosticEnv(base_url="...").sync() as env:
            obs = env.reset()
            obs = env.step(DiagnosticAction(...))
    """
    
    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """Create client connected to a running OpenEnv environment URL."""
        if base_url is None:
            base_url = os.getenv("ENV_URL", "ws://localhost:8000/ws")
        return cls(base_url=base_url, **kwargs)
    
    def _step_payload(self, action: DiagnosticAction) -> dict:
        """Convert action to JSON payload for server."""
        return {
            "action_type": action.action_type,
            "question": action.question,
            "test_name": action.test_name,
            "diagnosis": action.diagnosis,
        }
    
    def _parse_result(self, payload: dict) -> StepResult:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        
        # Parse nested dictionaries if present
        patient_data_revealed = obs_data.get("patient_data_revealed", {})
        if isinstance(patient_data_revealed, str):
            try:
                patient_data_revealed = json.loads(patient_data_revealed)
            except:
                patient_data_revealed = {}
        
        test_result = obs_data.get("test_result")
        if isinstance(test_result, str):
            try:
                test_result = json.loads(test_result)
            except:
                test_result = None
        
        observation = PatientObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            message=obs_data.get("message", ""),
            patient_response=obs_data.get("patient_response"),
            test_result=test_result,
            questions_asked=obs_data.get("questions_asked", []),
            tests_completed=obs_data.get("tests_completed", []),
            patient_data_revealed=patient_data_revealed,
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 15),
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: dict) -> ClinicalState:
        """Parse state response."""
        patient_details = payload.get("patient_details", {})
        if isinstance(patient_details, str):
            try:
                patient_details = json.loads(patient_details)
            except:
                patient_details = {}
        
        return ClinicalState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            true_diagnosis=payload.get("true_diagnosis", ""),
            patient_case=payload.get("patient_case", ""),
            patient_details=patient_details,
            difficulty=payload.get("difficulty", "easy"),
            questions_asked=payload.get("questions_asked", []),
            tests_completed=payload.get("tests_completed", []),
            final_diagnosis_submitted=payload.get("final_diagnosis_submitted"),
            final_accuracy=payload.get("final_accuracy", 0.0),
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Sync wrapper for convenience
    # ─────────────────────────────────────────────────────────────────────
    
    def sync(self) -> "SyncDiagnosticEnv":
        """
        Return synchronous wrapper for use in notebooks/simple scripts.
        
        Usage:
            with DiagnosticEnv(url).sync() as env:
                obs = env.reset()
                obs = env.step(action)
        """
        return SyncDiagnosticEnv(self.base_url)


class SyncDiagnosticEnv:
    """Synchronous wrapper around async client."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._loop = asyncio.new_event_loop()
        self._async_client = DiagnosticEnv(base_url)
    
    def __enter__(self):
        self._loop.run_until_complete(self._async_client.__aenter__())
        return self
    
    def __exit__(self, *args):
        self._loop.run_until_complete(self._async_client.__aexit__(*args))
        self._loop.close()
    
    def reset(self, difficulty: str = "easy") -> PatientObservation:
        """Reset environment and start new episode."""
        return self._loop.run_until_complete(
            self._async_client.reset(difficulty=difficulty)
        )
    
    def step(self, action: DiagnosticAction) -> StepResult:
        """Take a step in the environment."""
        return self._loop.run_until_complete(
            self._async_client.step(action)
        )
    
    def state(self) -> ClinicalState:
        """Get current state (includes hidden information)."""
        return self._loop.run_until_complete(
            self._async_client.state()
        )
