"""training_wrapper.py — Minimal training-ready wrapper for the Medical Diagnostic Environment.

This module exposes a small async/sync wrapper that is easy to plug into a training
loop or evaluation script. It is not a full RL algorithm, but it makes the
environment easy to consume for model-based training.
"""

import os
import asyncio
from typing import Optional

from client import DiagnosticEnv
from models import DiagnosticAction, PatientObservation


class TrainingEnv:
    """Minimal wrapper exposing a training-friendly environment interface."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("ENV_URL", "ws://localhost:8000/ws")
        self._env = DiagnosticEnv(base_url=self.base_url)

    async def __aenter__(self):
        await self._env.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._env.__aexit__(exc_type, exc, tb)

    async def reset(self, difficulty: str = "easy") -> PatientObservation:
        result = await self._env.reset(difficulty=difficulty)
        return result.observation if hasattr(result, "observation") else result

    async def step(
        self,
        action_type: str,
        question: Optional[str] = None,
        test_name: Optional[str] = None,
        diagnosis: Optional[str] = None,
    ) -> PatientObservation:
        action = DiagnosticAction(
            action_type=action_type,
            question=question,
            test_name=test_name,
            diagnosis=diagnosis,
        )
        result = await self._env.step(action)
        return result.observation if hasattr(result, "observation") else result

    async def state(self):
        return await self._env.state()


async def run_demo():
    """Example usage of the training wrapper."""
    async with TrainingEnv() as env:
        obs = await env.reset(difficulty="easy")
        print("Reset observation:", obs.message)
        result = await env.step(action_type="ask_question", question="Do you have a fever?")
        print("Step result:", result.message)


if __name__ == "__main__":
    asyncio.run(run_demo())
