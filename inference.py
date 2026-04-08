"""
inference.py — Baseline inference script for the Medical Diagnostic Environment.
This script demonstrates how to run the environment with an LLM (using OpenAI client)

and logs results in the exact format required by the hackathon.

Format requirements:
[START] task=<task_name> env=<env_name> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<float> done=<bool> error=<str|null>
[END] success=<bool> steps=<n> score=<float> rewards=<comma_separated_list>

All fields on a single line with NO NEWLINES within a line.
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional, Dict

from openai import OpenAI
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import DiagnosticEnv
from models import DiagnosticAction


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Configuration
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "medical-diagnostic-env:latest")
ENV_URL = os.getenv("ENV_URL", "ws://localhost:8000/ws")
BENCHMARK = os.getenv("BENCHMARK", "medical_diagnostic_env")

# Inference configuration
MAX_STEPS = 15  # Maximum steps per episode
TEMPERATURE = 0.7  # LLM temperature for reasoning
MAX_TOKENS = 256  # Max tokens per completion
TASK_NAMES = ["easy_diagnosis", "medium_diagnosis", "hard_diagnosis"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


# ==============================================================================
# LOGGING FUNCTIONS
# ==============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in required format."""
    # Clean model name for logging
    model_clean = model.split("/")[-1] if "/" in model else model
    print(f"[START] task={task} env={env} model={model_clean}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log single step in required format."""
    error_val = f'"{error}"' if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in required format."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ==============================================================================
# LLM INTERACTION
# ==============================================================================

def create_system_prompt() -> str:
    """Create system prompt for medical diagnostic reasoning."""
    return textwrap.dedent("""
        You are an expert medical diagnostic AI assistant. Your role is to:
        
        1. GATHER INFORMATION: Ask relevant clinical questions about symptoms, 
           history, and presentation.
        
        2. ORDER TESTS: Request appropriate diagnostic tests based on the 
           differential diagnosis.
        
        3. REASON DIAGNOSTICALLY: Consider the patient's presentation, 
           synthesize findings, and make a diagnosis.
        
        Your reasoning should follow clinical guidelines and prioritize:
        - Life-threatening conditions first (red flags)
        - Most common diagnoses for the presentation
        - Efficiency (minimize unnecessary tests)
        
        When responding, use EXACTLY ONE of these actions:
        
        ACTION: ask_question
        QUESTION: <your question here>
        
        OR
        
        ACTION: order_test
        TEST: <test name>
        
        OR
        
        ACTION: submit_diagnosis
        DIAGNOSIS: <final diagnosis>
        
        Be concise. Diagnose within 10-15 steps if possible.
    """).strip()


def extract_action_from_response(response: str) -> Optional[Dict]:
    """
    Extract structured action from LLM response.
    
    Returns dict with:
    {
        "action_type": "ask_question" | "order_test" | "submit_diagnosis",
        "question": str or None,
        "test_name": str or None,
        "diagnosis": str or None,
    }
    """
    response_lower = response.lower()
    
    # Try to find ACTION: directive
    if "action:" in response_lower:
        lines = response.split("\n")
        action_type = None
        question = None
        test_name = None
        diagnosis = None
        
        for i, line in enumerate(lines):
            if "action:" in line.lower():
                action_part = line.split(":", 1)[1].strip().lower()
                if "question" in action_part:
                    action_type = "ask_question"
                elif "test" in action_part:
                    action_type = "order_test"
                elif "diagnosis" in action_part:
                    action_type = "submit_diagnosis"
            
            if "question:" in line.lower():
                question = line.split(":", 1)[1].strip()
            elif "test:" in line.lower():
                test_name = line.split(":", 1)[1].strip()
            elif "diagnosis:" in line.lower():
                diagnosis = line.split(":", 1)[1].strip()
        
        if action_type:
            return {
                "action_type": action_type,
                "question": question,
                "test_name": test_name,
                "diagnosis": diagnosis,
            }
    
    # Fallback: try to infer action type
    if "question" in response_lower or "ask" in response_lower:
        # Extract the question
        for line in response.split("\n"):
            if "?" in line:
                return {
                    "action_type": "ask_question",
                    "question": line.strip(),
                    "test_name": None,
                    "diagnosis": None,
                }
    
    if "test" in response_lower or "order" in response_lower:
        # Try to extract test name
        words = response.split()
        for i, word in enumerate(words):
            if "test" in word.lower() and i + 1 < len(words):
                test = words[i + 1]
                return {
                    "action_type": "order_test",
                    "question": None,
                    "test_name": test,
                    "diagnosis": None,
                }
    
    if "diagnos" in response_lower:
        # Try to extract diagnosis
        for word in response.split():
            if len(word) > 3:  # Filter out small words
                return {
                    "action_type": "submit_diagnosis",
                    "question": None,
                    "test_name": None,
                    "diagnosis": response.strip(),
                }
    
    return None


def build_conversation_history(episode_history: List[Dict]) -> List[Dict]:
    """Build conversation history for multi-turn interaction."""
    conversation = [
        {
            "role": "system",
            "content": create_system_prompt(),
        }
    ]
    
    for turn in episode_history:
        # Add assistant message
        if turn.get("agent_action"):
            conversation.append({
                "role": "assistant",
                "content": turn["agent_action"],
            })
        
        # Add environment feedback
        if turn.get("environment_feedback"):
            conversation.append({
                "role": "user",
                "content": turn["environment_feedback"],
            })
    
    return conversation


# ==============================================================================
# EPISODE EXECUTION
# ==============================================================================

async def run_episode_async(
    client: OpenAI,
    image_name: str,
    difficulty: str,
    task_name: str,
) -> Dict:
    """
    Run a single episode with asyncio.
    
    Returns: {
        "task": task_name,
        "success": bool,
        "steps_taken": int,
        "total_reward": float,
        "episode_rewards": [float],
        "final_diagnosis_accuracy": float,
    }
    """
    
    log_start(task_name, "medical_diagnostic_env", MODEL_NAME)
    
    # Reset environment
    async with DiagnosticEnv.from_docker_image(image_name=image_name, base_url=ENV_URL) as env:
        obs_result = await env.reset(difficulty=difficulty)
        obs = obs_result.observation if hasattr(obs_result, 'observation') else obs_result
        
        episode_history = []
        episode_rewards = []
        step_count = 0
        error_occurred = False
        
        # Initial environment message
        initial_message = f"Patient presentation: {obs.message}"
        
        while step_count < MAX_STEPS and not obs.done:
            step_count += 1
            
            # Build conversation with history
            conversation = [
                {
                    "role": "system",
                    "content": create_system_prompt(),
                }
            ]
            
            # Add conversation history
            for turn in episode_history:
                if turn.get("agent_thought"):
                    conversation.append({
                        "role": "assistant",
                        "content": f"Thinking: {turn['agent_thought']}\nAction: {turn['agent_action']}",
                    })
                if turn.get("environment_response"):
                    conversation.append({
                        "role": "user",
                        "content": turn["environment_response"],
                    })
            
            # Add current observation if first step
            if step_count == 1:
                conversation.append({
                    "role": "user",
                    "content": initial_message,
                })
            
            try:
                # Get LLM response
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=conversation,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                
                llm_response = response.choices[0].message.content
                
                # Extract action from response
                action_dict = extract_action_from_response(llm_response)
                
                if not action_dict:
                    error_msg = "Could not parse action from response"
                    log_step(step_count, "parse_error", 0.0, False, error_msg)
                    error_occurred = True
                    break
                
                # Create action
                action = DiagnosticAction(
                    action_type=action_dict["action_type"],
                    question=action_dict.get("question"),
                    test_name=action_dict.get("test_name"),
                    diagnosis=action_dict.get("diagnosis"),
                )
                
                # Execute action with short action string for logging
                action_str = f"{action.action_type}"
                if action.question:
                    action_str += f"('{action.question[:30]}...')"
                elif action.test_name:
                    action_str += f"('{action.test_name}')"
                elif action.diagnosis:
                    action_str += f"('{action.diagnosis[:40]}...')"
                
                # Take step in environment
                step_result = await env.step(action)
                obs = step_result.observation if hasattr(step_result, 'observation') else step_result
                
                reward = obs.reward or 0.0
                episode_rewards.append(reward)
                
                log_step(step_count, action_str, reward, obs.done, None)
                
                # Store in history
                episode_history.append({
                    "agent_thought": llm_response[:100],
                    "agent_action": action_str,
                    "environment_response": obs.message[:200],
                })
                
            except Exception as e:
                error_msg = str(e)[:100]
                log_step(step_count, "error", 0.0, True, error_msg)
                error_occurred = True
                break
        
        # Get final state
        try:
            state = await env.state()
            final_accuracy = state.final_accuracy if hasattr(state, 'final_accuracy') else 0.0
        except:
            final_accuracy = 0.0
        
        # Calculate results
        total_reward = sum(episode_rewards)
        success = obs.done and final_accuracy > 0.3
        
        log_end(success, step_count, final_accuracy, episode_rewards)
        
        return {
            "task": task_name,
            "success": success,
            "steps_taken": step_count,
            "total_reward": total_reward,
            "episode_rewards": episode_rewards,
            "final_diagnosis_accuracy": final_accuracy,
        }


# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================

async def run_all_tasks() -> Dict:
    """Run all 3 difficulty levels and report overall results."""
    
    if not API_KEY:
        print("ERROR: API key not found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.", flush=True)
        return {}
    
    if not ENV_URL:
        print("ERROR: ENV_URL is not set. Set ENV_URL to the environment WebSocket URL.", flush=True)
        return {}
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )
    
    results = {
        "timestamp": None,
        "model": MODEL_NAME,
        "environment": "medical_diagnostic_env",
        "tasks_completed": 0,
        "task_results": [],
        "overall_score": 0.0,
    }
    
    # Run each task
    for i, (task_name, difficulty) in enumerate(zip(TASK_NAMES, DIFFICULTY_LEVELS)):
        print(f"\n--- Task {i+1}/3: {difficulty} difficulty ---", flush=True)
        
        try:
            result = await run_episode_async(
                client,
                LOCAL_IMAGE_NAME,
                difficulty=difficulty,
                task_name=task_name,
            )
            results["task_results"].append(result)
            results["tasks_completed"] += 1
        except Exception as e:
            print(f"ERROR in task {task_name}: {str(e)}", flush=True)
    
    # Calculate overall score
    if results["tasks_completed"] > 0:
        accuracies = [r["final_diagnosis_accuracy"] for r in results["task_results"]]
        results["overall_score"] = sum(accuracies) / len(accuracies)
    
    # Print summary
    print("\n" + "="*60, flush=True)
    print(f"Baseline Inference Complete", flush=True)
    print(f"Tasks completed: {results['tasks_completed']}/3", flush=True)
    print(f"Overall diagnostic accuracy: {results['overall_score']:.3f}", flush=True)
    print("="*60, flush=True)
    
    return results


def main():
    """Entry point."""
    # Run async loop
    results = asyncio.run(run_all_tasks())


if __name__ == "__main__":
    main()
