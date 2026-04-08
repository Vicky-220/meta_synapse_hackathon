# Medical Diagnostic Environment

A lightweight OpenEnv environment for training agents to diagnose patients using clinical reasoning. The agent interacts through a turn-based dialog, orders tests, and submits a final diagnosis.

## What this project does

The environment simulates a real medical workflow:
- Present a patient with symptoms and context
- Let the agent ask clinical questions
- Allow the agent to order diagnostic tests
- Score the agent on diagnosis accuracy and process quality

It is designed to be used for training or evaluation with reinforcement learning systems.

## Why this environment is useful

This is not a toy problem. It is a small clinical reasoning task with:
- real clinical cases and realistic feedback
- multi-step decisions
- partial reward signals for progress
- a clear end goal: accurate final diagnosis

## Tasks included

There are three difficulty tiers built into the environment:

### Easy
- Seasonal Influenza
- Urinary Tract Infection

### Medium
- Community-Acquired Pneumonia
- Acute Appendicitis

### Hard
- Infective Endocarditis
- Bacterial Meningitis

Each case is graded from 0.0 to 1.0 based on the agent's final diagnosis and stepwise decisions.

## Action and observation interface

### Actions
The agent sends one of three actions:

```python
class DiagnosticAction(Action):
    action_type: str  # ask_question | order_test | submit_diagnosis
    question: Optional[str] = None
    test_name: Optional[str] = None
    diagnosis: Optional[str] = None
```

### Observations
Each step returns a structured observation:

```python
class PatientObservation(Observation):
    done: bool
    reward: Optional[float]
    message: str
    patient_response: Optional[Dict]
    test_result: Optional[Dict]
    questions_asked: List[str]
    tests_completed: List[str]
    patient_data_revealed: Dict
    steps_taken: int
    max_steps: int
```

## Setup

### Requirements
- Python 3.10+
- Docker for containerized deployment

### Local setup

```bash
git clone <repository-url>
cd meta_synapse_hackathon
python -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
```

### Run validation

```bash
python validate.py
```

## Running the environment

### Start the server

```bash
cd server
python app.py
```

Then the environment is available at:
- WebSocket: `ws://localhost:8000/ws`
- Health: `http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`

### Use the client

```python
from client import DiagnosticEnv

async with DiagnosticEnv(base_url="ws://localhost:8000/ws") as env:
    obs = await env.reset(difficulty="easy")
    print(obs.message)
```

## Training-ready wrapper

A simple, training-ready wrapper is available in `training_wrapper.py`. It provides a minimal async interface for use in training loops.

```bash
python training_wrapper.py
```

Use it in your own code like this:

```python
from training_wrapper import TrainingEnv

async with TrainingEnv() as env:
    obs = await env.reset(difficulty="easy")
    step = await env.step(action_type="ask_question", question="Do you have a fever?")
```

## Baseline inference

Set the required environment variables then run the baseline script:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-huggingface-token"
export ENV_URL="ws://localhost:8000/ws"
python inference.py
```

## Optional dataset support

The environment always includes core static cases and can optionally load Hugging Face datasets when enabled.

To use Hugging Face dataset generation:

```bash
export OPENENV_USE_HF_DATASETS=true
export OPENENV_DATASET_SEED=42
```

If dataset loading is disabled or unavailable, the environment still works with the built-in cases.

## Docker deployment

### Build locally

```bash
docker build -t medical-diagnostic-env ./server
```

### Run locally

```bash
docker run -p 8000:8000 medical-diagnostic-env
```

### Deploy to Hugging Face Spaces

1. Create a new Space using Docker.
2. Upload the repository files.
3. The Space should build and expose the server automatically.

## Notes for judges and trainers

- The environment exposes standard reset/step/state semantics.
- It supports concurrent sessions and WebSocket interaction.
- The training wrapper is intentionally minimal so any agent loop can be added on top.

## Project structure

```
├── models.py
├── client.py
├── training_wrapper.py
├── inference.py
├── validate.py
├── openenv.yaml
├── server/
│   ├── app.py
│   ├── environment.py
│   ├── medical_data.py
│   ├── requirements.txt
│   └── Dockerfile
└── tests/
    └── test_environment.py
```

## Testing

```bash
python -m pytest tests/
python validate.py
```

## License

MIT License - see LICENSE file for details.
