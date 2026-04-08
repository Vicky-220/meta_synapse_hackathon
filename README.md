# Medical Diagnostic Environment

# Medical Diagnostic Environment

This repository provides a working OpenEnv-compatible medical diagnosis environment.
It is built so external RL agents and model-based trainers can connect, take actions, and learn from step-by-step clinical interaction.

## What this project includes

- `server/environment.py`: core environment logic for patient cases, rewards, and episode flow
- `server/app.py`: FastAPI app exposing OpenEnv endpoints and WebSocket access
- `client.py`: Python client for WebSocket-based environment interaction
- `training_wrapper.py`: minimal training wrapper that makes the environment easier to use in a trainer loop
- `test_deployment.py`: runtime checker for local or deployed environment endpoints
- `validate.py`: local self-check script for package imports, environment behavior, and core actions
- `models.py`: shared action, observation, and state classes
- `server/medical_data.py`: patient case data, questions, tests, and reward calculations
- `openenv.yaml`: environment metadata and interface definition for OpenEnv
- `docker-compose.yml`: local compose setup for development and testing

## Why this environment matters

This is not just a demo. It is a compact clinical reasoning environment with:

- 3 difficulty tiers: easy, medium, hard
- 6 realistic patient cases
- multi-step reasoning over questions, tests, and diagnosis
- shaped rewards for useful questions and tests plus final diagnosis accuracy
- episode-level state and environment compatibility for training loops

It is intended to be used as a training environment backend, not as a complete RL agent.

## How the environment works

The environment exposes a standard OpenEnv API:

- `/reset`: start a new case and receive the initial observation
- `/step`: execute one action and receive the next observation
- `/state`: inspect the current hidden state for debugging
- `/health`: confirm the server is alive
- WebSocket endpoint at `/ws` for session-based agent training

### Agent actions

Agents send one of these action types:

- `ask_question` — ask the patient a clinical question
- `order_test` — request a diagnostic test result
- `submit_diagnosis` — provide a final diagnosis

### Observation fields

Each step returns:

- `done`: whether the episode has ended
- `reward`: immediate reward for the action
- `message`: human-readable feedback
- `patient_response`: answer to a clinical question
- `test_result`: results of ordered tests
- `questions_asked`: questions asked so far
- `tests_completed`: tests ordered so far
- `patient_data_revealed`: what the agent has learned
- `steps_taken` and `max_steps`

## Setup

Requirements:

- Python 3.10+
- Docker (optional but recommended for deployment)

### Install locally

```bash
git clone <repository-url>
cd meta_synapse_hackathon
python -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
```

### Validate locally

```bash
python validate.py
```

That script checks imports, reset/step behavior, reward calculations, and environment state.

## Running locally

### Run server directly

```bash
cd server
python app.py
```

Then open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`
- `ws://localhost:8000/ws`

### Run with Docker locally

```bash
docker build -t medical-diagnostic-env -f server/Dockerfile .
docker run -p 8000:8000 medical-diagnostic-env
```

### Run with Docker Compose

```bash
docker-compose up --build
```

Access the same endpoints at `localhost:8000`.

## Client usage

The `client.py` module provides a `DiagnosticEnv` class with async and sync usage.

### Async example

```python
import asyncio
from client import DiagnosticEnv
from models import DiagnosticAction

async def main():
    async with DiagnosticEnv(base_url="ws://localhost:8000/ws") as env:
        obs = await env.reset(difficulty="easy")
        print(obs.message)
        result = await env.step(DiagnosticAction(action_type="ask_question", question="Do you have a fever?"))
        print(result.message)

asyncio.run(main())
```

### Sync example

```python
from client import DiagnosticEnv
from models import DiagnosticAction

with DiagnosticEnv(base_url="ws://localhost:8000/ws").sync() as env:
    obs = env.reset(difficulty="easy")
    print(obs.message)
    result = env.step(DiagnosticAction(action_type="order_test", test_name="CBC"))
    print(result.observation.message)
```

## Training wrapper

`training_wrapper.py` provides a thin wrapper to make the environment easier to consume in an async training loop.

It is intentionally minimal and meant to be paired with your own RL algorithm.

```bash
python training_wrapper.py
```

## Docker and Hugging Face deployment

This repo includes:

- `server/Dockerfile`: local development container using port `8000`
- `Dockerfile`: root-level container using port `7860` for production deployment

### Deploy to Hugging Face Spaces

1. Create a new Space using Docker.
2. Upload this repository.
3. The Space should build automatically.
4. Verify the deployed runtime.

The included `test_deployment.py` is the recommended way to confirm the deployed Space is healthy.

> Important: because the environment uses stateful HTTP behavior, the deployed container should run a single Uvicorn worker.

## Deployment testing

Install the runtime checker:

```bash
pip install requests
```

Run the checker locally:

```bash
python test_deployment.py
```

Run the checker against a deployed Space:

```bash
OPENENV_BASE_URL="https://your-space-domain.hf.space" python test_deployment.py
```

That script validates `/health`, `/openapi.json`, `/reset`, `/step`, and `/state`.

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Run the environment validation suite:

```bash
python validate.py
```

## Notes for judges

This repository is ready to serve as an environment backend for training.
Judges can run their own policy optimization or GPRO-style training algorithm by connecting to the environment and exchanging actions and observations.

### What judges will need to provide

- their own training/agent loop
- an action policy that sends `DiagnosticAction` objects
- a mechanism to handle observations and update model weights

### What this repo provides

- environment logic
- server endpoints for HTTP and WebSocket interaction
- a reusable Python client
- deployment and validation helpers

## Is this hackathon-ready?

Yes — the environment in this repo is ready to be submitted as the environment component.
It is not a full RL agent, but it is ready to host training and evaluation.

If the hackathon judges want to run their own training algorithms, they can use this repo purely as the environment and connect to it via:

- `ws://<host>/ws` for WebSocket sessions
- `/reset`, `/step`, `/state` for standard environment loops

That means the repo is suitable as an environment service, but the judges will still need to provide the actual training algorithm or agent logic on their side.

## Project structure

```
├── client.py
├── docker-compose.yml
├── Dockerfile
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── validate.py
├── test_deployment.py
├── training_wrapper.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── Dockerfile
│   ├── environment.py
│   ├── medical_data.py
│   └── requirements.txt
└── tests/
    └── test_environment.py
```

## License

MIT License — see `LICENSE`.
