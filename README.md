# Medical Diagnostic Environment

The Medical Diagnostic Environment is an OpenEnv-compatible reinforcement learning environment designed for training AI agents in clinical reasoning and medical diagnosis. It simulates realistic patient interactions where agents must ask questions, order tests, and make diagnoses through a structured, multi-step process.

## Project Overview

This repository contains a complete, deployable environment for training and evaluating AI models in medical diagnosis tasks. The environment provides:

- Realistic patient cases with varying difficulty levels
- Step-by-step clinical decision-making process
- Shaped rewards for intermediate actions and final outcomes
- Support for both WebSocket and HTTP-based interactions
- Docker-based deployment options including Hugging Face Spaces

The environment is built using Python, FastAPI, and the OpenEnv framework, making it suitable for integration with various RL training pipelines.

## Key Features

- **Multi-step Episodes**: Agents interact over up to 15 steps per patient case
- **Three Difficulty Tiers**: Easy, medium, and hard cases with increasing complexity
- **Realistic Medical Content**: Based on actual clinical scenarios and diagnostic workflows
- **Shaped Rewards**: Immediate feedback for questions, tests, and final diagnoses
- **Concurrent Sessions**: Supports multiple parallel training sessions
- **Multiple Interfaces**: WebSocket for real-time training, HTTP for testing and evaluation
- **Docker Deployment**: Easy containerization and cloud deployment
- **Validation Tools**: Built-in scripts for testing and deployment verification

## Architecture

The project is structured as follows:

### Core Components

- **`server/environment.py`**: Implements the `MedicalDiagnosticEnvironment` class with reset/step/state logic
- **`server/app.py`**: FastAPI application exposing OpenEnv endpoints
- **`models.py`**: Defines action, observation, and state data structures
- **`client.py`**: WebSocket client for environment interaction
- **`training_wrapper.py`**: Simplified wrapper for training loops

### Data and Logic

- **`server/medical_data.py`**: Contains patient cases, question/test rewards, and diagnostic logic
- **`openenv.yaml`**: Environment metadata and interface specification

### Deployment and Testing

- **`Dockerfile`**: Production container configuration
- **`server/Dockerfile`**: Development container configuration
- **`docker-compose.yml`**: Local development setup
- **`test_deployment.py`**: Deployment validation script
- **`validate.py`**: Local environment testing
- **`tests/`**: Unit tests

## How It Works

### Environment Interface

The environment follows standard RL semantics:

- **Reset**: Initialize a new patient case and return initial observation
- **Step**: Execute an agent action and return next observation with reward
- **State**: Access internal environment state for debugging

### Agent Actions

Agents can perform three types of actions:

1. **Ask Question**: Query the patient about symptoms or history
2. **Order Test**: Request diagnostic test results
3. **Submit Diagnosis**: Provide final diagnosis to end the episode

### Observations

Each step returns a structured observation containing:

- Episode status (done, steps taken)
- Immediate reward
- Human-readable feedback message
- Patient responses to questions
- Test results
- Progress tracking (questions asked, tests completed)
- Revealed patient data

### Reward Structure

- **Question Rewards**: 0.01-0.05 for relevant clinical questions
- **Test Rewards**: 0.05-0.10 for informative diagnostic tests
- **Diagnosis Rewards**: 0.0-1.0 based on accuracy and process quality
- **Penalties**: For irrelevant actions or exceeding step limits

## Installation and Setup

### Requirements

- Python 3.10 or higher
- Docker (recommended for deployment)

### Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd meta_synapse_hackathon
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r server/requirements.txt
   ```

4. Validate the installation:
   ```bash
   python validate.py
   ```

## Running Locally

### Direct Python Execution

Start the server:
```bash
cd server
python app.py
```

Access points:
- Health check: `http://localhost:8000/health`
- API documentation: `http://localhost:8000/docs`
- WebSocket endpoint: `ws://localhost:8000/ws`

### Docker (Development)

Build and run:
```bash
docker build -t medical-diagnostic-env -f server/Dockerfile .
docker run -p 8000:8000 medical-diagnostic-env
```

### Docker Compose

For a complete local setup:
```bash
docker-compose up --build
```

## Using the Client

The `client.py` module provides the `DiagnosticEnv` class for interacting with the environment.

### Asynchronous Usage

```python
import asyncio
from client import DiagnosticEnv
from models import DiagnosticAction

async def main():
    async with DiagnosticEnv(base_url="ws://localhost:8000/ws") as env:
        # Reset for a new case
        obs = await env.reset(difficulty="easy")
        print(f"Initial observation: {obs.message}")
        
        # Take an action
        action = DiagnosticAction(
            action_type="ask_question",
            question="Do you have a fever?"
        )
        result = await env.step(action)
        print(f"Response: {result.observation.message}")
        
        # Continue interacting...

asyncio.run(main())
```

### Synchronous Usage

```python
from client import DiagnosticEnv
from models import DiagnosticAction

with DiagnosticEnv(base_url="ws://localhost:8000/ws").sync() as env:
    obs = env.reset(difficulty="easy")
    print(obs.message)
    
    action = DiagnosticAction(action_type="order_test", test_name="CBC")
    result = env.step(action)
    print(result.observation.message)
```

## Training Integration

The `training_wrapper.py` provides a minimal interface for training loops:

```python
from training_wrapper import TrainingEnv

async with TrainingEnv() as env:
    obs = await env.reset(difficulty="easy")
    
    while not obs.done:
        # Your agent logic here
        action_type = "ask_question"  # or "order_test" or "submit_diagnosis"
        question = "Do you have chest pain?"  # if asking question
        test_name = None  # if ordering test
        diagnosis = None  # if submitting diagnosis
        
        obs = await env.step(
            action_type=action_type,
            question=question,
            test_name=test_name,
            diagnosis=diagnosis
        )
        
        # Update your model with obs.reward, etc.
```

This wrapper is designed to be integrated with your RL training pipeline.

## Deployment

### Docker Deployment

#### Local Docker

```bash
# Build production image
docker build -t medical-diagnostic-env .

# Run container
docker run -p 7860:7860 medical-diagnostic-env
```

#### Docker Compose

```bash
docker-compose up --build
```

### Hugging Face Spaces Deployment

Hugging Face Spaces provides free hosting for machine learning demos and environments.

#### Creating a Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name**: Choose a unique name (e.g., `your-username/medical-diagnostic-env`)
   - **License**: MIT (or your preferred license)
   - **SDK**: Docker
   - **Docker Template**: Blank (we provide our own Dockerfile)
4. Click "Create Space"

#### Uploading Your Code

1. Clone your repository locally if not already done
2. Copy all files from this repository to your Space
3. The Space will automatically detect the `Dockerfile` and start building

#### Accessing Your Deployed Space

Once built, your Space will be available at:
```
https://your-username-medical-diagnostic-env.hf.space
```

Available endpoints:
- Health: `https://your-username-medical-diagnostic-env.hf.space/health`
- API Docs: `https://your-username-medical-diagnostic-env.hf.space/docs`
- WebSocket: `wss://your-username-medical-diagnostic-env.hf.space/ws`

#### Using the Deployed Environment

You can interact with the deployed environment using the same client code, just change the URL:

```python
from client import DiagnosticEnv

# For deployed Space
async with DiagnosticEnv(base_url="wss://your-username-medical-diagnostic-env.hf.space/ws") as env:
    obs = await env.reset(difficulty="medium")
    # ... your training code
```

Or use HTTP endpoints directly:

```python
import requests

base_url = "https://your-username-medical-diagnostic-env.hf.space"

# Reset
response = requests.post(f"{base_url}/reset", json={})
print(response.json())

# Step
action = {"action": {"action_type": "ask_question", "question": "Any pain?"}}
response = requests.post(f"{base_url}/step", json=action)
print(response.json())
```

#### Important Notes for HF Spaces

- The environment maintains state across HTTP requests, so the deployment uses a single Uvicorn worker
- For high-traffic usage, consider upgrading to a paid Space plan
- Spaces have usage limits; monitor your Space's analytics

## Testing and Validation

### Local Validation

Run the comprehensive validation suite:
```bash
python validate.py
```

This checks:
- Package imports
- Environment initialization
- Action processing (questions, tests, diagnoses)
- Reward calculations
- State management

### Unit Tests

Run the test suite:
```bash
python -m pytest tests/
```

### Deployment Testing

Test a deployed instance:
```bash
pip install requests
python test_deployment.py --url https://your-space.hf.space
```

This validates:
- Health endpoint
- API schema
- Reset functionality
- Step execution
- State inspection

## API Reference

### HTTP Endpoints

- `GET /health`: Health check
- `GET /docs`: Interactive API documentation
- `GET /openapi.json`: OpenAPI specification
- `POST /reset`: Reset environment (optional: `{"difficulty": "easy|medium|hard"}`)
- `POST /step`: Execute action (requires: `{"action": DiagnosticAction}`)
- `GET /state`: Get current environment state

### WebSocket Endpoint

- `ws://host:port/ws`: WebSocket connection for session-based interaction

### Data Models

#### DiagnosticAction
```python
{
    "action_type": "ask_question" | "order_test" | "submit_diagnosis",
    "question": str | None,
    "test_name": str | None,
    "diagnosis": str | None
}
```

#### PatientObservation
```python
{
    "done": bool,
    "reward": float | None,
    "message": str,
    "patient_response": dict | None,
    "test_result": dict | None,
    "questions_asked": list[str],
    "tests_completed": list[str],
    "patient_data_revealed": dict,
    "steps_taken": int,
    "max_steps": int
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest

# Format code
black .
ruff --fix .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the validation and testing scripts for examples
