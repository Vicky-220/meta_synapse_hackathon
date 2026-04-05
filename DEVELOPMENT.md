# Development Guide - Medical Diagnostic Environment

This guide covers local development, testing, debugging, and deployment for the Medical Diagnostic Environment.

## Prerequisites

- Python 3.11+
- pip or conda
- Docker (optional, for containerization)
- Git (for version control)

## Local Development Setup

### 1. Clone or Download the Repository

```bash
cd /path/to/medical_diagnostic_env
```

### 2. Create a Virtual Environment

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n medical-env python=3.11
conda activate medical-env
```

### 3. Install Dependencies

```bash
pip install -r server/requirements.txt
pip install pytest pytest-asyncio  # For testing
```

## Running the Server Locally

### Option 1: Using Uvicorn Directly

```bash
cd server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`.

- **Health Check**: `curl http://localhost:8000/health`
- **Docs**: `http://localhost:8000/docs` (OpenAPI/Swagger UI)
- **WebSocket**: `ws://localhost:8000/ws` (for client connections)

### Option 2: Using Docker Compose

```bash
docker-compose up --build
```

The server will be available at `http://localhost:8000`.

### Option 3: Using Python Directly

```bash
python -c "from server.app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

## Testing

### 1. Quick Validation (No Dependencies)

Run the quick validation script to check core functionality:

```bash
python validate.py
```

This checks:
- ✅ Imports work
- ✅ Models can be created
- ✅ Environment initializes
- ✅ All actions (question, test, diagnosis) work
- ✅ Reward functions are correct
- ✅ State property works
- ✅ Concurrent sessions are supported

Expected output:
```
======================================================================
MEDICAL DIAGNOSTIC ENVIRONMENT - VALIDATION SUITE
======================================================================

✅ PASS: Imports
✅ PASS: Model Creation
✅ PASS: Environment Initialization
... (8 more checks)

======================================================================
SUMMARY: 11/11 checks passed
======================================================================
```

### 2. Unit Tests (Requires pytest)

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_environment.py -v

# Run specific test
python -m pytest tests/test_environment.py::TestMedicalDiagnosticEnvironment::test_reset_easy -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=server --cov=models --cov-report=html
```

### 3. Integration Tests (Server Required)

With server running on `localhost:8000`:

```bash
# Test HTTP endpoints
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Test WebSocket client
python client.py  # If you add a main block
```

## Running Baseline Inference

### 1. Set Up OpenAI API

Using official OpenAI API:
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4"
```

Using HuggingFace router:
```bash
export HF_TOKEN="hf_..."
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://api-inference.huggingface.co/models"
```

Using local vLLM:
```bash
export MODEL_NAME="meta-llama/Llama-2-7b-chat"
export API_BASE_URL="http://localhost:8000/v1"
```

### 2. Start Server (Terminal 1)

```bash
cd server
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Run Inference (Terminal 2)

```bash
python inference.py
```

Expected output:
```
[START] task=dataset_case env=medical-diagnostic model=gpt-4
[STEP] step=1 action=ask_question reward=0.05 done=False error=None
[STEP] step=2 action=order_test reward=0.10 done=False error=None
[STEP] step=3 action=submit_diagnosis reward=1.0 done=True error=None
[END] success=True steps=3 score=0.87 rewards=[0.05, 0.10, 1.0]
...
```

### 4. Customize Inference

Edit `inference.py` to:
- Change `create_system_prompt()` for different medical instructions
- Modify `extract_action_from_response()` for different parsing logic
- Use `run_all_tasks()` vs `specific_task_inference()` for different evaluation modes

## Debugging

### 1. Enable Verbose Logging

```python
# In your code
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment variable
export PYTHONVERBOSE=2
```

### 2. Inspect Hidden State

During an episode, the environment's hidden state is available:

```python
from server.environment import MedicalDiagnosticEnvironment

env = MedicalDiagnosticEnvironment()
env.reset(difficulty="easy")

# Access hidden true diagnosis
print(f"True diagnosis: {env.state.true_diagnosis}")
print(f"Patient ID: {env.state.patient_id}")

# After step
result = env.step(action)
print(f"Step count: {env.state.step_count}")
print(f"Current accuracy: {env.current_accuracy}")
```

### 3. Python Debugger

```python
import pdb

# Set breakpoint
pdb.set_trace()

# Or use built-in breakpoint() (Python 3.7+)
breakpoint()
```

### 4. Common Issues

**Issue**: "Connection refused" when running inference
- **Solution**: Make sure server is running on port 8000. Check with `curl http://localhost:8000/health`

**Issue**: ImportError for `openenv`
- **Solution**: Install with `pip install openenv-core==0.2.3`

**Issue**: DiagnosticAction validation fails
- **Solution**: Check that action_type is one of: "ask_question", "order_test", "submit_diagnosis"

**Issue**: Episode ends immediately
- **Solution**: Make sure you're not submitting a diagnosis on the first step. Ask questions or order tests first.

## Code Structure

```
medical_diagnostic_env/
├── __init__.py                 # Package exports
├── models.py                    # Pydantic models (Action, Observation, State)
├── client.py                    # Async/sync client wrappers
├── inference.py                 # OpenAI baseline inference
├── validate.py                  # Quick validation script
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI application
│   ├── environment.py           # Core environment logic
│   ├── medical_data.py          # Medical knowledge base
│   ├── requirements.txt
│   └── Dockerfile
├── tests/
│   ├── __init__.py
│   └── test_environment.py      # Unit tests
├── docker-compose.yml           # Local development
├── openenv.yaml                 # OpenEnv spec
├── pyproject.toml              # Package config
└── README.md                    # User documentation
```

## Contributing

### Code Style

Use black for formatting:
```bash
pip install black
black --line-length 100 *.py server/*.py
```

Use ruff for linting:
```bash
pip install ruff
ruff check --fix *.py server/*.py
```

### Type Checking

```bash
pip install mypy
mypy --strict *.py server/*.py
```

### Before Committing

```bash
python validate.py
python -m pytest tests/ -v
black --check *.py server/*.py
ruff check *.py server/*.py
```

## Deployment

### Docker Build

```bash
docker build -t medical-diagnostic-env -f server/Dockerfile .
docker run -p 8000:8000 medical-diagnostic-env
```

### HuggingFace Spaces

1. Create a new Space: https://huggingface.co/new-space
2. Select "Docker" as the Space type
3. Push your code:
```bash
git clone https://huggingface.co/spaces/username/medical-diagnostic-env
cd medical-diagnostic-env
git add .
git commit -m "Initial commit"
git push
```

### Server Monitoring

Check health status:
```bash
curl http://localhost:8000/health
```

View OpenAPI docs:
```bash
open http://localhost:8000/docs  # Or visit in browser
```

Monitor concurrent sessions:
```bash
# The /state endpoint shows current session state
curl http://localhost:8000/state
```

## Performance Tips

1. **Use async client** for better throughput when running multiple episodes
2. **Batch API calls** if using external LLM APIs
3. **Cache medical data** if running many episodes (already done)
4. **Use uvicorn workers** (default: 4) for concurrent training
5. **Profile with cProfile** for bottleneck analysis

```bash
python -m cProfile -s cumtime inference.py > profile.txt
```

## Resources

- OpenEnv Documentation: https://openenv.io
- FastAPI Documentation: https://fastapi.tiangolo.com
- Pydantic Documentation: https://docs.pydantic.dev
- OpenAI API: https://platform.openai.com/docs
- HuggingFace Inference API: https://huggingface.co/inference-api

## Support

For issues or questions:
1. Check this development guide
2. Run `python validate.py` to diagnose problems
3. Check the main README.md for architecture questions
4. Review test cases in `tests/` for usage examples
