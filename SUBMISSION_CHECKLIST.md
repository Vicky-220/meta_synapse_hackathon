# Hackathon Submission Checklist

Use this checklist to ensure your Medical Diagnostic Environment meets all hackathon requirements before submission.

## Project Scope & Requirements

- [ ] **Environment is based on real-world domain** (Medical Diagnosis)
  - Utility Score: >30% (Medical diagnosis is high-utility domain)
  
- [ ] **OpenEnv spec compliance**
  - [ ] Inherits from `openenv.core.env_server.Environment`
  - [ ] Implements `reset(difficulty)` method
  - [ ] Implements `step(action)` method
  - [ ] Implements `state` property
  - [ ] Defines Action, Observation, State with type hints
  - [ ] Creates FastAPI app with `create_fastapi_app()`

- [ ] **Three difficulty levels with varying complexity**
  - [ ] Easy (3 cases: Flu, UTI, Migraine) - 80% expected accuracy
  - [ ] Medium (2 cases: Pneumonia vs Bronchitis, Appendicitis vs IBS) - 60% expected accuracy
  - [ ] Hard (2 cases: SLE, Multiple Myeloma) - 30% expected accuracy

- [ ] **Deterministic graders for each task**
  - [ ] Graders return float scores in range [0.0, 1.0]
  - [ ] Grading logic based on medical knowledge (not random)
  - [ ] Different grading logic for each difficulty level
  - [ ] Handling of acceptable differentials (e.g., "Pneumonia" vs "Bacterial Pneumonia")

- [ ] **Trajectory-based rewards (not sparse)**
  - [ ] +0.05 for clinically relevant questions
  - [ ] +0.10 for informative tests
  - [ ] +0.30-1.0 for diagnosis accuracy
  - [ ] Rewards guide agent toward good heuristics
  - [ ] Total episode reward is sum of trajectory rewards

## Code Quality & Structure

- [ ] **Project structure is clean and organized**
  - [ ] `models.py` - Data contracts (Action, Observation, State)
  - [ ] `server/environment.py` - Core environment logic
  - [ ] `server/medical_data.py` - Knowledge base and graders
  - [ ] `server/app.py` - FastAPI server setup
  - [ ] `client.py` - Training client (async + sync)
  - [ ] `inference.py` - Baseline agent implementation

- [ ] **Type safety with Pydantic**
  - [ ] All data classes inherit from appropriate Pydantic models
  - [ ] Type hints on all functions
  - [ ] Validation in data class definitions
  - [ ] No `Any` types (minimize use)

- [ ] **Code is well-documented**
  - [ ] Docstrings on all public methods
  - [ ] Comments explaining key logic
  - [ ] README.md with full documentation
  - [ ] DEVELOPMENT.md with setup and testing instructions
  - [ ] Type hints serve as inline documentation

- [ ] **No hard-coded values in critical paths**
  - [ ] Difficulty levels configurable
  - [ ] Case selection deterministic but flexible
  - [ ] Reward thresholds documented

## Deployment Readiness

- [ ] **Docker support**
  - [ ] `server/Dockerfile` exists
  - [ ] Dockerfile uses appropriate base image (python:3.11-slim)
  - [ ] All dependencies installed in Dockerfile
  - [ ] Health check configured
  - [ ] Server runs on port 8000
  - [ ] Build succeeds: `docker build -t medical-diagnostic-env -f server/Dockerfile .`
  - [ ] Container runs: `docker run -p 8000:8000 medical-diagnostic-env`

- [ ] **Docker Compose for local development**
  - [ ] `docker-compose.yml` includes service definition
  - [ ] Ports properly mapped
  - [ ] Health checks configured
  - [ ] Environment variables documented

- [ ] **OpenEnv manifest (`openenv.yaml`)**
  - [ ] Metadata: name, version, description, keywords
  - [ ] Interface specification for Action and Observation
  - [ ] Task definitions with difficulty levels
  - [ ] Performance metrics and expectations
  - [ ] Deployment configuration
  - [ ] `openenv validate openenv.yaml` passes (if tool available)

- [ ] **HuggingFace Spaces ready**
  - [ ] Dockerfile can be used as-is for Spaces
  - [ ] Environment variables clearly documented
  - [ ] API keys not embedded in code
  - [ ] Server health check endpoint exposed (/health)

## Testing & Validation

- [ ] **Local validation passes**
  - [ ] `python validate.py` shows 11/11 checks passed
  - [ ] No import errors
  - [ ] Environment initializes correctly
  - [ ] All actions work (question, test, diagnosis)
  - [ ] Reward functions return valid numbers

- [ ] **Unit tests exist and pass**
  - [ ] `pytest tests/` runs without errors
  - [ ] Tests cover all three actions
  - [ ] Tests verify all three difficulty levels
  - [ ] Episode summaries are generated correctly

- [ ] **Server starts without errors**
  - [ ] `python -m uvicorn server.app:app` starts cleanly
  - [ ] Health endpoint responds: `curl http://localhost:8000/health`
  - [ ] OpenAPI docs available: `http://localhost:8000/docs`
  - [ ] No warnings about missing dependencies

## Inference Baseline

- [ ] **Baseline inference script works**
  - [ ] `inference.py` can connect to server
  - [ ] Runs all three difficulty levels successfully
  - [ ] Produces output in exact hackathon format:
    ```
    [START] task=... env=... model=...
    [STEP] step=... action=... reward=... done=... error=...
    [END] success=... steps=... score=... rewards=...
    ```
  - [ ] Score calculations are correct
  - [ ] Error handling is robust

- [ ] **Baseline performance is documented**
  - [ ] Easy task: Expected accuracy 80%, achieved: __%
  - [ ] Medium task: Expected accuracy 60%, achieved: __%
  - [ ] Hard task: Expected accuracy 30%, achieved: __%
  - [ ] Average steps per episode documented

- [ ] **Different LLM backends supported**
  - [ ] Works with OpenAI API
  - [ ] Works with HuggingFace inference API (fallback ready)
  - [ ] API key handling is secure (environment variables)
  - [ ] Error messages are helpful

## Documentation

- [ ] **README.md is comprehensive**
  - [ ] Quick start guide
  - [ ] Project overview and architecture
  - [ ] Installation instructions
  - [ ] Three ways to run (local Python, Docker, Docker Compose)
  - [ ] API usage examples
  - [ ] Reward structure explanation
  - [ ] Client usage examples (async and sync)
  - [ ] Patient cases documentation
  - [ ] Deployment to HF Spaces
  - [ ] Evaluation criteria and scoring
  - [ ] Links to external resources

- [ ] **DEVELOPMENT.md exists**
  - [ ] Local development setup
  - [ ] Testing instructions
  - [ ] Debugging tips
  - [ ] Code structure explanation
  - [ ] Performance optimization tips

- [ ] **Inline code documentation**
  - [ ] Complex algorithms have comments
  - [ ] Medical knowledge base explains the logic
  - [ ] Reward calculations are explained
  - [ ] Edge cases are documented

## Performance & Optimization

- [ ] **Environment runs efficiently**
  - [ ] Reset time < 1 second per episode
  - [ ] Step time < 500ms (excluding LLM inference)
  - [ ] Memory usage is reasonable (<500MB for server)
  - [ ] Supports 4 concurrent episodes

- [ ] **Scalability considerations**
  - [ ] `SUPPORTS_CONCURRENT_SESSIONS = True`
  - [ ] Server configured for multiple workers (4 default)
  - [ ] Medical data is cached (not re-generated per step)
  - [ ] No blocking operations in environment

## Security & Robustness

- [ ] **No secrets in code**
  - [ ] API keys use environment variables
  - [ ] No hardcoded tokens or passwords
  - [ ] .gitignore excludes sensitive files

- [ ] **Input validation**
  - [ ] Action validation prevents invalid actions
  - [ ] Difficulty level validation
  - [ ] Diagnosis strings are validated
  - [ ] Test names are validated against known tests

- [ ] **Error handling**
  - [ ] Invalid actions produce helpful error messages
  - [ ] Network errors are handled gracefully
  - [ ] Non-existent cases are caught
  - [ ] Timeout handling for LLM calls

- [ ] **Rate limiting ready**
  - [ ] Code can be extended with rate limiting
  - [ ] API calls are batched appropriately
  - [ ] No unnecessary API calls

## Final Checks

- [ ] **All files present and accounted for**
  ```
  ✓ models.py
  ✓ client.py
  ✓ inference.py
  ✓ validate.py
  ✓ server/environment.py
  ✓ server/medical_data.py
  ✓ server/app.py
  ✓ server/requirements.txt
  ✓ server/Dockerfile
  ✓ docker-compose.yml
  ✓ openenv.yaml
  ✓ pyproject.toml
  ✓ README.md
  ✓ DEVELOPMENT.md
  ✓ .gitignore
  ```

- [ ] **Repository is clean**
  - [ ] No debug code left
  - [ ] No TODOs in critical paths
  - [ ] No unused imports
  - [ ] Code is formatted (black)
  - [ ] No linting errors (ruff)

- [ ] **Git is ready**
  - [ ] Commit message is descriptive
  - [ ] All changes are committed
  - [ ] No uncommitted changes
  - [ ] Repository history is clean

## Submission Preparation

- [ ] **Screenshot baseline results**
  - [ ] Run `python inference.py` with a working LLM model
  - [ ] Screenshot the [START]/[END] output
  - [ ] Record accuracy for each difficulty level
  - [ ] Note the average steps taken

- [ ] **Prepare submission materials**
  - [ ] Copy of baseline inference output
  - [ ] Link to code repository (GitHub/GitLab)
  - [ ] Link to deployed HF Space (if deployed)
  - [ ] Brief written summary of the environment

- [ ] **Test submission on clean machine**
  - [ ] Clone fresh copy of repository
  - [ ] Follow README setup instructions
  - [ ] Verify local validation passes
  - [ ] Verify Docker build succeeds
  - [ ] Verify inference runs without errors

## Deadline Tracking

- [ ] Submission deadline: **[Your Deadline Here]**
- [ ] Time buffer: Please complete by 24 hours before deadline
- [ ] Final review: Please complete by 12 hours before deadline

---

## Quick Reference Commands

```bash
# Validate locally
python validate.py

# Run unit tests
python -m pytest tests/ -v

# Start server
cd server && python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Run inference
python inference.py

# Build Docker image
docker build -t medical-diagnostic-env -f server/Dockerfile .

# Run Docker container
docker run -p 8000:8000 medical-diagnostic-env

# Format code
black --line-length 100 *.py server/*.py

# Check code
ruff check --fix *.py server/*.py

# Check types
mypy --strict *.py server/*.py

# Check Docker build
docker build --no-cache -f server/Dockerfile .
```

## Scoring Breakdown (For Reference)

| Component | Points | Status |
|---|---|---|
| Real-world utility | 15 | Domain: Medical Diagnosis (High utility) |
| 3 difficulty levels | 15 | Easy, Medium, Hard cases defined |
| Deterministic graders | 15 | Graders based on medical knowledge |
| Trajectory rewards | 15 | 0.05, 0.10, 0.30-1.0 structure |
| OpenEnv compliance | 10 | Inherits, implements interface |
| Code quality | 10 | Type-safe, documented, tested |
| Deployment ready | 10 | Docker, docker-compose, Spaces ready |
| **Total** | **90** | / 90 (+ up to 10 bonus) |

---

**Last Updated**: [Current Date]  
**Project**: Medical Diagnostic Environment for Meta PyTorch OpenEnv Hackathon  
**Team**: SYNAPSE
