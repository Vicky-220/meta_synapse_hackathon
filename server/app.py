"""
server/app.py — FastAPI server for the Medical Diagnostic Environment.

This exposes the environment over WebSocket and HTTP using OpenEnv's built-in
create_fastapi_app helper. One line of meaningful code!

The helper automatically creates:
- /ws endpoint for WebSocket connections (stateful, for training)
- /reset, /step, /state endpoints (stateless, for testing)
- /health endpoint (for Docker health checks)
- /docs endpoint (auto-generated OpenAPI documentation)
"""

from openenv.core.env_server import create_fastapi_app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DiagnosticAction, PatientObservation
from .environment import MedicalDiagnosticEnvironment


# Create FastAPI app with all endpoints
app = create_fastapi_app(
    MedicalDiagnosticEnvironment,  # Pass the class, not an instance
    DiagnosticAction,
    PatientObservation,
    max_concurrent_envs=100,  # Support up to 100 parallel training sessions
)

# Optional: Add custom middleware or endpoints here if needed
# (Most common use cases are already handled by create_fastapi_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
    )
