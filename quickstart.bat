@echo off
REM Quick start script for Medical Diagnostic Environment (Windows)
REM Usage: quickstart.bat [server|test|docker|inference]

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
echo Project directory: %PROJECT_DIR%

REM Colors are limited on Windows, using text-based indicators
set SUCCESS=[OK]
set ERROR=[ERROR]
set INFO=[INFO]

if "%~1"=="" (
    call :show_help
    exit /b 1
)

if /i "%~1"=="server" (
    call :run_server
) else if /i "%~1"=="test" (
    call :run_tests
) else if /i "%~1"=="validate" (
    call :run_validation
) else if /i "%~1"=="inference" (
    call :run_inference
) else if /i "%~1"=="docker" (
    call :build_docker
) else if /i "%~1"=="docker-compose" (
    call :run_docker_compose
) else if /i "%~1"=="help" (
    call :show_help
) else if /i "%~1"=="--help" (
    call :show_help
) else if /i "%~1"=="-h" (
    call :show_help
) else (
    echo %ERROR% Unknown command: %~1
    echo.
    call :show_help
    exit /b 1
)

goto :end

REM ===== FUNCTIONS =====

:setup_venv
echo.
echo ========================================
echo Setting up virtual environment
echo ========================================
if exist "%PROJECT_DIR%venv" (
    echo %INFO% Virtual environment already exists
) else (
    echo %INFO% Creating virtual environment...
    python -m venv "%PROJECT_DIR%venv"
    if errorlevel 1 (
        python3 -m venv "%PROJECT_DIR%venv"
    )
    echo %SUCCESS% Virtual environment created
)

call "%PROJECT_DIR%venv\Scripts\activate.bat"
echo %SUCCESS% Virtual environment activated

echo %INFO% Installing dependencies...
pip install -q --upgrade pip
pip install -q -r "%PROJECT_DIR%server\requirements.txt"
echo %SUCCESS% Dependencies installed
exit /b 0

:run_validation
echo.
echo ========================================
echo Running validation
echo ========================================
if not exist "%PROJECT_DIR%validate.py" (
    echo %ERROR% validate.py not found
    exit /b 1
)
python "%PROJECT_DIR%validate.py"
exit /b 0

:run_server
call :setup_venv
echo.
echo ========================================
echo Starting server
echo ========================================
cd /d "%PROJECT_DIR%server"
echo %INFO% Server starting on http://localhost:8000
echo %INFO% OpenAPI docs: http://localhost:8000/docs
echo %INFO% Press Ctrl+C to stop
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
exit /b 0

:run_tests
call :setup_venv
echo.
echo ========================================
echo Running tests
echo ========================================
echo %INFO% Installing pytest...
pip install -q pytest pytest-asyncio
python -m pytest "%PROJECT_DIR%tests\" -v
exit /b 0

:run_inference
call :setup_venv
echo.
echo ========================================
echo Running inference baseline
echo ========================================

REM Check if server is running
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Server is not running on http://localhost:8000
    echo %INFO% Please start the server first with: quickstart.bat server
    exit /b 1
)

REM Check API keys
if not defined OPENAI_API_KEY (
    if not defined HF_TOKEN (
        echo %ERROR% Neither OPENAI_API_KEY nor HF_TOKEN is set
        echo %INFO% Set one of these environment variables:
        echo %INFO%   set OPENAI_API_KEY=sk-...
        echo %INFO%   set HF_TOKEN=hf_...
        exit /b 1
    )
)

echo %INFO% Running inference...
cd /d "%PROJECT_DIR%"
python inference.py
exit /b 0

:build_docker
echo.
echo ========================================
echo Building Docker image
echo ========================================
docker --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker is not installed
    exit /b 1
)

echo %INFO% Building image: medical-diagnostic-env
docker build -t medical-diagnostic-env -f "%PROJECT_DIR%server\Dockerfile" "%PROJECT_DIR%"
if errorlevel 1 (
    echo %ERROR% Docker build failed
    exit /b 1
)
echo %SUCCESS% Docker image built successfully
echo %INFO% To run the container:
echo %INFO%   docker run -p 8000:8000 medical-diagnostic-env
exit /b 0

:run_docker_compose
echo.
echo ========================================
echo Starting with Docker Compose
echo ========================================
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker Compose is not installed
    exit /b 1
)

cd /d "%PROJECT_DIR%"
docker-compose up --build
exit /b 0

:show_help
echo.
echo Medical Diagnostic Environment - Quick Start
echo.
echo Usage: quickstart.bat [COMMAND]
echo.
echo Commands:
echo   server        Start development server (auto-activates venv)
echo   test          Run unit tests
echo   validate      Run quick validation (no dependencies)
echo   inference     Run baseline inference (requires server running)
echo   docker        Build Docker image
echo   docker-compose Start with Docker Compose
echo   help          Show this help message
echo.
echo Examples:
echo   quickstart.bat server
echo   quickstart.bat test
echo   quickstart.bat inference (after starting server)
echo.
echo Environment Setup:
echo   For inference with OpenAI:
echo     set OPENAI_API_KEY=sk-...
echo     set MODEL_NAME=gpt-4
echo.
echo   For inference with HuggingFace:
echo     set HF_TOKEN=hf_...
echo     set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
echo.
exit /b 0

:end
endlocal
