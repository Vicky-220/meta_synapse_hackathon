#!/bin/bash
# Quick start script for Medical Diagnostic Environment
# Usage: bash quickstart.sh [server|test|docker|inference]

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

function setup_venv() {
    print_header "Setting up virtual environment"
    
    if [ -d "$PROJECT_DIR/venv" ]; then
        print_info "Virtual environment already exists"
    else
        print_info "Creating virtual environment..."
        python3.11 -m venv "$PROJECT_DIR/venv" || python3 -m venv "$PROJECT_DIR/venv"
        print_success "Virtual environment created"
    fi
    
    # Activate venv
    source "$PROJECT_DIR/venv/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1
    
    # Install dependencies
    print_info "Installing dependencies..."
    pip install -r "$PROJECT_DIR/server/requirements.txt" > /dev/null 2>&1
    print_success "Dependencies installed"
}

function run_validation() {
    print_header "Running validation"
    
    if [ ! -f "$PROJECT_DIR/validate.py" ]; then
        print_error "validate.py not found"
        exit 1
    fi
    
    python "$PROJECT_DIR/validate.py"
}

function run_server() {
    print_header "Starting server"
    
    setup_venv
    
    cd "$PROJECT_DIR/server"
    print_info "Server starting on http://localhost:8000"
    print_info "OpenAPI docs: http://localhost:8000/docs"
    print_info "Press Ctrl+C to stop"
    
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
}

function run_tests() {
    print_header "Running tests"
    
    setup_venv
    
    # Install pytest if not already installed
    pip install pytest pytest-asyncio > /dev/null 2>&1
    
    python -m pytest "$PROJECT_DIR/tests/" -v
}

function run_inference() {
    print_header "Running inference baseline"
    
    setup_venv
    
    # Check if server is running
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_error "Server is not running on http://localhost:8000"
        print_info "Please start the server first with: bash quickstart.sh server"
        exit 1
    fi
    
    print_info "Checking API keys..."
    
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$HF_TOKEN" ]; then
        print_error "Neither OPENAI_API_KEY nor HF_TOKEN is set"
        print_info "Set one of these environment variables:"
        print_info "  export OPENAI_API_KEY=\"sk-...\""
        print_info "  export HF_TOKEN=\"hf_...\""
        exit 1
    fi
    
    print_info "Running inference..."
    cd "$PROJECT_DIR"
    python inference.py
}

function build_docker() {
    print_header "Building Docker image"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    print_info "Building image: medical-diagnostic-env"
    docker build -t medical-diagnostic-env -f "$PROJECT_DIR/server/Dockerfile" "$PROJECT_DIR"
    print_success "Docker image built successfully"
    
    print_info "To run the container:"
    print_info "  docker run -p 8000:8000 medical-diagnostic-env"
}

function run_docker_compose() {
    print_header "Starting with Docker Compose"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    docker-compose up --build
}

function show_help() {
    cat << EOF
${BLUE}Medical Diagnostic Environment - Quick Start${NC}

Usage: bash quickstart.sh [COMMAND]

Commands:
  server       Start development server (auto-activates venv)
  test         Run unit tests
  validate     Run quick validation (no dependencies)
  inference    Run baseline inference (requires server running)
  docker       Build Docker image
  docker-compose Start with Docker Compose
  help         Show this help message

Examples:
  bash quickstart.sh server
  bash quickstart.sh test
  bash quickstart.sh inference  # After starting server

Environment Setup:
  For inference with OpenAI:
    export OPENAI_API_KEY="sk-..."
    export MODEL_NAME="gpt-4"

  For inference with HuggingFace:
    export HF_TOKEN="hf_..."
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

EOF
}

# Main script
case "${1:-}" in
    server)
        run_server
        ;;
    test)
        run_tests
        ;;
    validate)
        run_validation
        ;;
    inference)
        run_inference
        ;;
    docker)
        build_docker
        ;;
    docker-compose)
        run_docker_compose
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac
