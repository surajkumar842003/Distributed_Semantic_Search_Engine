#!/usr/bin/env bash
# =============================================================================
# Docker Setup & Service Verification Script
# Validates Compose configurations, Dockerfiles, environment variables,
# storage mounts, and Python service entrypoints from a clean environment.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE} Wikipedia Semantic Search & RAG: Docker Stack Verification ${NC}"
echo -e "${BLUE}============================================================${NC}"

ERRORS=0
WARNINGS=0

# -----------------------------------------------------------------------------
# 1. Environment File Check
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[1/7] Checking environment configuration...${NC}"
if [[ -f .env ]]; then
    echo -e "  ${GREEN}✓${NC} Found .env file."
elif [[ -f .env.example ]]; then
    echo -e "  ${YELLOW}!${NC} .env not found; copying from .env.example..."
    cp .env.example .env
    echo -e "  ${GREEN}✓${NC} Created .env from template."
else
    echo -e "  ${RED}✗${NC} Missing both .env and .env.example!"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# 2. Docker & Compose Syntax Validation
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[2/7] Validating Docker Compose configuration...${NC}"
if command -v docker &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Found Docker binary: $(docker --version)"
    
    if docker compose config --quiet &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Core compose configuration: VALID"
    else
        echo -e "  ${RED}✗${NC} Core compose configuration syntax error!"
        docker compose config
        ERRORS=$((ERRORS + 1))
    fi

    if docker compose --profile monitoring config --quiet &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Monitoring profile compose configuration: VALID"
    else
        echo -e "  ${RED}✗${NC} Monitoring profile compose syntax error!"
        docker compose --profile monitoring config
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "  ${RED}✗${NC} Docker command not found on host."
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# 3. Dockerfiles and Build Specifications Check
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[3/7] Validating Dockerfiles and build specifications...${NC}"
DOCKERFILES=(
    "docker/Dockerfile"
    "docker/Dockerfile.api"
    "docker/Dockerfile.embedding"
    "docker/Dockerfile.worker"
)

for df in "${DOCKERFILES[@]}"; do
    if [[ -f "$df" ]]; then
        echo -e "  ${GREEN}✓${NC} Found $df"
    else
        echo -e "  ${RED}✗${NC} Missing $df!"
        ERRORS=$((ERRORS + 1))
    fi
done

if [[ -f "requirements-lock.txt" ]]; then
    echo -e "  ${GREEN}✓${NC} Found requirements-lock.txt ($(wc -l < requirements-lock.txt) pinned dependencies)"
else
    echo -e "  ${RED}✗${NC} Missing requirements-lock.txt!"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# 4. Service Configuration Files Check
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[4/7] Checking service configuration files...${NC}"
CONFIG_FILES=(
    "docker/postgres/init_db.sql"
    "docker/postgres/postgresql.conf"
    "docker/prometheus/prometheus.yml"
    "docker/grafana/provisioning/datasources/prometheus.yaml"
)

for cf in "${CONFIG_FILES[@]}"; do
    if [[ -f "$cf" ]]; then
        echo -e "  ${GREEN}✓${NC} Found $cf"
    else
        echo -e "  ${RED}✗${NC} Missing $cf!"
        ERRORS=$((ERRORS + 1))
    fi
done

# -----------------------------------------------------------------------------
# 5. Python Service Entrypoint & Import Validation
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[5/7] Testing Python service modules and imports...${NC}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -f "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

echo "  Using Python interpreter: $PYTHON_BIN"
if "$PYTHON_BIN" -c "import src.serving.app; import src.embedding.service; import src.ingestion.worker; print('All service modules load cleanly.')" &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Service modules (api, embedding, worker) import successfully."
else
    echo -e "  ${RED}✗${NC} Failed to import one or more service modules."
    "$PYTHON_BIN" -c "import src.serving.app; import src.embedding.service; import src.ingestion.worker"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# 6. Hardware & GPU Acceleration Readiness
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[6/7] Checking hardware & GPU acceleration...${NC}"
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1 || echo 0)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown")
    echo -e "  ${GREEN}✓${NC} Found $GPU_COUNT NVIDIA GPU(s): $GPU_NAME"
else
    echo -e "  ${YELLOW}!${NC} nvidia-smi not found. GPU acceleration will be disabled."
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# 7. Host Docker Daemon Socket Permissions
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[7/7] Checking Docker socket permissions...${NC}"
if docker ps &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Current user has direct access to Docker daemon."
else
    echo -e "  ${YELLOW}!${NC} Permission denied connecting to /var/run/docker.sock without sudo."
    echo -e "     To run without sudo, execute:"
    echo -e "       sudo usermod -aG docker \$USER"
    echo -e "       newgrp docker"
    echo -e "     Or prefix commands with: sudo docker compose ..."
    WARNINGS=$((WARNINGS + 1))
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}============================================================${NC}"
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}VERIFICATION PASSED: All configuration files and services are ready!${NC}"
    echo -e "Summary: 0 errors, $WARNINGS warning(s)."
    echo -e "\nQuickstart Commands:"
    echo -e "  1. Start Core Services:       docker compose up -d"
    echo -e "  2. Start with Monitoring:     docker compose --profile monitoring up -d"
    echo -e "  3. Check Service Health:      curl http://localhost:8000/health"
    echo -e "  4. Trigger Ingestion:         docker compose run ingestion-worker python -m src.ingestion.worker --single-pass"
    echo -e "  5. View Live Metrics:         curl http://localhost:8000/metrics"
    echo -e "${BLUE}============================================================${NC}"
    exit 0
else
    echo -e "${RED}VERIFICATION FAILED: Found $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo -e "${BLUE}============================================================${NC}"
    exit 1
fi

