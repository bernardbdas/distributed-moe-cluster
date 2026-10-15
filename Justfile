# ═══════════════════════════════════════════════════════════════════════
# SYS_02 // DISTRIBUTED MOE CLUSTER JUSTFILE
# Frontend: Next.js (App Router) + React 19 + TypeScript (Bun)
# Backend:  FastAPI + PyTorch FSDP + DeepSpeed ZeRO-3 (Python uv)
# ═══════════════════════════════════════════════════════════════════════

default:
    @just --list

install:
    @echo "[+] Syncing Python backend dependencies with uv..."
    uv sync --extra dev
    @echo "[+] Installing Next.js frontend dependencies with bun..."
    cd frontend && bun install

# Run backend independently on port 8002
backend host="0.0.0.0" port="8002":
    @echo "[*] Launching Distributed MoE Cluster API on http://{{host}}:{{port}}..."
    uv run uvicorn src.server:app --host {{host}} --port {{port}} --reload

# Run frontend independently on port 5172
frontend:
    @echo "[*] Launching Next.js frontend on http://localhost:5172..."
    cd frontend && bun run dev

# Run both backend and frontend concurrently
dev:
    @echo "[*] Launching Distributed MoE Cluster (Backend: :8002 | Frontend: :5172)..."
    (trap 'kill 0' SIGINT; just backend & just frontend)

# Build static production bundle with Next.js
build:
    @echo "[*] Building Next.js production bundle..."
    cd frontend && bun run build

# Run test suite
test:
    @echo "[*] Running backend pytest suite..."
    uv run pytest
    @echo "[*] Typechecking Next.js frontend..."
    cd frontend && bun run build

# Launch FSDP Pre-Training loop via CLI
train steps="10":
    uv run moe-cluster train --steps {{steps}}

# Launch Direct Preference Optimization (DPO) alignment via CLI
align steps="5":
    uv run moe-cluster align --steps {{steps}}

# Inspect GPU Superpod topology and memory via CLI
cluster-status:
    uv run moe-cluster cluster-status

docker-build:
    docker build -t distributed-moe-cluster:latest .

docker-up:
    docker compose up -d

docker-down:
    docker compose down
