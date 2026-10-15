# ==============================================================================
# AGENTS.MD — DISTRIBUTED MOE CLUSTER (SYS_02)
# Production Architecture, Rules & Operational Directives
# Author: Bernard Birendra Das (https://github.com/bernardbdas)
# ==============================================================================

> **MANDATORY SPECIFICATION FOR AUTONOMOUS AGENTS AND DEVELOPERS**
> This repository strictly adheres to the engineering standards, architecture, and design styles specified herein. No simplifications, removals, or degradations are permitted.

---

## 1. System Topology & Identity

- **System ID**: `SYS_02`
- **Repository Name**: `distributed-moe-cluster`
- **Primary Domain**: PyTorch FSDP & Ray Train Distributed Mixture-of-Experts (MoE) Foundation Model with Direct Preference Optimization (DPO)
- **Backend Port**: `8002` (FastAPI)
- **Frontend Framework**: Next.js 16 (App Router + Turbopack) + React 19 + TypeScript (Scaffolded with `bun`, statically exported via `output: 'export'`)
- **Frontend Port (Dev)**: `3000`
- **UI Aesthetic Style**: **Retro Phosphor Green CRT / Amber Monospace Terminal** (Inspired by Bernard's portfolio `bernardbdas.github.io` & UXPlanet Terminal Brutalism)
  - Color Palette: Deep black `#030704`, phosphor terminal green `#00ff66`, CRT scanline highlights, amber orange warnings `#ffb000`, hairline borders `#0d2818`
  - Typography: Monospace (`JetBrains Mono`, `IBM Plex Mono`)
  - Layout: High-density MoE routing telemetry dashboard with real-time per-expert load balancing histograms, token drop counters, and DPO alignment sliders
- **Domain CLI**: `moe-cluster` (`src/cli.py`)

---

## 2. Decoupled Architecture & Execution Directives

1. **Decoupled Independence**:
   - Backend and frontend are completely decoupled and must run independently.
   - Run backend independently: `just backend` (FastAPI on port `8002`).
   - Run frontend independently: `just frontend` (Next.js Turbopack dev server on port `3000`).
   - Run both concurrently: `just dev`.
2. **Production Serving**:
   - Compiling frontend via `just build` executes `next build` with `output: 'export'` targeting `frontend/dist`.
   - In production, FastAPI statically mounts `frontend/dist` at `/`.
   - API endpoints (`/health`, `/metrics`, `/api/forward`, `/api/dpo`, `/api/telemetry`) always take routing priority.

---

## 3. Project Management with `uv`

1. **UV Enforcement**: All Python environment management, locking, and execution MUST use `uv`.
2. **Build System**: Managed via `hatchling` in `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src"]
   ```
3. **Cross-Platform Compatibility**: Linux-specific packages (e.g. `deepspeed`, `triton`) must declare platform markers:
   ```toml
   dependencies = [
       "deepspeed>=0.14.0; sys_platform == 'linux'",
       "triton>=2.3.0; sys_platform == 'linux'",
   ]
   ```
   Ensuring development on macOS ARM64 and deployment across NVIDIA CUDA Linux clusters.

---

## 4. Frontend & UI Design Rules (`bun` & Phosphor Brutalism)

1. Managed exclusively using `bun` inside `frontend/`.
2. Retro-futuristic phosphor CRT terminal styling:
   - High-density monospace ribbons and ASCII headers.
   - Live expert gate routing histograms and coefficient sliders.
   - Zero soft shadows, sharp corners, and phosphor glow effects.

---

## 5. Command Runner (`just`) Specification

Standardized `Justfile` tasks:
- `just backend`: Run FastAPI server (`http://localhost:8002`) with reload.
- `just frontend`: Run Next.js Turbopack dev server.
- `just dev`: Launch backend and frontend concurrently.
- `just install`: Install backend with `uv sync --extra dev` and frontend with `bun install`.
- `just test`: Run `uv run pytest` and frontend TypeScript build.
- `just build`: Compile Next.js static export to `dist/` and build Python wheel.
- `just check`: Run `uv run ruff check src` and format checks.
- `just train-fsdp`: Execute simulated PyTorch FSDP MoE pre-training step.
- `just dpo-step`: Run Direct Preference Optimization alignment step.
- `just docker-build`: Build multi-stage production Docker image.
- `just docker-up`: Start MoE cluster services via Docker Compose.

---

## 6. Containerization & Docker

- Multi-stage `Dockerfile`:
  - Stage 1: `oven/bun:1-alpine` compiles Next.js to `frontend/dist`.
  - Stage 2: `python:3.11-slim` installs dependencies via `uv sync --frozen --no-dev` and hosts FastAPI.
- `docker-compose.yml`: Configured for cluster deployment with telemetry exporters.

---

## 7. Developer Tooling & `.vscode` Setup

- `.vscode/settings.json`: Configured with Ruff default formatter, format on save, and `.venv` python interpreter path.
- `.vscode/launch.json`: Debugging configurations for FastAPI backend, CLI commands, and Chrome frontend.
- `.vscode/extensions.json`: Recommended VS Code extensions.
- `.vscode/tasks.json`: Standard VS Code build and test tasks wired to `just`.

---

## 8. Environment Management Rules

- `example.env`: Tracked template with descriptions for world size, expert counts, and learning rates.
- `local.env`: Tracked developer defaults for immediate out-of-the-box local execution.
- `.env` / `.env.local`: Ignored in `.gitignore` for private cluster credentials.

---

## 9. Code Quality, Linters & Pre-Commit Hooks

- Linter & Formatter: `ruff` with `select = ["E", "F", "I"]`, `line-length = 100`.
- Pre-commit hooks (`.pre-commit-config.yaml`):
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files` (`--maxkb=5000`), `ruff`, and `ruff-format`.
- All git commits must be verified against pre-commit hooks.

---

## 10. Continuous Integration (CI)

- `.github/workflows/ci.yml` runs on every push and PR to `main`:
  - Python lint (`ruff check src`) and test (`pytest`).
  - Bun Next.js typecheck and build (`bun run build`).

---

## 11. Git & Remote Repository Rules

- Branch: `main`.
- Remote: `https://github.com/bernardbdas/distributed-moe-cluster.git`.
