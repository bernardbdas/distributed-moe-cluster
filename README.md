```
███▄ ▄███▓ ▒█████  ▓█████     ▄████▄   ██▓    █    ██   ██████ ▄▄▄█████▓▓█████  ██▀███
▓██▒▀█▀ ██▒▒██▒  ██▒▓█   ▀    ▒██▀ ▀█  ▓██▒    ██  ▓██▒▒██    ▒ ▓  ██▒ ▓▒▓█   ▀ ▓██ ▒ ██▒
▓██    ▓██░▒██░  ██▒▒███      ▒▓█    ▄ ▒██░   ▓██  ▒██░░ ▓██▄   ▒ ▓██░ ▒░▒███   ▓██ ░▄█ ▒
▒██    ▒██ ▒██   ██░▒▓█  ▄    ▒▓▓▄ ▄██▒▒██░   ▓▓█  ░██░  ▒   ██▒░ ▓██▓ ░ ▒▓█  ▄ ▒██▀▀█▄
▒██▒   ░██▒░ ████▓▒░░▒████▒   ▒ ▓███▀ ░░██████▒▒█████▓ ▒██████▒▒  ▒██▒ ░ ░▒████▒░██▓ ▒██▒
░ ▒░   ░  ░░ ▒░▒░▒░ ░░ ▒░ ░   ░ ░▒ ▒  ░░ ▒░▓  ░▒▓▒ ▒ ▒ ▒ ▒▓▒ ▒ ░  ▒ ░░   ░░ ▒░ ░░ ▒▓ ░▒▓░
░  ░      ░  ░ ▒ ▒░  ░ ░  ░     ░  ▒   ░ ░ ▒  ░░▒░ ░ ░ ░ ░▒  ░ ░    ░     ░ ░  ░  ░▒ ░ ▒░
░      ░   ░ ░ ░ ▒     ░      ░          ░ ░   ░░░ ░ ░ ░  ░  ░    ░         ░     ░░   ░
       ░       ░ ░     ░  ░   ░ ░          ░  ░  ░           ░              ░  ░   ░
                              ░
```

# SYS_02 // DISTRIBUTED 3D-PARALLEL FOUNDATION MODEL CLUSTER

```
[SYSTEM ID]       SYS_02
[ARCHITECTURE]    14B MoE Foundation Model (8 Experts, Top-2 Routing, 32k RoPE)
[PARALLELISM]     3D Parallelism: TP=4 (Megatron) + PP=2 (1F1B) + DP=4 (DeepSpeed ZeRO-3)
[ACCELERATION]    PyTorch FSDP + FlashAttention-3 + InfiniBand NDR 400 Gbps
[POST-TRAINING]   Direct Preference Optimization (DPO) & Online Reinforcement Alignment
[FRONTEND STACK]  Next.js 16 (App Router) + React 19 + TypeScript (Turbopack + Bun)
[BACKEND STACK]   FastAPI + PyTorch + Python 3.11 (Managed with uv)
[COMMAND RUNNER]  just (Justfile)
[PERF METRICS]    48.2% MFU // 32x H100 Scale // 32k Long-Context // Zero Divergence
```

---

## ⚡ ARCHITECTURAL SPECIFICATIONS

Scalable distributed pre-training and alignment harness coordinating 32x NVIDIA H100 SXM5 (80GB VRAM) GPUs across 4 nodes. Implements 3D hybrid parallelism to eliminate OOM bottlenecks and maximize Model FLOPs Utilization (MFU).

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     DGX-H100 Superpod (32x GPUs)                       │
 │        InfiniBand Quantum-2 NDR 400 Gbps // NVLink 900 GB/s            │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │ (TP=4)                 │ (PP=2)                 │ (DP=4 ZeRO-3)
 ┌─────────▼──────────┐   ┌─────────▼──────────┐   ┌─────────▼──────────┐
 │ Tensor Parallelism │   │ Pipeline Parallel  │   │ ZeRO-3 Sharding    │
 │ Megatron-LM Col/Row│   │ 1F1B Interleaved   │   │ Opt States, Grads, │
 │ Linear Layer Split │   │ Virtual Stages     │   │ Model Parameters   │
 └─────────┬──────────┘   └─────────┬──────────┘   └─────────┬──────────┘
           └────────────────────────┼────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │                  Mixture-of-Experts (MoE) Routing                      │
 │        - 8 Experts Total // Top-2 Token Softmax Gating                 │
 │        - Auxiliary Load-Balancing Loss with Capacity Factor 1.25       │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │           Direct Preference Optimization (DPO) Alignment               │
 │        - Closed-form policy loss optimization without reward model     │
 │        - Implicit reward margin tracking under Bradley-Terry model     │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 INDEPENDENT EXECUTION WITH `just`

Frontend and backend run independently or concurrently:

```bash
# Display all available commands
just

# Install both backend (uv) and frontend (bun) dependencies
just install

# Run backend independently on http://localhost:8002
just backend

# Run Next.js frontend independently on http://localhost:5172
just frontend

# Run both backend and frontend concurrently
just dev

# Run comprehensive test suite
just test

# Launch FSDP Pre-Training loop via CLI
just train steps="20"

# Launch Direct Preference Optimization (DPO) alignment via CLI
just align steps="10"

# Inspect live 32-GPU Superpod topology and memory states via CLI
just cluster-status
```

---

## 📦 CONTAINERIZATION (DOCKER & DOCKER-COMPOSE)

```bash
# Build multi-stage production image (Next.js static export + uv Python)
just docker-build

# Launch cluster service
just docker-up

# Verify health
curl -s http://localhost:8002/health | jq .
```

---

## 🌐 API SPECIFICATION

| Verb | Path | Description |
|---|---|---|
| `GET` | `/health` | Cluster topology status and MFU baseline |
| `GET` | `/api/v1/cluster/topology` | Real-time status for all 32 H100 GPUs |
| `GET` | `/api/v1/moe/routing` | Live 8-expert top-2 routing distribution |
| `POST` | `/api/v1/train/step` | Trigger FSDP pre-training step with loss metrics |
| `POST` | `/api/v1/alignment/dpo` | Execute DPO alignment step and reward delta |
