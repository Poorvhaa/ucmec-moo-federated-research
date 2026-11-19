# UCMEC-mmWave-Fronthaul: Multi-Objective Optimization Platform

> **Upstream Attribution**: Based on research by [qlt315/UCMEC-mmWave-Fronthaul](https://github.com/qlt315/UCMEC-mmWave-Fronthaul) published in *IEEE Transactions on Mobile Computing*: *"Towards Decentralized Task Offloading and Resource Allocation in User-Centric MEC"*.

This repository extends the original UCMEC simulator into a full-stack **Multi-Objective Optimization (MOO)** platform optimizing:
1. **Execution Delay** ($ms$)
2. **Device Energy Consumption** ($mJ$)
3. **Communication Overhead** ($kB$)
4. **Network Task Reliability** (Success rate $\%$ under deadlines and channel outages)

It includes a conic interior-point optimization engine (CLARABEL), durable asynchronous experiment runner, structured SQLite metadata store, FastAPI backend, and an interactive React/TypeScript research dashboard.

---

## 1. Quick Start (Windows / PowerShell)

### Prerequisites
- Conda environment named `ucmec` (Python 3.10)
- Node.js (v20+ or v22+) & npm

### Activating the Environment & Verifying Dependencies
```powershell
# Verify packages in the ucmec environment
conda run -n ucmec python -c "import torch, cvxpy, clarabel, fastapi; print('Dependencies verified successfully')"
```

### Running Unit Tests
```powershell
# Run the complete test suite (21 automated unit and integration tests)
conda run -n ucmec pytest tests -v
```

### Running a Training Smoke Run
```powershell
# Train with the balanced multi-objective preset (400 steps, 10 agents)
conda run -n ucmec python train/train.py --algorithm_name mappo --num_agents 10 --num_env_steps 400 --episode_length 200 --n_rollout_threads 1 --n_training_threads 1 --experiment_name smoke_moo_balanced --moo_preset balanced --log_interval 1 --share_policy --use_centralized_V --cuda
```

> **Important Flag Semantics Note**: In this repository, flags such as `--cuda`, `--share_policy`, and `--use_centralized_V` use `store_false` semantics with `default=True`. Passing `--cuda` sets `cuda=False` (forcing CPU execution). Passing `--share_policy` and `--use_centralized_V` configures independent decentralized training (IPPO).

---

## 2. CLI Policy Evaluation & Pareto Frontier Analysis

Evaluate policies deterministically across independent seeds on fixed held-out scenarios:

```powershell
# Fast multi-preset evaluation
conda run -n ucmec python evaluate.py --presets legacy delay_only balanced energy_priority reliability_priority --seeds 42 43 44 --num_steps 200
```

Evaluation outputs are aggregated at the independent seed level ($\mu \pm \sigma$) and saved to `experiments_storage/evaluations/evaluation_summary.json` and `.csv`.

---

## 3. Launching the Backend API & Frontend Dashboard

### Starting the FastAPI Server (Port 8000)
The backend hosts the REST API, serves the built React frontend, and manages background experiment subprocesses:
```powershell
conda run -n ucmec uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to access the research dashboard.

### API Endpoints
- `GET  /api/health`: Health status, worker state, and queue length.
- `GET  /api/capabilities`: Solvers, algorithms, and system dependency versions.
- `GET  /api/presets`: Pre-configured scalarization weight profiles.
- `POST /api/experiments`: Launch a new durable training experiment.
- `GET  /api/experiments`: Paginated list of experiments.
- `GET  /api/experiments/{id}`: Detailed experiment state and parameters.
- `POST /api/experiments/{id}/cancel`: Safely terminate a running experiment.
- `GET  /api/experiments/{id}/metrics`: Structured time-series metrics.
- `GET  /api/experiments/{id}/logs`: Tail live subprocess stdout/stderr.
- `GET  /api/experiments/{id}/artifacts`: List generated checkpoints and CSV/JSON files.
- `GET  /api/experiments/{id}/artifacts/{path}`: Secure artifact download with path traversal shield.
- `GET  /api/comparisons`: 4D empirical non-dominance comparison across all completed runs.

---

## 4. Building the Frontend
The frontend is built with React 19, TypeScript, Vite, Recharts, and Lucide-React:
```powershell
cd frontend
npm install
npm run build
```
Static production assets are outputted to `frontend/dist/` and automatically mounted by FastAPI.

---

## 5. Deployment with Docker Compose

A production-ready container stack is provided:
```bash
# Build and start services in background
docker compose up -d --build

# View logs
docker compose logs -f

# Check health
curl http://localhost:8000/api/health
```

---

## 6. Documentation Index
- [Metric Contract & Physical Models](docs/metric_contract.md)
- [System Architecture](docs/architecture.md)
- [Changelog & Baseline Repairs](docs/changelog.md)
- [Research Evaluation Protocol](docs/research_protocol.md)
- [Research Contribution Draft](docs/research_contribution_draft.md)
