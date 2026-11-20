# Quickstart Guide for Windows

This guide provides tested, copy-paste PowerShell commands to verify, start, train, and evaluate the UCMEC Multi-Objective Optimization and Federated Multi-Agent Learning platform on Windows 10/11.

---

## 1. Prerequisites & Environment

The project requires Python 3.10 inside the Anaconda/Miniconda environment named `ucmec`.

```powershell
# Open project directory
cd D:\UCMEC-mmWave-Fronthaul

# Verify Python version and interpreter path
conda run -n ucmec python --version
conda run -n ucmec python -c "import sys; print(sys.executable)"
```

---

## 2. Preflight Diagnostic Doctor

Run the automated diagnostic verifier to validate conda environment dependencies, CLARABEL conic solver availability, frontend assets, and writable directories:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
```

*Expected output:* `Doctor Diagnosis: ALL SYSTEMS HEALTHY & OPERATIONAL.`

---

## 3. Frontend Production Build

The interactive research dashboard is built with React 19, TypeScript, and Vite. Compile the frontend production bundle:

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
```

---

## 4. Launch Local Backend & Research Dashboard

The application runs via FastAPI with an integrated durable experiment worker thread (concurrency: 1). **Do not launch a separate worker daemon**, as the queue engine is managed internally by the service process:

```powershell
# Start server on default port 8000
powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1

# Alternative direct Uvicorn command:
conda run --no-capture-output -n ucmec python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Once running, navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 5. Multi-Objective Optimization Smoke Training

Execute a short single-site multi-objective PPO training run (10 agents, 200 environment steps, CLARABEL conic solver):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_moo.ps1 -Preset balanced -NumSteps 200

# Direct CLI equivalent:
conda run --no-capture-output -n ucmec python train/train.py `
    --algorithm_name mappo `
    --experiment_name smoke_moo_run `
    --num_agents 10 `
    --num_env_steps 200 `
    --episode_length 200 `
    --seed 42 `
    --moo_preset balanced `
    --solver_name CLARABEL `
    --channel_mode legacy `
    --log_interval 1 `
    --n_rollout_threads 1 `
    --n_training_threads 1 `
    --cuda
```

---

## 6. Federated Multi-Agent Learning Smoke Run

Run a synchronous federated training session with 3 simulated edge sites, 2 communication rounds, and sample-weighted FedAvg:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_federated.ps1 -NumClients 3 -NumRounds 2 -ScenarioType iid

# Direct CLI equivalent:
conda run --no-capture-output -n ucmec python federated/runner.py `
    --experiment_name smoke_fed_run `
    --fed_num_clients 3 `
    --fed_num_rounds 2 `
    --fed_min_clients 1 `
    --fed_participation_rate 1.0 `
    --fed_dropout_prob 0.0 `
    --fed_scenario_type iid `
    --moo_preset balanced `
    --seed 42 `
    --solver_name CLARABEL `
    --channel_mode legacy `
    --cuda
```

---

## 7. Evaluate Saved Policy Checkpoints

Evaluate a saved federated or centralized actor policy checkpoint on held-out scenario seeds without exploration:

```powershell
# Automatically evaluates the latest checkpoint in experiments_storage/federated_runs:
powershell -ExecutionPolicy Bypass -File scripts\evaluate_federated.ps1 -NumSteps 50 -Seed 999

# Evaluate a specific checkpoint path:
powershell -ExecutionPolicy Bypass -File scripts\evaluate_federated.ps1 `
    -CheckpointPath "experiments_storage/federated_runs/<run_id>/checkpoints/global_actor_round_2.pt" `
    -NumSteps 100 `
    -Seed 999 `
    -Preset balanced
```

---

## 8. Full Comparative Benchmark Study

Run the matched three-method comparative benchmark (Local-Only vs Federated PPO vs Centralized Pooled-Data Reference):

```powershell
conda run --no-capture-output -n ucmec python federated/runner.py `
    --compare_methods `
    --fed_num_clients 3 `
    --fed_num_rounds 2 `
    --seed 42 `
    --moo_preset balanced `
    --cuda
```

---

## 9. Automated Test Suite Execution

Execute the full pytest verification suite across all 41 unit, integration, and e2e test modules:

```powershell
conda run -n ucmec python -m pytest tests -v
```
