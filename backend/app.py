"""
FastAPI Application for Multi-Objective Optimization Experiment Dashboard.
"""
import os
import sys
import uuid
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.database import Database, DB_PATH
from backend.worker import ExperimentWorker
from backend.schemas import (
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentListResponse,
    CapabilitiesResponse,
    PresetInfo,
    FederatedExperimentCreateRequest,
    FederatedExperimentResponse,
    FederatedExperimentListResponse,
    FederatedCapabilitiesResponse,
)
from federated.scenarios import generate_federation_scenarios
from moo.config import PRESETS, DEFAULT_REFERENCE_SCALES, DEFAULT_PHYSICAL_PARAMS
from moo.pareto import find_nondominated_set, check_scenario_compatibility
from moo.scalarization import validate_and_normalize_weights


app = FastAPI(
    title="UCMEC Multi-Objective Optimization API",
    description="Research dashboard and experiment execution engine for wireless mobile edge computing.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(DB_PATH)
# Reconcile interrupted jobs on startup
db.reconcile_interrupted_jobs()
worker = ExperimentWorker(db, max_concurrent=1)


def get_git_info() -> Dict[str, Any]:
    commit = None
    dirty = False
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            commit = res.stdout.strip()
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
        if status_res.returncode == 0 and status_res.stdout.strip():
            dirty = True
    except Exception:
        pass
    return {"git_commit": commit, "git_dirty": dirty}


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    expected = os.getenv("API_AUTH_KEY")
    if expected:
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key header")


@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "database": "connected",
        "worker": "active",
        "active_jobs": len(worker.active_processes),
        "queued_jobs": worker.job_queue.qsize()
    }


@app.get("/api/capabilities", response_model=CapabilitiesResponse)
def get_capabilities():
    import torch
    import cvxpy as cp
    import clarabel
    return CapabilitiesResponse(
        available_algorithms=["mappo", "rmappo"],
        available_solvers=["CLARABEL", "SCS"],
        supported_channel_modes=["legacy", "sinr_outage"],
        max_agents=60,
        max_steps=100000,
        system_python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cvxpy_version=cp.__version__,
        clarabel_version=clarabel.__version__,
    )


@app.get("/api/presets", response_model=List[PresetInfo])
def get_presets():
    result = []
    for k, v in PRESETS.items():
        result.append(PresetInfo(
            id=k,
            name=v["name"],
            description=v["description"],
            weights=v["weights"],
            mode=v["mode"],
            use_corrected_cpu_alloc=v.get("use_corrected_cpu_alloc", True),
        ))
    return result


@app.post("/api/experiments", response_model=ExperimentResponse, dependencies=[Depends(verify_api_key)])
def create_experiment(req: ExperimentCreateRequest):
    exp_id = str(uuid.uuid4())[:8]
    
    # Resolve preset and weights
    preset_cfg = PRESETS.get(req.preset_name, PRESETS["balanced"])
    if any(w is not None for w in [req.w_delay, req.w_energy, req.w_overhead, req.w_reliability]):
        raw_weights = {
            "w_delay": req.w_delay or 0.0,
            "w_energy": req.w_energy or 0.0,
            "w_overhead": req.w_overhead or 0.0,
            "w_reliability": req.w_reliability or 0.0,
        }
        weights = validate_and_normalize_weights(raw_weights)
    else:
        weights = preset_cfg["weights"]

    log_dir = Path("experiments_storage") / "runs" / f"{req.name}_{exp_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info()

    exp_data = {
        "id": exp_id,
        "name": f"{req.name}_{exp_id}",
        "status": "queued",
        "algorithm_name": req.algorithm_name,
        "preset_name": req.preset_name,
        "weights_json": json.dumps(weights),
        "num_agents": req.num_agents,
        "num_env_steps": req.num_env_steps,
        "episode_length": req.episode_length,
        "seed": req.seed,
        "current_step": 0,
        "total_steps": req.num_env_steps,
        "progress_percent": 0.0,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "runtime_s": 0.0,
        "git_commit": git_info["git_commit"],
        "git_dirty": 1 if git_info["git_dirty"] else 0,
        "solver_name": req.solver_name,
        "use_corrected_cpu_alloc": 1 if req.use_corrected_cpu_alloc else 0,
        "channel_mode": req.channel_mode,
        "config_json": json.dumps(req.model_dump()),
        "final_metrics_json": None,
        "error_message": None,
        "log_dir": str(log_dir),
    }

    db.create_experiment(exp_data)
    worker.enqueue(exp_id)

    created = db.get_experiment(exp_id)
    return ExperimentResponse(**created)


@app.get("/api/experiments", response_model=ExperimentListResponse)
def list_experiments(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    total, items = db.list_experiments(page=page, page_size=page_size)
    return ExperimentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[ExperimentResponse(**it) for it in items]
    )


@app.get("/api/experiments/{exp_id}", response_model=ExperimentResponse)
def get_experiment(exp_id: str):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse(**exp)


@app.post("/api/experiments/{exp_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_experiment(exp_id: str):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp["status"] not in ["queued", "running"]:
        return {"status": exp["status"], "message": f"Experiment is already {exp['status']}"}

    cancelled = worker.cancel(exp_id)
    db.update_experiment_status(exp_id, status="cancelled", finished_at=time.time())
    return {"status": "cancelled", "message": "Experiment cancellation requested"}


@app.get("/api/experiments/{exp_id}/metrics")
def get_experiment_metrics(exp_id: str):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    metrics = db.get_experiment_metrics(exp_id)
    return {"experiment_id": exp_id, "total_records": len(metrics), "metrics": metrics}


@app.get("/api/experiments/{exp_id}/logs")
def get_experiment_logs(exp_id: str, tail: int = Query(200, ge=1, le=2000)):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    log_dir = Path(exp["log_dir"])
    stdout_file = log_dir / "stdout.log"
    stderr_file = log_dir / "stderr.log"

    stdout_content = ""
    stderr_content = ""

    if stdout_file.exists():
        try:
            with open(stdout_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                stdout_content = "".join(lines[-tail:])
        except Exception:
            pass

    if stderr_file.exists():
        try:
            with open(stderr_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                stderr_content = "".join(lines[-tail:])
        except Exception:
            pass

    return {
        "experiment_id": exp_id,
        "stdout": stdout_content,
        "stderr": stderr_content
    }


@app.get("/api/experiments/{exp_id}/artifacts")
def list_experiment_artifacts(exp_id: str):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    artifacts = []
    # Check both log_dir and results folder
    log_dir = Path(exp["log_dir"])
    if log_dir.exists():
        for p in log_dir.glob("*"):
            if p.is_file():
                artifacts.append({"name": p.name, "size_bytes": p.stat().st_size, "type": "run_log"})

    run_base = Path("results") / "MyEnv" / "MyEnv" / exp["algorithm_name"] / exp["name"]
    if run_base.exists():
        for p in run_base.glob("**/*"):
            if p.is_file() and p.suffix in [".json", ".csv", ".pt", ".mat", ".tfevents"]:
                artifacts.append({
                    "name": str(p.relative_to(run_base)),
                    "size_bytes": p.stat().st_size,
                    "type": "result_file"
                })

    return {"experiment_id": exp_id, "artifacts": artifacts}


@app.get("/api/experiments/{exp_id}/artifacts/{artifact_path:path}")
def download_artifact(exp_id: str, artifact_path: str):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    allowed_roots = [
        Path(exp["log_dir"]).resolve(),
        (Path("results") / "MyEnv" / "MyEnv" / exp["algorithm_name"] / exp["name"]).resolve(),
        Path("experiments_storage").resolve(),
    ]

    # Resolve target file securely to prevent directory traversal
    target = None
    for root in allowed_roots:
        if root.exists():
            candidate = (root / artifact_path).resolve()
            try:
                if candidate.is_relative_to(root) and candidate.is_file():
                    target = candidate
                    break
            except AttributeError:
                # Python < 3.9 compatibility (though we are on Python 3.10)
                if str(candidate).startswith(str(root)) and candidate.is_file():
                    target = candidate
                    break

    if not target or not target.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found or path disallowed")

    return FileResponse(path=str(target), filename=target.name)


@app.get("/api/comparisons")
def get_comparisons():
    completed = db.get_all_completed_experiments()
    
    # Format candidates for Pareto evaluation
    candidates = []
    for exp in completed:
        fm = exp.get("final_metrics")
        if fm:
            candidates.append({
                "id": exp["id"],
                "name": exp["name"],
                "preset": exp["preset_name"],
                "algorithm": exp["algorithm_name"],
                "seed": exp["seed"],
                "num_agents": exp["num_agents"],
                "episode_length": exp["episode_length"],
                "scenario_name": "MyEnv",
                "delay_ms": fm.get("delay_ms", 0.0),
                "energy_mj": fm.get("energy_mj", 0.0),
                "overhead_kb": fm.get("overhead_kb", 0.0),
                "task_success_rate": fm.get("task_success_rate", 0.0),
                "reward": fm.get("reward", 0.0),
            })

    # Group by scenario compatibility
    nondominated_results = find_nondominated_set(candidates)

    return {
        "total_compared": len(candidates),
        "experiments": nondominated_results,
        "nondominated_subset": [c for c in nondominated_results if c.get("is_nondominated")],
    }


# ==========================================
# Federated Learning API Endpoints
# ==========================================

@app.get("/api/federation/capabilities", response_model=FederatedCapabilitiesResponse)
def get_federation_capabilities():
    return FederatedCapabilitiesResponse(
        supported_scenarios=[
            "iid",
            "task_heterogeneity",
            "channel_heterogeneity",
            "compute_heterogeneity",
            "sample_budget_heterogeneity",
            "full_heterogeneity",
        ],
        supported_aggregation="sample_weighted_fedavg",
        privacy_disclaimer="Raw trajectories remain local in this simulation. Model updates may still reveal information. Differential privacy and secure aggregation are not implemented.",
        max_clients=20,
        max_rounds=100,
        default_smoke_config={
            "num_clients": 3,
            "num_rounds": 2,
            "scenario_type": "iid",
            "preset_name": "balanced",
            "min_clients": 1,
            "participation_rate": 1.0,
            "dropout_prob": 0.0,
            "seed": 42,
            "solver_name": "CLARABEL",
            "channel_mode": "legacy",
        },
    )


@app.post("/api/federated-experiments", response_model=FederatedExperimentResponse, dependencies=[Depends(verify_api_key)])
def create_federated_experiment(req: FederatedExperimentCreateRequest):
    exp_id = str(uuid.uuid4())[:8]
    full_name = f"{req.name}_{exp_id}"
    log_dir = Path("experiments_storage") / "federated_runs" / full_name
    log_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info()

    # Pre-generate client scenario configurations
    client_configs = generate_federation_scenarios(
        num_clients=req.num_clients,
        scenario_type=req.scenario_type,
        base_seed=req.seed,
        rollout_steps=200,
    )

    fed_data = {
        "id": exp_id,
        "name": full_name,
        "status": "queued",
        "preset_name": req.preset_name,
        "scenario_type": req.scenario_type,
        "num_clients": req.num_clients,
        "num_rounds": req.num_rounds,
        "current_round": 0,
        "min_clients": req.min_clients,
        "participation_rate": req.participation_rate,
        "dropout_prob": req.dropout_prob,
        "seed": req.seed,
        "progress_percent": 0.0,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "runtime_s": 0.0,
        "git_commit": git_info["git_commit"],
        "git_dirty": 1 if git_info["git_dirty"] else 0,
        "solver_name": req.solver_name,
        "channel_mode": req.channel_mode,
        "config_json": json.dumps(req.model_dump()),
        "final_metrics_json": None,
        "communication_summary_json": None,
        "error_message": None,
        "log_dir": str(log_dir),
    }

    db.create_federated_experiment(fed_data)
    db.insert_federated_clients(exp_id, [c.to_dict() for c in client_configs])

    worker.enqueue(exp_id)

    created = db.get_federated_experiment(exp_id)
    return FederatedExperimentResponse(**created)


@app.get("/api/federated-experiments", response_model=FederatedExperimentListResponse)
def list_federated_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    total, items = db.list_federated_experiments(page=page, page_size=page_size)
    return FederatedExperimentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[FederatedExperimentResponse(**i) for i in items]
    )


@app.get("/api/federated-experiments/{exp_id}", response_model=FederatedExperimentResponse)
def get_federated_experiment(exp_id: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")
    return FederatedExperimentResponse(**exp)


@app.get("/api/federated-experiments/{exp_id}/rounds")
def get_federated_rounds(exp_id: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")
    rounds = db.get_federated_rounds(exp_id)
    return {"experiment_id": exp_id, "total_rounds": len(rounds), "rounds": rounds}


@app.get("/api/federated-experiments/{exp_id}/clients")
def get_federated_clients(exp_id: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")
    clients = db.get_federated_clients(exp_id)
    return {"experiment_id": exp_id, "total_clients": len(clients), "clients": clients}


@app.get("/api/federated-experiments/{exp_id}/metrics")
def get_federated_metrics(exp_id: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")
    rounds = db.get_federated_rounds(exp_id)
    return {
        "experiment_id": exp_id,
        "total_rounds": len(rounds),
        "final_metrics": exp.get("final_metrics"),
        "communication_summary": exp.get("communication_summary"),
        "round_metrics": [
            {
                "round": r["round"],
                "runtime_s": r["runtime_s"],
                "total_valid_samples": r["total_valid_samples"],
                "mean_delay_ms": r.get("mean_delay_ms"),
                "mean_energy_mj": r.get("mean_energy_mj"),
                "mean_overhead_kb": r.get("mean_overhead_kb"),
                "mean_task_success_rate": r.get("mean_task_success_rate"),
                "worst_task_success_rate": r.get("worst_task_success_rate"),
                "worst_delay_ms": r.get("worst_delay_ms"),
                "jains_fairness_index": r.get("jains_fairness_index"),
                "eval_delay_ms": r.get("eval_delay_ms"),
                "eval_energy_mj": r.get("eval_energy_mj"),
                "eval_overhead_kb": r.get("eval_overhead_kb"),
                "eval_task_success_rate": r.get("eval_task_success_rate"),
                "eval_reward": r.get("eval_reward"),
                "download_bytes": r.get("round_download_bytes"),
                "upload_bytes": r.get("round_upload_bytes"),
                "failed_bytes": r.get("round_failed_bytes"),
            }
            for r in rounds
        ],
    }


@app.post("/api/federated-experiments/{exp_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_federated_experiment(exp_id: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")
    if exp["status"] not in ["queued", "running"]:
        return {"status": exp["status"], "message": f"Experiment is already {exp['status']}"}

    worker.cancel(exp_id)
    db.update_federated_experiment_status(exp_id, status="cancelled", finished_at=time.time())
    return {"status": "cancelled", "message": "Federated experiment cancellation requested"}


@app.post("/api/federated-experiments/{exp_id}/resume", dependencies=[Depends(verify_api_key)])
def resume_federated_experiment(exp_id: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")
    if exp["status"] in ["running", "completed"]:
        return {"status": exp["status"], "message": f"Experiment cannot be resumed in '{exp['status']}' state"}

    db.update_federated_experiment_status(exp_id, status="queued", error_message=None)
    worker.enqueue(exp_id)
    return {"status": "queued", "message": "Federated experiment queued for warm resume"}


@app.get("/api/federated-experiments/{exp_id}/artifacts")
def list_federated_artifacts(exp_id: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")

    artifacts = []
    log_dir = Path(exp["log_dir"])
    if log_dir.exists():
        for p in log_dir.glob("*"):
            if p.is_file():
                artifacts.append({"name": p.name, "size_bytes": p.stat().st_size, "type": "root_artifact"})
        checkpoints_dir = log_dir / "checkpoints"
        if checkpoints_dir.exists():
            for p in checkpoints_dir.glob("*"):
                if p.is_file():
                    artifacts.append({"name": f"checkpoints/{p.name}", "size_bytes": p.stat().st_size, "type": "checkpoint_artifact"})

    return {"experiment_id": exp_id, "artifacts": artifacts}


@app.get("/api/federated-experiments/{exp_id}/artifacts/{subpath:path}")
def download_federated_artifact(exp_id: str, subpath: str):
    exp = db.get_federated_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Federated experiment not found")

    log_dir = Path(exp["log_dir"]).resolve()
    target = (log_dir / subpath).resolve()

    # Path traversal protection
    if not str(target).startswith(str(log_dir)):
        raise HTTPException(status_code=400, detail="Invalid path or directory traversal detected")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(path=str(target), filename=target.name)


@app.get("/api/federated-comparisons")
def get_federated_comparisons():
    """Returns comparative summary data across Local-Only, Federated, and Pooled references."""
    # Look for any comparative study artifacts
    studies = []
    base_dir = Path("experiments_storage") / "federated_runs"
    if base_dir.exists():
        for f in base_dir.rglob("comparative_study.json"):
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    studies.append(json.load(jf))
            except Exception:
                pass

    return {
        "total_studies": len(studies),
        "studies": studies,
    }



# Mount built frontend if available
frontend_dist = Path("frontend") / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
