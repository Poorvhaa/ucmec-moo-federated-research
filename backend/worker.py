"""
Background Worker and Subprocess Execution Engine for Experiments.
"""
import subprocess
import threading
import queue
import time
import os
import sys
import signal
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional

from backend.database import Database


class ExperimentWorker:
    def __init__(self, db: Database, max_concurrent: int = 1):
        self.db = db
        self.max_concurrent = max_concurrent
        self.job_queue: queue.Queue = queue.Queue()
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.cancelled_jobs: set = set()
        self.lock = threading.Lock()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def enqueue(self, exp_id: str):
        self.job_queue.put(exp_id)

    def cancel(self, exp_id: str) -> bool:
        with self.lock:
            self.cancelled_jobs.add(exp_id)
            proc = self.active_processes.get(exp_id)
            if proc:
                self._terminate_process(proc)
                return True
        return False

    def _terminate_process(self, proc: subprocess.Popen):
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    check=False
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(0.5)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_loop(self):
        while True:
            try:
                exp_id = self.job_queue.get()
                if exp_id in self.cancelled_jobs:
                    self.db.update_experiment_status(exp_id, status="cancelled", finished_at=time.time())
                    self.db.update_federated_experiment_status(exp_id, status="cancelled", finished_at=time.time())
                    self.job_queue.task_done()
                    continue

                fed_exp = self.db.get_federated_experiment(exp_id)
                if fed_exp:
                    self._execute_federated_experiment(exp_id)
                else:
                    self._execute_experiment(exp_id)

                self.job_queue.task_done()
            except Exception as e:
                time.sleep(1.0)

    def _execute_experiment(self, exp_id: str):
        exp = self.db.get_experiment(exp_id)
        if not exp:
            return

        start_time = time.time()
        self.db.update_experiment_status(exp_id, status="running", started_at=start_time)
        
        log_dir = Path(exp["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"

        # Determine python executable: use current sys.executable
        python_exe = sys.executable

        # Build clean argument list without shell concatenation
        cmd = [
            python_exe,
            "train/train.py",
            "--algorithm_name", exp["algorithm_name"],
            "--num_agents", str(exp["num_agents"]),
            "--num_env_steps", str(exp["num_env_steps"]),
            "--episode_length", str(exp["episode_length"]),
            "--seed", str(exp["seed"]),
            "--experiment_name", exp["name"],
            "--moo_preset", exp["preset_name"],
            "--solver_name", exp["solver_name"],
            "--channel_mode", exp["channel_mode"],
            "--log_interval", "1",
            "--n_rollout_threads", "1",
            "--n_training_threads", "1",
            "--cuda", # Remember store_false semantics: passing --cuda sets cuda=False
        ]

        if not exp.get("share_policy", True):
            cmd.append("--share_policy") # store_false -> False
        if not exp.get("use_centralized_V", True):
            cmd.append("--use_centralized_V") # store_false -> False

        if not exp.get("use_corrected_cpu_alloc", True):
            cmd.append("--legacy_cpu_alloc")

        weights = exp.get("weights", {})
        if weights:
            if "w_delay" in weights:
                cmd.extend(["--w_delay", str(weights["w_delay"])])
            if "w_energy" in weights:
                cmd.extend(["--w_energy", str(weights["w_energy"])])
            if "w_overhead" in weights:
                cmd.extend(["--w_overhead", str(weights["w_overhead"])])
            if "w_reliability" in weights:
                cmd.extend(["--w_reliability", str(weights["w_reliability"])])

        with open(stdout_path, "w", encoding="utf-8") as f_out, open(stderr_path, "w", encoding="utf-8") as f_err:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f_out,
                    stderr=f_err,
                    cwd=str(Path.cwd()),
                    env=os.environ.copy()
                )
                with self.lock:
                    self.active_processes[exp_id] = proc
            except Exception as e:
                self.db.update_experiment_status(
                    exp_id,
                    status="failed",
                    finished_at=time.time(),
                    error_message=f"Failed to start training subprocess: {str(e)}"
                )
                return

        # Poll metrics file while running
        csv_file = None
        # Locate the run directory dynamically
        run_base = Path("results") / "MyEnv" / "MyEnv" / exp["algorithm_name"] / exp["name"]
        
        while proc.poll() is None:
            time.sleep(1.0)
            if exp_id in self.cancelled_jobs:
                self._terminate_process(proc)
                self.db.update_experiment_status(
                    exp_id,
                    status="cancelled",
                    finished_at=time.time(),
                    runtime_s=time.time() - start_time
                )
                with self.lock:
                    self.active_processes.pop(exp_id, None)
                return

            # Check for latest metrics.json / csv
            if not csv_file and run_base.exists():
                candidates = list(run_base.glob("run*/logs/metrics.json"))
                if candidates:
                    csv_file = candidates[-1]

            if csv_file and csv_file.exists():
                try:
                    with open(csv_file, "r", encoding="utf-8") as jf:
                        records = json.load(jf)
                        if records:
                            latest = records[-1]
                            curr_step = latest.get("step", 0)
                            prog = min(100.0, (curr_step / exp["num_env_steps"]) * 100.0)
                            self.db.update_experiment_status(
                                exp_id,
                                status="running",
                                current_step=curr_step,
                                progress_percent=prog,
                                runtime_s=time.time() - start_time
                            )
                except Exception:
                    pass

        returncode = proc.poll()
        finish_time = time.time()
        runtime = finish_time - start_time

        with self.lock:
            self.active_processes.pop(exp_id, None)

        # Ingest final metrics
        final_metrics = None
        if run_base.exists():
            json_candidates = list(run_base.glob("run*/logs/metrics.json"))
            if json_candidates:
                try:
                    with open(json_candidates[-1], "r", encoding="utf-8") as jf:
                        records = json.load(jf)
                        for r in records:
                            self.db.insert_metric_record(exp_id, r)
                        if records:
                            final_metrics = records[-1]
                except Exception:
                    pass

        if returncode == 0:
            self.db.update_experiment_status(
                exp_id,
                status="completed",
                current_step=exp["num_env_steps"],
                progress_percent=100.0,
                finished_at=finish_time,
                runtime_s=runtime,
                final_metrics=final_metrics
            )
        else:
            err_msg = f"Process exited with non-zero code {returncode}."
            if stderr_path.exists():
                try:
                    with open(stderr_path, "r", encoding="utf-8") as f_err:
                        lines = f_err.readlines()
                        if lines:
                            err_msg += " " + "".join(lines[-5:]).strip()
                except Exception:
                    pass
            self.db.update_experiment_status(
                exp_id,
                status="failed",
                finished_at=finish_time,
                runtime_s=runtime,
                final_metrics=final_metrics,
                error_message=err_msg[:500]
            )

    def _execute_federated_experiment(self, exp_id: str):
        exp = self.db.get_federated_experiment(exp_id)
        if not exp:
            return

        start_time = time.time()
        self.db.update_federated_experiment_status(exp_id, status="running", started_at=start_time)
        
        log_dir = Path(exp["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"

        python_exe = sys.executable

        cmd = [
            python_exe,
            "federated/runner.py",
            "--experiment_name", exp["name"],
            "--fed_num_clients", str(exp["num_clients"]),
            "--fed_num_rounds", str(exp["num_rounds"]),
            "--fed_min_clients", str(exp["min_clients"]),
            "--fed_participation_rate", str(exp["participation_rate"]),
            "--fed_dropout_prob", str(exp["dropout_prob"]),
            "--fed_scenario_type", exp["scenario_type"],
            "--moo_preset", exp["preset_name"],
            "--seed", str(exp["seed"]),
            "--solver_name", exp.get("solver_name", "CLARABEL"),
            "--channel_mode", exp.get("channel_mode", "legacy"),
            "--fed_output_dir", str(log_dir),
            "--cuda",
        ]

        with open(stdout_path, "w", encoding="utf-8") as f_out, open(stderr_path, "w", encoding="utf-8") as f_err:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f_out,
                    stderr=f_err,
                    cwd=str(Path.cwd()),
                    env=os.environ.copy()
                )
                with self.lock:
                    self.active_processes[exp_id] = proc
            except Exception as e:
                self.db.update_federated_experiment_status(
                    exp_id,
                    status="failed",
                    finished_at=time.time(),
                    error_message=f"Failed to start federated subprocess: {str(e)}"
                )
                return

        checkpoints_dir = log_dir / "checkpoints"
        ingested_rounds = set()

        while proc.poll() is None:
            time.sleep(1.0)
            if exp_id in self.cancelled_jobs:
                self._terminate_process(proc)
                self.db.update_federated_experiment_status(
                    exp_id,
                    status="cancelled",
                    finished_at=time.time(),
                    runtime_s=time.time() - start_time
                )
                with self.lock:
                    self.active_processes.pop(exp_id, None)
                return

            if checkpoints_dir.exists():
                manifests = sorted(list(checkpoints_dir.glob("round_*_manifest.json")))
                for mf in manifests:
                    try:
                        r_num = int(mf.stem.split("_")[1])
                        if r_num not in ingested_rounds:
                            with open(mf, "r", encoding="utf-8") as jf:
                                r_data = json.load(jf)
                                self.db.insert_federated_round(exp_id, r_data)
                                ingested_rounds.add(r_num)
                                prog = min(100.0, (r_num / exp["num_rounds"]) * 100.0)
                                self.db.update_federated_experiment_status(
                                    exp_id,
                                    status="running",
                                    current_round=r_num,
                                    progress_percent=prog,
                                    runtime_s=time.time() - start_time
                                )
                    except Exception:
                        pass

        returncode = proc.poll()
        finish_time = time.time()
        runtime = finish_time - start_time

        with self.lock:
            self.active_processes.pop(exp_id, None)

        # Ingest remaining round manifests
        if checkpoints_dir.exists():
            manifests = sorted(list(checkpoints_dir.glob("round_*_manifest.json")))
            for mf in manifests:
                try:
                    r_num = int(mf.stem.split("_")[1])
                    if r_num not in ingested_rounds:
                        with open(mf, "r", encoding="utf-8") as jf:
                            r_data = json.load(jf)
                            self.db.insert_federated_round(exp_id, r_data)
                            ingested_rounds.add(r_num)
                except Exception:
                    pass

        comm_summary = None
        ledger_file = log_dir / "communication_ledger.json"
        if ledger_file.exists():
            try:
                with open(ledger_file, "r", encoding="utf-8") as lf:
                    ledger_data = json.load(lf)
                    comm_summary = ledger_data.get("summary")
            except Exception:
                pass

        final_metrics = None
        latest_manifest = checkpoints_dir / f"round_{exp['num_rounds']}_manifest.json"
        if latest_manifest.exists():
            try:
                with open(latest_manifest, "r", encoding="utf-8") as jf:
                    m_data = json.load(jf)
                    final_metrics = {
                        "local": m_data.get("local_training_metrics"),
                        "eval": m_data.get("eval_metrics")
                    }
            except Exception:
                pass

        if returncode == 0:
            self.db.update_federated_experiment_status(
                exp_id,
                status="completed",
                current_round=exp["num_rounds"],
                progress_percent=100.0,
                finished_at=finish_time,
                runtime_s=runtime,
                final_metrics=final_metrics,
                communication_summary=comm_summary
            )
        else:
            err_msg = f"Federated process exited with code {returncode}."
            if stderr_path.exists():
                try:
                    with open(stderr_path, "r", encoding="utf-8") as f_err:
                        lines = f_err.readlines()
                        if lines:
                            err_msg += " " + "".join(lines[-5:]).strip()
                except Exception:
                    pass
            self.db.update_federated_experiment_status(
                exp_id,
                status="failed",
                finished_at=finish_time,
                runtime_s=runtime,
                final_metrics=final_metrics,
                communication_summary=comm_summary,
                error_message=err_msg[:500]
            )
