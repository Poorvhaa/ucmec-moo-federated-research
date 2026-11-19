"""
Structured Metric Logging and Persistence for MOO Experiments.
"""
import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class ExperimentMetricLogger:
    """
    Handles incremental structured logging to JSON, CSV, and SummaryWriter.
    Guarantees partial results are retained even if a run is aborted or cancelled.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_config: Dict[str, Any],
        csv_filename: str = "metrics.csv",
        json_filename: str = "metrics.json",
        summary_filename: str = "moo_summary.json",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_config = experiment_config
        self.csv_path = self.log_dir / csv_filename
        self.json_path = self.log_dir / json_filename
        self.summary_path = self.log_dir / summary_filename

        self.history: List[Dict[str, Any]] = []
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            headers = [
                "step",
                "episode",
                "timestamp",
                "reward",
                "delay_ms",
                "energy_mj",
                "overhead_kb",
                "task_success_rate",
                "deadline_satisfaction_rate",
                "channel_success_rate",
                "local_users",
                "offloaded_users",
                "solver_mean_ms",
                "solver_max_residual"
            ]
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_episode(
        self,
        episode: int,
        total_steps: int,
        rewards: np.ndarray,
        infos: List[List[Dict[str, Any]]],
        writer=None
    ) -> Dict[str, float]:
        """Extracts metrics from episode infos and logs to CSV, JSON, and Writer."""
        delays = []
        energies = []
        overheads = []
        successes = []
        deadline_rates = []
        channel_rates = []
        local_counts = []
        offloaded_counts = []
        solver_durations = []
        solver_residuals = []

        def iter_agent_infos(item):
            if isinstance(item, dict):
                yield item
            elif isinstance(item, (list, tuple, np.ndarray)):
                for sub in item:
                    yield from iter_agent_infos(sub)

        for agent_info in iter_agent_infos(infos):
            if "delay_ms" in agent_info:
                delays.append(agent_info["delay_ms"])
            if "energy_mj" in agent_info:
                energies.append(agent_info["energy_mj"])
            if "overhead_bytes" in agent_info:
                overheads.append(agent_info["overhead_bytes"])
            if "task_success" in agent_info:
                successes.append(1.0 if agent_info["task_success"] else 0.0)

            step_metrics = agent_info.get("step_metrics", {})
            if step_metrics:
                if "deadline_satisfaction_rate" in step_metrics:
                    deadline_rates.append(step_metrics["deadline_satisfaction_rate"])
                if "channel_success_rate" in step_metrics:
                    channel_rates.append(step_metrics["channel_success_rate"])
                if "num_local_users" in step_metrics:
                    local_counts.append(step_metrics["num_local_users"])
                if "num_offloaded_users" in step_metrics:
                    offloaded_counts.append(step_metrics["num_offloaded_users"])

            solver_diags = agent_info.get("solver_diagnostics", [])
            for diag in solver_diags:
                if "duration_s" in diag:
                    solver_durations.append(diag["duration_s"] * 1000.0)
                if "residual_capacity" in diag:
                    solver_residuals.append(diag["residual_capacity"])

        mean_rew = float(np.mean(rewards))
        mean_delay = float(np.mean(delays)) if delays else 0.0
        mean_energy = float(np.mean(energies)) if energies else 0.0
        mean_overhead_b = float(np.mean(overheads)) if overheads else 0.0
        mean_overhead_kb = mean_overhead_b / 1024.0
        mean_success = float(np.mean(successes)) if successes else 0.0
        mean_deadline = float(np.mean(deadline_rates)) if deadline_rates else mean_success
        mean_channel = float(np.mean(channel_rates)) if channel_rates else 1.0
        mean_local = float(np.mean(local_counts)) if local_counts else 0.0
        mean_offload = float(np.mean(offloaded_counts)) if offloaded_counts else 0.0
        solver_ms = float(np.mean(solver_durations)) if solver_durations else 0.0
        solver_max_res = float(np.max(solver_residuals)) if solver_residuals else 0.0

        record = {
            "step": int(total_steps),
            "episode": int(episode),
            "timestamp": float(time.time()),
            "reward": mean_rew,
            "delay_ms": mean_delay,
            "energy_mj": mean_energy,
            "overhead_kb": mean_overhead_kb,
            "task_success_rate": mean_success,
            "deadline_satisfaction_rate": mean_deadline,
            "channel_success_rate": mean_channel,
            "local_users": mean_local,
            "offloaded_users": mean_offload,
            "solver_mean_ms": solver_ms,
            "solver_max_residual": solver_max_res,
        }

        self.history.append(record)

        # Append to CSV
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    record["step"],
                    record["episode"],
                    f"{record['timestamp']:.2f}",
                    f"{record['reward']:.6f}",
                    f"{record['delay_ms']:.4f}",
                    f"{record['energy_mj']:.4f}",
                    f"{record['overhead_kb']:.4f}",
                    f"{record['task_success_rate']:.4f}",
                    f"{record['deadline_satisfaction_rate']:.4f}",
                    f"{record['channel_success_rate']:.4f}",
                    f"{record['local_users']:.1f}",
                    f"{record['offloaded_users']:.1f}",
                    f"{record['solver_mean_ms']:.2f}",
                    f"{record['solver_max_residual']:.6e}",
                ])
        except Exception:
            pass

        # Write recent history to JSON atomically
        try:
            temp_json = self.json_path.with_suffix(".tmp")
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            temp_json.replace(self.json_path)
        except Exception:
            pass

        # SummaryWriter / TensorBoard scalars
        if writer is not None:
            try:
                writer.add_scalar("moo/reward", mean_rew, total_steps)
                writer.add_scalar("moo/delay_ms", mean_delay, total_steps)
                writer.add_scalar("moo/energy_mj", mean_energy, total_steps)
                writer.add_scalar("moo/overhead_kb", mean_overhead_kb, total_steps)
                writer.add_scalar("moo/task_success_rate", mean_success, total_steps)
                writer.add_scalar("moo/deadline_rate", mean_deadline, total_steps)
                writer.add_scalar("moo/channel_success_rate", mean_channel, total_steps)
                writer.add_scalar("solver/duration_ms", solver_ms, total_steps)
                writer.add_scalar("solver/max_residual", solver_max_res, total_steps)
            except Exception:
                pass

        return record

    def finalize_summary(
        self,
        status: str = "completed",
        runtime_s: float = 0.0,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generates comprehensive final summary JSON."""
        final_metrics = self.history[-1] if self.history else {}
        
        # Compute overall mean across recent updates
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        recent_avg = {}
        for key in ["reward", "delay_ms", "energy_mj", "overhead_kb", "task_success_rate"]:
            vals = [h[key] for h in recent if key in h]
            recent_avg[f"final_mean_{key}"] = float(np.mean(vals)) if vals else 0.0

        summary = {
            "status": status,
            "runtime_s": float(runtime_s),
            "total_episodes_recorded": len(self.history),
            "final_step": final_metrics.get("step", 0),
            "final_metrics": final_metrics,
            "recent_average_metrics": recent_avg,
            "config": self.experiment_config,
            **(extra_info or {})
        }

        try:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

        return summary
