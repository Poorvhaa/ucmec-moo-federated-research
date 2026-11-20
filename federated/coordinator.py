"""
Federated Multi-Agent Learning: Central Coordinator and Aggregation Engine.
"""
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import torch

from algorithms.algorithm.r_actor_critic import R_Actor
from config import get_config
from federated.client import FederatedClient
from federated.core import (
    sample_weighted_fedavg,
    validate_tensor_schema,
    check_finite_tensors,
    save_atomic_checkpoint,
    load_safe_checkpoint,
    compute_jains_fairness,
)
from federated.accounting import (
    CommunicationLedger,
    calculate_tensor_payload_bytes,
    calculate_json_metadata_bytes,
)
from federated.scenarios import ClientScenarioConfig, generate_federation_scenarios
from evaluate import evaluate_policy_on_scenario


class FederatedCoordinator:
    """
    Coordinates synchronous Federated PPO rounds across independent edge sites:
    - Manages versioned global actor checkpoints.
    - Selects participating clients per round.
    - Broadcasts global model parameters and records download communication bytes.
    - Enforces update validation: rejects stale, duplicate, incompatible, or non-finite updates.
    - Executes sample-weighted FedAvg aggregation.
    - Evaluates global checkpoints on held-out scenarios.
    - Atomically persists checkpoints and round manifests.
    """

    def __init__(
        self,
        experiment_id: str,
        clients: List[FederatedClient],
        all_args: Any,
        output_dir: Path,
        min_clients: int = 1,
        participation_rate: float = 1.0,
        reset_optimizer_on_sync: bool = True,
        dropout_prob: float = 0.0,
        selection_seed: int = 42,
        eval_seeds: Optional[List[int]] = None,
        preset_name: str = "balanced",
        on_round_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.experiment_id = experiment_id
        self.clients = {c.client_id: c for c in clients}
        self.client_list = clients
        self.all_args = all_args
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.min_clients = min_clients
        self.participation_rate = max(0.1, min(1.0, participation_rate))
        self.reset_optimizer_on_sync = reset_optimizer_on_sync
        self.dropout_prob = max(0.0, min(0.9, dropout_prob))
        self.selection_rng = np.random.RandomState(selection_seed)
        self.eval_seeds = eval_seeds or [201, 202, 203]
        self.preset_name = preset_name
        self.on_round_progress = on_round_progress

        # Communication ledger
        self.ledger = CommunicationLedger()

        # Initialize canonical global actor model
        sample_client = self.client_list[0]
        self.global_actor = R_Actor(
            self.all_args,
            sample_client.obs_space,
            sample_client.act_space,
            device=torch.device("cpu"),
        )
        self.global_version = 0
        self.canonical_schema = {
            k: v.detach().cpu().clone()
            for k, v in self.global_actor.state_dict().items()
        }

        # Save initial checkpoint (version 0)
        self.initial_checkpoint_path = self.checkpoint_dir / "global_actor_round_0.pt"
        save_atomic_checkpoint(self.canonical_schema, self.initial_checkpoint_path)

        self.cancelled = False
        self.completed_rounds = 0
        self.round_history: List[Dict[str, Any]] = []

    def cancel(self):
        """Signals graceful cancellation of ongoing training."""
        self.cancelled = True

    def select_clients(self, round_idx: int) -> List[str]:
        """Selects eligible client IDs for the round."""
        all_ids = sorted(list(self.clients.keys()))
        if self.participation_rate >= 1.0 or len(all_ids) <= self.min_clients:
            return all_ids

        k = max(self.min_clients, int(round(len(all_ids) * self.participation_rate)))
        k = min(k, len(all_ids))
        selected = list(self.selection_rng.choice(all_ids, size=k, replace=False))
        return sorted(selected)

    def execute_round(self, round_idx: int) -> Dict[str, Any]:
        """
        Executes a single synchronous federated round:
        1. Select clients
        2. Broadcast global model
        3. Local rollout & PPO training at each site
        4. Validate submissions
        5. FedAvg aggregation
        6. Atomic checkpoint & held-out evaluation
        """
        if self.cancelled:
            return {"status": "cancelled", "round": round_idx}

        round_start_time = time.time()
        base_version = self.global_version
        selected_ids = self.select_clients(round_idx)

        # Current global actor weights
        current_global_weights = {
            k: v.detach().cpu().clone()
            for k, v in self.global_actor.state_dict().items()
        }
        broadcast_payload_bytes = calculate_tensor_payload_bytes(current_global_weights)

        # 1. Broadcast model to selected clients
        for cid in selected_ids:
            client = self.clients[cid]
            client.receive_global_actor(
                current_global_weights,
                reset_optimizer=self.reset_optimizer_on_sync,
            )
            # Log broadcast download
            self.ledger.record_transmission(
                round_idx=round_idx,
                client_id=cid,
                direction="download",
                payload_bytes=broadcast_payload_bytes,
                protocol_bytes=128,
                success=True,
                notes="Global actor broadcast",
            )

        # 2. Local rollout & updates per client
        submissions = []
        client_statuses = {}

        for cid in selected_ids:
            if self.cancelled:
                return {"status": "cancelled", "round": round_idx}

            client = self.clients[cid]
            client_start = time.time()

            # Check simulated dropout
            if self.dropout_prob > 0.0 and self.selection_rng.rand() < self.dropout_prob:
                # Client dropped out during transmission
                client_statuses[cid] = "dropped"
                self.ledger.record_transmission(
                    round_idx=round_idx,
                    client_id=cid,
                    direction="upload",
                    payload_bytes=broadcast_payload_bytes,
                    protocol_bytes=64,
                    success=False,
                    notes="Simulated client dropout",
                )
                continue

            try:
                # Fresh rollout collection
                valid_samples, rollout_metrics = client.collect_fresh_rollouts()

                # Local PPO updates
                train_metrics = client.perform_local_updates()
                client_duration = time.time() - client_start

                # Package update
                sub = client.package_submission(
                    round_idx=round_idx,
                    base_version=base_version,
                    valid_samples=valid_samples,
                    runtime_s=client_duration,
                    train_metrics=train_metrics,
                )

                # Log upload
                self.ledger.record_transmission(
                    round_idx=round_idx,
                    client_id=cid,
                    direction="upload",
                    payload_bytes=sub["payload_bytes"],
                    protocol_bytes=calculate_json_metadata_bytes(sub["metrics"]),
                    success=True,
                    notes="Client model update",
                )

                submissions.append(sub)
                client_statuses[cid] = "accepted"

            except Exception as e:
                client_statuses[cid] = f"failed: {str(e)}"
                self.ledger.record_transmission(
                    round_idx=round_idx,
                    client_id=cid,
                    direction="upload",
                    payload_bytes=0,
                    protocol_bytes=64,
                    success=False,
                    notes=f"Client execution failed: {str(e)}",
                )

        # 3. Validation
        valid_updates = []
        for sub in submissions:
            cid = sub["client_id"]
            # Reject stale version
            if sub["base_version"] != base_version:
                client_statuses[cid] = f"rejected: stale version {sub['base_version']} != {base_version}"
                continue

            # Reject duplicate update
            if any(u["client_id"] == cid for u in valid_updates):
                client_statuses[cid] = "rejected: duplicate submission"
                continue

            # Reject zero or negative samples
            if sub["valid_samples"] <= 0:
                client_statuses[cid] = "rejected: invalid sample count <= 0"
                continue

            # Validate tensor schema
            is_valid, err = validate_tensor_schema(sub["actor_state"], self.canonical_schema)
            if not is_valid:
                client_statuses[cid] = f"rejected: schema mismatch ({err})"
                continue

            # Check finite numbers
            if not check_finite_tensors(sub["actor_state"]):
                client_statuses[cid] = "rejected: non-finite tensors"
                continue

            valid_updates.append(sub)

        # Check minimum accepted clients threshold
        if len(valid_updates) < self.min_clients:
            err_msg = (
                f"Round {round_idx} failed: only {len(valid_updates)} valid updates, "
                f"minimum required is {self.min_clients}."
            )
            return {
                "round": round_idx,
                "status": "failed",
                "error": err_msg,
                "client_statuses": client_statuses,
            }

        # 4. FedAvg Aggregation
        new_global_state = sample_weighted_fedavg(
            valid_updates,
            reference_state=self.canonical_schema,
            computation_dtype=torch.float64,
        )

        # Update global actor
        self.global_actor.load_state_dict(new_global_state)
        self.global_version = round_idx

        # 5. Atomic checkpoint persistence
        checkpoint_path = self.checkpoint_dir / f"global_actor_round_{round_idx}.pt"
        save_atomic_checkpoint(new_global_state, checkpoint_path)

        # 6. Evaluation on held-out scenarios
        eval_metrics = self.evaluate_checkpoint(checkpoint_path)

        round_duration = time.time() - round_start_time

        # Calculate sample weights for logging
        total_samples = sum(u["valid_samples"] for u in valid_updates)
        client_weights = {
            u["client_id"]: float(u["valid_samples"]) / float(total_samples)
            for u in valid_updates
        }

        # Aggregate client-reported local metrics
        mean_client_delay = float(np.mean([u["metrics"]["delay_ms"] for u in valid_updates]))
        mean_client_energy = float(np.mean([u["metrics"]["energy_mj"] for u in valid_updates]))
        mean_client_overhead = float(np.mean([u["metrics"]["overhead_kb"] for u in valid_updates]))
        mean_client_success = float(np.mean([u["metrics"]["task_success_rate"] for u in valid_updates]))
        worst_client_success = float(np.min([u["metrics"]["task_success_rate"] for u in valid_updates]))
        worst_client_delay = float(np.max([u["metrics"]["delay_ms"] for u in valid_updates]))

        # Jain's fairness across client success rates
        jains_fairness = compute_jains_fairness([u["metrics"]["task_success_rate"] for u in valid_updates])

        comm_summary = self.ledger.get_round_summary(round_idx)

        round_manifest = {
            "round": round_idx,
            "global_version": self.global_version,
            "status": "completed",
            "runtime_s": round_duration,
            "participating_clients": selected_ids,
            "accepted_clients": [u["client_id"] for u in valid_updates],
            "client_statuses": client_statuses,
            "client_weights": client_weights,
            "total_valid_samples": total_samples,
            "local_training_metrics": {
                "mean_delay_ms": mean_client_delay,
                "mean_energy_mj": mean_client_energy,
                "mean_overhead_kb": mean_client_overhead,
                "mean_task_success_rate": mean_client_success,
                "worst_task_success_rate": worst_client_success,
                "worst_delay_ms": worst_client_delay,
                "jains_fairness_index": jains_fairness,
            },
            "eval_metrics": eval_metrics,
            "communication": comm_summary,
            "checkpoint_path": str(checkpoint_path.name),
        }

        manifest_path = self.checkpoint_dir / f"round_{round_idx}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(round_manifest, f, indent=2)

        self.round_history.append(round_manifest)
        self.completed_rounds = round_idx

        if self.on_round_progress:
            self.on_round_progress(round_manifest)

        return round_manifest

    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict[str, float]:
        """
        Evaluates a checkpoint deterministically on held-out test scenarios across independent seeds.
        """
        eval_results = []
        for seed in self.eval_seeds:
            res = evaluate_policy_on_scenario(
                preset_name=self.preset_name,
                seed=seed,
                num_steps=100,
                num_agents=self.client_list[0].num_agents,
                model_dir=str(checkpoint_path),
                use_corrected_cpu_alloc=True,
                solver_name="CLARABEL",
                channel_mode="legacy",
            )
            eval_results.append(res)

        return {
            "eval_delay_ms": float(np.mean([r["delay_ms"] for r in eval_results])),
            "eval_energy_mj": float(np.mean([r["energy_mj"] for r in eval_results])),
            "eval_overhead_kb": float(np.mean([r["overhead_kb"] for r in eval_results])),
            "eval_task_success_rate": float(np.mean([r["task_success_rate"] for r in eval_results])),
            "eval_reward": float(np.mean([r["reward"] for r in eval_results])),
        }

    def train_federation(self, total_rounds: int) -> Dict[str, Any]:
        """Runs the entire federated learning loop for total_rounds."""
        for r in range(1, total_rounds + 1):
            if self.cancelled:
                break
            result = self.execute_round(r)
            if result.get("status") == "failed":
                return {
                    "status": "failed",
                    "failed_round": r,
                    "error": result.get("error"),
                    "round_history": self.round_history,
                    "ledger_summary": self.ledger.get_summary(),
                }

        # Export full communication ledger
        ledger_path = self.output_dir / "communication_ledger.json"
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": self.ledger.get_summary(),
                "records": self.ledger.to_list(),
            }, f, indent=2)

        return {
            "status": "completed" if not self.cancelled else "cancelled",
            "completed_rounds": self.completed_rounds,
            "round_history": self.round_history,
            "ledger_summary": self.ledger.get_summary(),
        }
