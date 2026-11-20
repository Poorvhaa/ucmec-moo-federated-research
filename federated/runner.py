"""
Federated Multi-Agent Learning: CLI Entrypoint and Worker Execution Script.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
from config import get_config
from federated.scenarios import generate_federation_scenarios
from federated.client import FederatedClient
from federated.coordinator import FederatedCoordinator
from federated.benchmarks import run_full_comparative_study


def parse_federated_args(parser):
    group = parser.add_argument_group("Federated Learning Parameters")
    group.add_argument("--federated", action="store_true", default=True, help="Run federated training")
    group.add_argument("--fed_num_clients", type=int, default=3, help="Number of independent client sites")
    group.add_argument("--fed_num_rounds", type=int, default=2, help="Number of global communication rounds")
    group.add_argument("--fed_min_clients", type=int, default=1, help="Minimum accepted clients per round")
    group.add_argument("--fed_participation_rate", type=float, default=1.0, help="Client selection fraction")
    group.add_argument("--fed_dropout_prob", type=float, default=0.0, help="Simulated client dropout probability")
    group.add_argument(
        "--fed_scenario_type",
        type=str,
        default="iid",
        choices=[
            "iid",
            "task_heterogeneity",
            "channel_heterogeneity",
            "compute_heterogeneity",
            "sample_budget_heterogeneity",
            "full_heterogeneity",
        ],
        help="Heterogeneity profile across client sites",
    )
    group.add_argument(
        "--reset_optimizer_on_sync",
        action="store_true",
        default=True,
        help="Reset actor optimizer state upon receiving global weights",
    )
    group.add_argument("--fed_output_dir", type=str, default=None, help="Directory for federated artifacts")
    group.add_argument("--compare_methods", action="store_true", default=False, help="Run full comparison across 3 methods")
    return parser


def main():
    parser = get_config()
    parser = parse_federated_args(parser)
    # Additional MOO arguments
    parser.add_argument("--moo_preset", type=str, default="balanced", choices=["legacy", "balanced", "energy_priority", "reliability_priority", "custom"])
    parser.add_argument("--solver_name", type=str, default="CLARABEL", choices=["CLARABEL", "SCS"])
    parser.add_argument("--channel_mode", type=str, default="legacy", choices=["legacy", "sinr_outage"])
    parser.add_argument("--legacy_cpu_alloc", action="store_true", default=False)
    parser.add_argument("--w_delay", type=float, default=None)
    parser.add_argument("--w_energy", type=float, default=None)
    parser.add_argument("--w_overhead", type=float, default=None)
    parser.add_argument("--w_reliability", type=float, default=None)

    all_args = parser.parse_known_args()[0]

    # Output directory
    exp_name = getattr(all_args, "experiment_name", "federated_check")
    if all_args.fed_output_dir:
        output_dir = Path(all_args.fed_output_dir)
    else:
        output_dir = Path("experiments_storage") / "federated_runs" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f" Starting Federated Multi-Agent Learning")
    print(f" Clients: {all_args.fed_num_clients} | Rounds: {all_args.fed_num_rounds}")
    print(f" Scenario Type: {all_args.fed_scenario_type} | Preset: {all_args.moo_preset}")
    print(f" Output Dir: {output_dir}")
    print(f"============================================================")

    # Generate scenario configurations with independent seeds
    scenarios = generate_federation_scenarios(
        num_clients=all_args.fed_num_clients,
        scenario_type=all_args.fed_scenario_type,
        base_seed=all_args.seed,
        rollout_steps=all_args.episode_length,
        local_epochs=all_args.ppo_epoch if hasattr(all_args, "ppo_epoch") else 5,
    )

    if all_args.compare_methods:
        print("\n[Mode] Running Full Comparative Study (Local-Only vs Federated vs Pooled)...")
        results = run_full_comparative_study(
            all_args=all_args,
            scenarios=scenarios,
            num_rounds=all_args.fed_num_rounds,
            output_dir=output_dir,
            preset_name=all_args.moo_preset,
        )
        print("\nComparative Study Completed Successfully!")
        print(json.dumps(results["tradeoff_table"], indent=2))
        return

    # Standard Federated Training
    clients = [
        FederatedClient(
            config=sc,
            all_args=all_args,
            preset_name=all_args.moo_preset,
            use_corrected_cpu_alloc=not all_args.legacy_cpu_alloc,
            solver_name=all_args.solver_name,
            channel_mode=all_args.channel_mode,
        )
        for sc in scenarios
    ]

    coordinator = FederatedCoordinator(
        experiment_id=exp_name,
        clients=clients,
        all_args=all_args,
        output_dir=output_dir,
        min_clients=all_args.fed_min_clients,
        participation_rate=all_args.fed_participation_rate,
        reset_optimizer_on_sync=all_args.reset_optimizer_on_sync,
        dropout_prob=all_args.fed_dropout_prob,
        selection_seed=all_args.seed + 77,
        eval_seeds=[all_args.seed + 501, all_args.seed + 502],
        preset_name=all_args.moo_preset,
    )

    result = coordinator.train_federation(total_rounds=all_args.fed_num_rounds)
    print("\nFederated Learning Finished with status:", result["status"])
    print(f"Completed Rounds: {result['completed_rounds']}/{all_args.fed_num_rounds}")
    ledger_sum = result["ledger_summary"]
    print(f"Total Communication: {ledger_sum['cumulative_total_mb']} MB across {ledger_sum['total_transmissions']} transmissions")


if __name__ == "__main__":
    main()
