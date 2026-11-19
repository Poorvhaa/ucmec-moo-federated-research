"""
Reproducible Policy Evaluation and MOO Comparison Workflow.

Evaluates policies on fixed held-out scenarios across independent seeds,
disabling stochastic exploration where applicable.
Reports individual objectives: Delay (ms), Energy (mJ), Overhead (kB), Reliability (task success rate).
Computes 4D empirical non-dominance and exports evaluation artifacts.
"""
import argparse
import json
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from envs.env_discrete import DiscreteActionEnv
from moo.config import PRESETS, DEFAULT_PHYSICAL_PARAMS, DEFAULT_REFERENCE_SCALES
from moo.pareto import find_nondominated_set, check_scenario_compatibility


def evaluate_policy_on_scenario(
    preset_name: str,
    seed: int,
    num_steps: int = 200,
    num_agents: int = 10,
    model_dir: Optional[str] = None,
    use_corrected_cpu_alloc: bool = True,
    solver_name: str = "CLARABEL",
    channel_mode: str = "legacy",
) -> Dict[str, Any]:
    """
    Evaluates a single scenario deterministically.
    Disables exploration when model checkpoint is provided.
    """
    preset_cfg = PRESETS.get(preset_name, PRESETS["balanced"])
    env = DiscreteActionEnv(
        preset_name=preset_name,
        moo_weights=preset_cfg.get("weights"),
        use_corrected_cpu_alloc=use_corrected_cpu_alloc,
        solver_name=solver_name,
        physical_params={"channel_mode": channel_mode},
        max_steps=num_steps,
    )
    env.seed(seed)
    obs = env.reset()

    # Load actor policies if model directory provided
    actors = []
    if model_dir and Path(model_dir).exists():
        from algorithms.algorithm.r_actor_critic import R_Actor
        from config import get_config
        parser = get_config()
        all_args = parser.parse_known_args([])[0]
        for a_idx in range(num_agents):
            actor_path = Path(model_dir) / f"actor_agent{a_idx}.pt"
            if actor_path.exists():
                actor = R_Actor(all_args, env.observation_space[a_idx], env.action_space[a_idx])
                actor.load_state_dict(torch.load(str(actor_path), map_location="cpu"))
                actor.eval()
                actors.append(actor)

    step_delays = []
    step_energies = []
    step_overheads = []
    step_successes = []
    step_rewards = []
    solver_durations = []
    solver_residuals = []

    rnn_states = np.zeros((num_agents, 1, 64), dtype=np.float32)
    masks = np.ones((num_agents, 1), dtype=np.float32)

    for step in range(num_steps):
        actions = []
        if len(actors) == num_agents:
            with torch.no_grad():
                for a_idx in range(num_agents):
                    act_out, rnn_out = actors[a_idx](
                        torch.from_numpy(obs[a_idx : a_idx + 1]).float(),
                        torch.from_numpy(rnn_states[a_idx : a_idx + 1]).float(),
                        torch.from_numpy(masks[a_idx : a_idx + 1]).float(),
                        deterministic=True,
                    )
                    rnn_states[a_idx] = rnn_out.numpy()[0]
                    act_int = act_out.numpy()[0, 0]
                    one_hot = np.zeros(env.action_space[a_idx].n)
                    one_hot[int(act_int)] = 1.0
                    actions.append(one_hot)
        else:
            # Deterministic pseudo-random baseline using step RNG
            rng = np.random.RandomState(seed * 10000 + step)
            for a_idx in range(num_agents):
                act_choice = rng.randint(0, env.action_space[a_idx].n)
                one_hot = np.zeros(env.action_space[a_idx].n)
                one_hot[act_choice] = 1.0
                actions.append(one_hot)

        actions_arr = np.array(actions)
        obs, rews, dones, infos = env.step(actions_arr)

        for a_info in infos:
            if isinstance(a_info, dict):
                step_delays.append(a_info.get("delay_ms", 0.0))
                step_energies.append(a_info.get("energy_mj", 0.0))
                step_overheads.append(a_info.get("overhead_bytes", 0.0))
                step_successes.append(1.0 if a_info.get("task_success", False) else 0.0)
                step_rewards.append(a_info.get("individual_reward", 0.0))

                for diag in a_info.get("solver_diagnostics", []):
                    solver_durations.append(diag.get("duration_s", 0.0) * 1000.0)
                    solver_residuals.append(diag.get("residual_capacity", 0.0))

        if np.all(dones):
            break

    mean_overhead_b = float(np.mean(step_overheads)) if step_overheads else 0.0

    return {
        "preset": preset_name,
        "seed": seed,
        "num_steps_completed": step + 1,
        "reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "delay_ms": float(np.mean(step_delays)) if step_delays else 0.0,
        "energy_mj": float(np.mean(step_energies)) if step_energies else 0.0,
        "overhead_kb": float(mean_overhead_b / 1024.0),
        "task_success_rate": float(np.mean(step_successes)) if step_successes else 0.0,
        "solver_mean_ms": float(np.mean(solver_durations)) if solver_durations else 0.0,
        "solver_max_residual": float(np.max(solver_residuals)) if solver_residuals else 0.0,
    }


def run_evaluation_suite(
    presets: List[str],
    seeds: List[int],
    num_steps: int = 200,
    num_agents: int = 10,
    model_dirs: Optional[Dict[str, str]] = None,
    output_dir: str = "experiments_storage/evaluations",
) -> Dict[str, Any]:
    """
    Runs multi-preset, multi-seed evaluation.
    Aggregates at the independent seed level (never treating correlated steps as independent).
    Computes empirical non-dominance and returns structured summary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_dirs = model_dirs or {}

    raw_runs = []
    aggregated_results = []

    print(f"\n=======================================================")
    print(f"Starting Multi-Objective Evaluation Suite ({len(presets)} presets, {len(seeds)} seeds)")
    print(f"=======================================================")

    for preset in presets:
        preset_runs = []
        for s in seeds:
            m_dir = model_dirs.get(preset)
            res = evaluate_policy_on_scenario(
                preset_name=preset,
                seed=s,
                num_steps=num_steps,
                num_agents=num_agents,
                model_dir=m_dir,
            )
            raw_runs.append(res)
            preset_runs.append(res)

        # Aggregate strictly across independent seeds
        delays = [r["delay_ms"] for r in preset_runs]
        energies = [r["energy_mj"] for r in preset_runs]
        overheads = [r["overhead_kb"] for r in preset_runs]
        successes = [r["task_success_rate"] for r in preset_runs]
        rewards = [r["reward"] for r in preset_runs]

        agg = {
            "preset": preset,
            "num_seeds": len(seeds),
            "seeds": seeds,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "delay_ms_mean": float(np.mean(delays)),
            "delay_ms_std": float(np.std(delays)),
            "energy_mj_mean": float(np.mean(energies)),
            "energy_mj_std": float(np.std(energies)),
            "overhead_kb_mean": float(np.mean(overheads)),
            "overhead_kb_std": float(np.std(overheads)),
            "task_success_rate_mean": float(np.mean(successes)),
            "task_success_rate_std": float(np.std(successes)),
            # Key names required for non-dominance calculation:
            "delay_ms": float(np.mean(delays)),
            "energy_mj": float(np.mean(energies)),
            "overhead_kb": float(np.mean(overheads)),
            "task_success_rate": float(np.mean(successes)),
        }
        aggregated_results.append(agg)

    # Compute 4D empirical non-dominance
    nondominated_results = find_nondominated_set(aggregated_results)

    # Export to JSON and CSV
    json_path = out_path / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "num_seeds": len(seeds),
                "presets_evaluated": presets,
                "nondominated_set": [r for r in nondominated_results if r.get("is_nondominated")],
                "all_aggregated_results": nondominated_results,
                "raw_runs": raw_runs,
            },
            f,
            indent=2,
        )

    csv_path = out_path / "evaluation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "preset",
            "is_nondominated",
            "delay_ms_mean",
            "delay_ms_std",
            "energy_mj_mean",
            "energy_mj_std",
            "overhead_kb_mean",
            "overhead_kb_std",
            "task_success_rate_mean",
            "task_success_rate_std",
            "reward_mean",
            "reward_std"
        ])
        for r in nondominated_results:
            writer.writerow([
                r["preset"],
                r.get("is_nondominated", False),
                f"{r['delay_ms_mean']:.3f}",
                f"{r['delay_ms_std']:.3f}",
                f"{r['energy_mj_mean']:.3f}",
                f"{r['energy_mj_std']:.3f}",
                f"{r['overhead_kb_mean']:.4f}",
                f"{r['overhead_kb_std']:.4f}",
                f"{r['task_success_rate_mean']:.4f}",
                f"{r['task_success_rate_std']:.4f}",
                f"{r['reward_mean']:.4f}",
                f"{r['reward_std']:.4f}",
            ])

    # Print summary table
    print("\n-----------------------------------------------------------------------------------------------------------------")
    print(f"{'Preset':<22} | {'NonDom':<6} | {'Delay (ms)':<15} | {'Energy (mJ)':<15} | {'Overhead (kB)':<15} | {'Reliability':<15}")
    print("-----------------------------------------------------------------------------------------------------------------")
    for r in nondominated_results:
        nd_str = "YES" if r.get("is_nondominated") else "NO"
        d_str = f"{r['delay_ms_mean']:.2f} ± {r['delay_ms_std']:.2f}"
        e_str = f"{r['energy_mj_mean']:.2f} ± {r['energy_mj_std']:.2f}"
        o_str = f"{r['overhead_kb_mean']:.3f} ± {r['overhead_kb_std']:.3f}"
        rel_str = f"{r['task_success_rate_mean']:.3f} ± {r['task_success_rate_std']:.3f}"
        print(f"{r['preset']:<22} | {nd_str:<6} | {d_str:<15} | {e_str:<15} | {o_str:<15} | {rel_str:<15}")
    print("-----------------------------------------------------------------------------------------------------------------\n")

    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "results": nondominated_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOO policies on fixed held-out scenarios.")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["legacy", "delay_only", "balanced", "energy_priority", "reliability_priority"],
        help="List of presets to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="List of evaluation seeds",
    )
    parser.add_argument("--num_steps", type=int, default=200, help="Steps per evaluation episode")
    parser.add_argument("--num_agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--output_dir", type=str, default="experiments_storage/evaluations")
    args = parser.parse_args()

    run_evaluation_suite(
        presets=args.presets,
        seeds=args.seeds,
        num_steps=args.num_steps,
        num_agents=args.num_agents,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
