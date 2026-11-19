# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import time

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv

"""Train script for MPEs."""


def get_env_kwargs(all_args):
    moo_weights = None
    if any(getattr(all_args, k, None) is not None for k in ["w_delay", "w_energy", "w_overhead", "w_reliability"]):
        moo_weights = {
            "w_delay": getattr(all_args, "w_delay", 0.0) or 0.0,
            "w_energy": getattr(all_args, "w_energy", 0.0) or 0.0,
            "w_overhead": getattr(all_args, "w_overhead", 0.0) or 0.0,
            "w_reliability": getattr(all_args, "w_reliability", 0.0) or 0.0,
        }

    use_corrected = not getattr(all_args, "legacy_cpu_alloc", False)

    physical_params = {
        "task_deadline_s": getattr(all_args, "task_deadline_s", 0.1),
        "channel_mode": getattr(all_args, "channel_mode", "legacy"),
    }

    reference_scales = {
        "delay_ref_s": getattr(all_args, "delay_ref_s", 0.1),
        "energy_ref_j": getattr(all_args, "energy_ref_j", 0.005),
        "overhead_ref_bytes": getattr(all_args, "overhead_ref_bytes", 1000.0),
        "reliability_ref": 1.0,
    }

    return {
        "preset_name": getattr(all_args, "moo_preset", "legacy"),
        "moo_weights": moo_weights,
        "reference_scales": reference_scales,
        "physical_params": physical_params,
        "use_corrected_cpu_alloc": use_corrected,
        "max_steps": getattr(all_args, "episode_length", 200),
        "solver_name": getattr(all_args, "solver_name", "CLARABEL"),
    }


def make_train_env(all_args):
    env_kwargs = get_env_kwargs(all_args)
    def get_env_fn(rank):
        def init_env():
            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv(**env_kwargs)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    env_kwargs = get_env_kwargs(all_args)
    def get_env_fn(rank):
        def init_env():
            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv(**env_kwargs)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=10, help="number of players")

    # MOO parameters
    parser.add_argument(
        "--moo_preset",
        type=str,
        default="legacy",
        choices=["legacy", "delay_only", "balanced", "energy_priority", "reliability_priority", "custom"],
        help="MOO preset configuration",
    )
    parser.add_argument("--w_delay", type=float, default=None, help="Scalarization weight for delay")
    parser.add_argument("--w_energy", type=float, default=None, help="Scalarization weight for energy")
    parser.add_argument("--w_overhead", type=float, default=None, help="Scalarization weight for overhead")
    parser.add_argument("--w_reliability", type=float, default=None, help="Scalarization weight for reliability")
    parser.add_argument("--delay_ref_s", type=float, default=0.1, help="Reference delay scale (s)")
    parser.add_argument("--energy_ref_j", type=float, default=0.005, help="Reference energy scale (J)")
    parser.add_argument("--overhead_ref_bytes", type=float, default=1000.0, help="Reference overhead scale (B)")
    parser.add_argument("--task_deadline_s", type=float, default=0.1, help="Task deadline (s)")
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="legacy",
        choices=["legacy", "sinr_outage"],
        help="Channel reliability model mode",
    )
    parser.add_argument(
        "--legacy_cpu_alloc",
        action="store_true",
        default=False,
        help="Use uncorrected legacy CPU allocation omitting fronthaul delay",
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        default="CLARABEL",
        choices=["CLARABEL", "SCS"],
        help="Convex solver name",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
                all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
                   all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
           ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
            Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
            / all_args.env_name
            / all_args.scenario_name
            / all_args.algorithm_name
            / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/tb_summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    start = time.perf_counter()
    main(sys.argv[1:])
    end = time.perf_counter()
    runTime = end - start
    print("Running Time：", runTime, "s")
