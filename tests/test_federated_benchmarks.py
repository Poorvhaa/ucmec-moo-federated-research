"""
Integration tests for Federated Benchmark Suite: Local-Only vs Federated vs Pooled-Data Reference.
"""
from pathlib import Path
import pytest
from config import get_config
from federated.scenarios import ClientScenarioConfig
from federated.benchmarks import (
    run_local_only_benchmark,
    run_federated_benchmark,
    run_pooled_data_benchmark,
    run_full_comparative_study,
)


@pytest.fixture
def smoke_args():
    parser = get_config()
    args = parser.parse_known_args([])[0]
    args.num_agents = 10
    args.episode_length = 10
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.cuda = False
    args.hidden_size = 64
    args.use_valuenorm = True
    args.ppo_epoch = 1
    args.num_mini_batch = 1
    return args


@pytest.fixture
def smoke_scenarios():
    return [
        ClientScenarioConfig(
            client_id="client_smoke_0",
            site_name="Site-A",
            seed=201,
            local_rollout_steps=10,
            local_epochs=1,
        ),
        ClientScenarioConfig(
            client_id="client_smoke_1",
            site_name="Site-B",
            seed=202,
            local_rollout_steps=10,
            local_epochs=1,
        ),
    ]


def test_smoke_benchmarks_execution(tmp_path, smoke_args, smoke_scenarios):
    """
    Verifies that all 3 methods execute properly under matched conditions
    and output valid multi-objective metrics, fairness index, and communication bytes.
    """
    comp_dir = tmp_path / "study"
    results = run_full_comparative_study(
        all_args=smoke_args,
        scenarios=smoke_scenarios,
        num_rounds=1,
        output_dir=comp_dir,
        eval_seeds=[501],
        preset_name="balanced",
    )

    assert "methods" in results
    assert "local_only" in results["methods"]
    assert "federated_ppo" in results["methods"]
    assert "pooled_data_reference" in results["methods"]

    local_res = results["methods"]["local_only"]
    fed_res = results["methods"]["federated_ppo"]
    pooled_res = results["methods"]["pooled_data_reference"]

    # Invariants verification
    assert local_res["total_transitions"] == fed_res["total_transitions"] == pooled_res["total_transitions"]
    assert local_res["communication_bytes"] == 0
    assert fed_res["communication_bytes"] > 0
    assert pooled_res["communication_bytes"] > 0

    # Fairness & objectives
    assert 0.0 <= fed_res["jains_fairness_index"] <= 1.0
    assert fed_res["mean_delay_ms"] > 0.0
    assert fed_res["mean_energy_mj"] > 0.0

    # Tradeoff table verification
    assert len(results["tradeoff_table"]) == 3
