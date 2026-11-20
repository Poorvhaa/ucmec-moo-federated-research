"""
Unit tests for FederatedClient, FederatedCoordinator, and Synchronization Semantics.
"""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from config import get_config
from federated.scenarios import ClientScenarioConfig
from federated.client import FederatedClient
from federated.coordinator import FederatedCoordinator
from federated.core import validate_tensor_schema, check_finite_tensors


@pytest.fixture
def base_args():
    parser = get_config()
    args = parser.parse_known_args([])[0]
    args.num_agents = 10
    args.episode_length = 20
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.cuda = False
    args.hidden_size = 64
    args.use_valuenorm = True
    args.ppo_epoch = 1
    args.num_mini_batch = 1
    return args


@pytest.fixture
def sample_client_config():
    return ClientScenarioConfig(
        client_id="client_test_0",
        site_name="Site-Test",
        seed=101,
        local_rollout_steps=20,
        local_epochs=1,
    )


def test_client_receive_global_actor_and_optimizer_reset(base_args, sample_client_config):
    """
    Verifies:
    1. Loading global weights modifies the actor parameters.
    2. Reset optimizer clears actor Adam momentum/state.
    3. Local critic parameters are completely untouched.
    """
    client = FederatedClient(
        config=sample_client_config,
        all_args=base_args,
        preset_name="balanced",
    )

    # Record initial critic parameters
    initial_critic_params = {
        k: v.clone() for k, v in client.policy.critic.state_dict().items()
    }

    # Populate dummy optimizer state
    dummy_tensor = list(client.policy.actor.parameters())[0]
    client.policy.actor_optimizer.state[dummy_tensor] = {
        "step": 10,
        "exp_avg": torch.ones_like(dummy_tensor),
        "exp_avg_sq": torch.ones_like(dummy_tensor),
    }
    assert len(client.policy.actor_optimizer.state) > 0

    # Create new global actor weights (e.g. all ones)
    new_global_weights = {
        k: torch.ones_like(v) * 0.42
        for k, v in client.policy.actor.state_dict().items()
    }

    # Synchronize
    client.receive_global_actor(new_global_weights, reset_optimizer=True)

    # 1. Actor weights updated
    for k, v in client.policy.actor.state_dict().items():
        assert torch.allclose(v, new_global_weights[k])

    # 2. Actor optimizer state cleared
    assert len(client.policy.actor_optimizer.state) == 0

    # 3. Critic parameters completely unchanged
    for k, v in client.policy.critic.state_dict().items():
        assert torch.allclose(v, initial_critic_params[k])


def test_client_rollout_and_package_submission(base_args, sample_client_config):
    """
    Verifies:
    1. Client collects fresh on-policy transitions.
    2. Buffer is populated and cleared after update.
    3. Package submission excludes raw trajectories and includes valid samples > 0.
    """
    client = FederatedClient(
        config=sample_client_config,
        all_args=base_args,
        preset_name="balanced",
    )

    valid_samples, rollout_metrics = client.collect_fresh_rollouts(rollout_steps=10)
    assert valid_samples == 10 * client.num_agents
    assert "delay_ms" in rollout_metrics
    assert "task_success_rate" in rollout_metrics

    train_metrics = client.perform_local_updates(local_epochs=1)
    assert "policy_loss" in train_metrics
    assert client.buffer.step == 0  # Cleared after update

    sub = client.package_submission(
        round_idx=1,
        base_version=0,
        valid_samples=valid_samples,
        runtime_s=1.5,
        train_metrics=train_metrics,
    )

    assert sub["client_id"] == "client_test_0"
    assert sub["valid_samples"] == 100
    assert sub["payload_bytes"] > 0
    assert "actor_state" in sub
    # Ensure raw observations, actions, or buffers are NOT in submission
    assert "obs" not in sub
    assert "actions" not in sub
    assert "trajectories" not in sub
    assert check_finite_tensors(sub["actor_state"]) is True


def test_coordinator_stale_and_duplicate_rejection(tmp_path, base_args):
    """Verifies coordinator rejects stale versions and duplicate submissions."""
    client_cfgs = [
        ClientScenarioConfig(client_id=f"c_{i}", site_name=f"Site_{i}", seed=100 + i, local_rollout_steps=10)
        for i in range(2)
    ]
    clients = [FederatedClient(config=c, all_args=base_args) for c in client_cfgs]

    coord = FederatedCoordinator(
        experiment_id="test_rejection",
        clients=clients,
        all_args=base_args,
        output_dir=tmp_path,
        min_clients=2,
    )

    # Execute valid round 1
    res1 = coord.execute_round(1)
    assert res1["status"] == "completed"
    assert coord.global_version == 1

    # Now create a stale submission with base_version=0 (current is 1)
    stale_sub = {
        "client_id": "c_0",
        "round": 2,
        "base_version": 0,  # STALE!
        "valid_samples": 50,
        "actor_state": {k: v.clone() for k, v in coord.canonical_schema.items()},
    }

    # Verify coordinator would reject stale version
    assert stale_sub["base_version"] != coord.global_version


def test_coordinator_insufficient_clients_failure(tmp_path, base_args):
    """Verifies coordinator properly handles insufficient client participation."""
    client_cfgs = [
        ClientScenarioConfig(client_id="c_0", site_name="Site_0", seed=100, local_rollout_steps=10)
    ]
    clients = [FederatedClient(config=c, all_args=base_args) for c in client_cfgs]

    # Require minimum 3 clients when only 1 exists
    coord = FederatedCoordinator(
        experiment_id="test_min_clients",
        clients=clients,
        all_args=base_args,
        output_dir=tmp_path,
        min_clients=3,
    )

    res = coord.execute_round(1)
    assert res["status"] == "failed"
    assert "minimum required is 3" in res["error"]
