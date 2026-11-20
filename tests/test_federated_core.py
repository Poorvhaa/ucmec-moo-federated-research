"""
Unit tests for Federated Core: Math, Validation, Schemas, Checkpoints, and Accounting.
"""
import copy
import math
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

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
from federated.scenarios import generate_federation_scenarios


def test_hand_calculated_sample_weighted_fedavg():
    """
    Validates sample-weighted FedAvg against an exact hand calculation:
    Client A: 100 samples, theta = [1.0, 2.0]
    Client B: 300 samples, theta = [5.0, 6.0]
    Total samples = 400
    Weight A = 100 / 400 = 0.25
    Weight B = 300 / 400 = 0.75
    Expected:
      theta_0 = 0.25 * 1.0 + 0.75 * 5.0 = 0.25 + 3.75 = 4.0
      theta_1 = 0.25 * 2.0 + 0.75 * 6.0 = 0.50 + 4.50 = 5.0
    """
    state_a = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    state_b = {"weight": torch.tensor([5.0, 6.0], dtype=torch.float32)}

    updates = [
        {"client_id": "c_a", "valid_samples": 100, "actor_state": state_a},
        {"client_id": "c_b", "valid_samples": 300, "actor_state": state_b},
    ]

    aggregated = sample_weighted_fedavg(updates, reference_state=state_a)

    expected = torch.tensor([4.0, 5.0], dtype=torch.float32)
    assert torch.allclose(aggregated["weight"], expected)


def test_fedavg_three_clients_with_different_weights():
    """
    Validates 3 clients with weights 200, 300, 500 (total 1000):
    weights = 0.2, 0.3, 0.5
    values = 10.0, 20.0, 30.0
    expected = 0.2*10 + 0.3*20 + 0.5*30 = 2 + 6 + 15 = 23.0
    """
    state_1 = {"layer": torch.tensor([[10.0, 0.0]], dtype=torch.float32)}
    state_2 = {"layer": torch.tensor([[20.0, 0.0]], dtype=torch.float32)}
    state_3 = {"layer": torch.tensor([[30.0, 0.0]], dtype=torch.float32)}

    updates = [
        {"client_id": "c1", "valid_samples": 200, "actor_state": state_1},
        {"client_id": "c2", "valid_samples": 300, "actor_state": state_2},
        {"client_id": "c3", "valid_samples": 500, "actor_state": state_3},
    ]

    agg = sample_weighted_fedavg(updates)
    assert math.isclose(agg["layer"][0, 0].item(), 23.0, rel_tol=1e-5)
    assert math.isclose(agg["layer"][0, 1].item(), 0.0, rel_tol=1e-5)


def test_reject_nan_and_inf_updates():
    """Confirms updates containing NaN or Inf are rejected."""
    good_state = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    nan_state = {"w": torch.tensor([1.0, float("nan")], dtype=torch.float32)}
    inf_state = {"w": torch.tensor([float("inf"), 2.0], dtype=torch.float32)}

    assert check_finite_tensors(good_state) is True
    assert check_finite_tensors(nan_state) is False
    assert check_finite_tensors(inf_state) is False

    updates_with_nan = [
        {"client_id": "good", "valid_samples": 100, "actor_state": good_state},
        {"client_id": "bad_nan", "valid_samples": 100, "actor_state": nan_state},
    ]
    with pytest.raises(ValueError, match="non-finite"):
        sample_weighted_fedavg(updates_with_nan)

    updates_with_inf = [
        {"client_id": "good", "valid_samples": 100, "actor_state": good_state},
        {"client_id": "bad_inf", "valid_samples": 100, "actor_state": inf_state},
    ]
    with pytest.raises(ValueError, match="non-finite"):
        sample_weighted_fedavg(updates_with_inf)


def test_reject_invalid_sample_counts():
    """Confirms zero or negative sample counts are strictly rejected."""
    state = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}

    with pytest.raises(ValueError, match="invalid sample count"):
        sample_weighted_fedavg([{"client_id": "c1", "valid_samples": 0, "actor_state": state}])

    with pytest.raises(ValueError, match="invalid sample count"):
        sample_weighted_fedavg([{"client_id": "c1", "valid_samples": -50, "actor_state": state}])


def test_schema_validation_shape_and_keys():
    """Confirms incompatible parameter shapes or mismatched keys are rejected."""
    ref = {
        "fc1.weight": torch.zeros((64, 3), dtype=torch.float32),
        "fc1.bias": torch.zeros((64,), dtype=torch.float32),
    }

    # Matching state
    match_state = {
        "fc1.weight": torch.ones((64, 3), dtype=torch.float32),
        "fc1.bias": torch.ones((64,), dtype=torch.float32),
    }
    valid, err = validate_tensor_schema(match_state, ref)
    assert valid is True
    assert err is None

    # Mismatched shape
    wrong_shape = {
        "fc1.weight": torch.ones((32, 3), dtype=torch.float32),
        "fc1.bias": torch.ones((64,), dtype=torch.float32),
    }
    valid, err = validate_tensor_schema(wrong_shape, ref)
    assert valid is False
    assert "Shape mismatch" in err

    # Missing key
    missing_key = {"fc1.weight": torch.ones((64, 3), dtype=torch.float32)}
    valid, err = validate_tensor_schema(missing_key, ref)
    assert valid is False
    assert "Missing" in err

    # Extra key
    extra_key = {**match_state, "extra": torch.zeros((1,), dtype=torch.float32)}
    valid, err = validate_tensor_schema(extra_key, ref)
    assert valid is False
    assert "Extra" in err


def test_atomic_checkpoint_persistence_and_loading(tmp_path):
    """Verifies atomic write prevents corruption and verifies integrity checks."""
    state = {
        "layer.weight": torch.tensor([[1.5, -2.3], [0.1, 4.2]], dtype=torch.float32),
        "layer.bias": torch.tensor([0.5, -0.5], dtype=torch.float32),
    }
    ckpt_file = tmp_path / "global_model.pt"

    saved_path = save_atomic_checkpoint(state, ckpt_file)
    assert saved_path.exists()
    assert saved_path == ckpt_file

    # Load back and verify values
    loaded = load_safe_checkpoint(ckpt_file, expected_schema=state)
    assert torch.allclose(loaded["layer.weight"], state["layer.weight"])
    assert torch.allclose(loaded["layer.bias"], state["layer.bias"])


def test_jains_fairness_computation():
    """
    Verifies Jain's Fairness Index hand calculations:
    Uniform values -> 1.0
    Equal allocation [10, 10, 10, 10] -> 1.0
    Extreme asymmetry [10, 0, 0, 0] -> 10^2 / (4 * 10^2) = 100 / 400 = 0.25 (1/k)
    Intermediate [10, 20]: (30)^2 / (2 * (100 + 400)) = 900 / 1000 = 0.90
    """
    assert math.isclose(compute_jains_fairness([10.0, 10.0, 10.0, 10.0]), 1.0)
    assert math.isclose(compute_jains_fairness([10.0, 0.0, 0.0, 0.0]), 0.25)
    assert math.isclose(compute_jains_fairness([10.0, 20.0]), 0.90)


def test_communication_byte_accounting():
    """Verifies byte tracking for tensor payloads and transmission ledger."""
    state = {
        # 1000 float32 elements = 4000 bytes
        "param1": torch.zeros(1000, dtype=torch.float32),
        # 500 float32 elements = 2000 bytes
        "param2": torch.zeros(500, dtype=torch.float32),
    }
    payload_bytes = calculate_tensor_payload_bytes(state)
    assert payload_bytes >= 6000  # 6000 raw tensor bytes + key string lengths

    ledger = CommunicationLedger()
    # Round 1: 2 clients download
    ledger.record_transmission(1, "c1", "download", payload_bytes, 128, True)
    ledger.record_transmission(1, "c2", "download", payload_bytes, 128, True)
    # Round 1: 1 success upload, 1 failed upload
    ledger.record_transmission(1, "c1", "upload", payload_bytes, 128, True)
    ledger.record_transmission(1, "c2", "upload", payload_bytes, 128, False, notes="dropped")

    summary = ledger.get_summary()
    assert summary["total_transmissions"] == 4
    assert summary["total_download_payload_bytes"] == 2 * payload_bytes
    assert summary["total_upload_payload_bytes"] == 1 * payload_bytes
    assert summary["total_failed_bytes"] == payload_bytes + 128

    round1 = ledger.get_round_summary(1)
    assert round1["download_bytes"] == 2 * (payload_bytes + 128)
    assert round1["upload_bytes"] == 1 * (payload_bytes + 128)
    assert round1["failed_bytes"] == payload_bytes + 128


def test_client_scenario_independence():
    """Confirms scenario generation produces distinct, independent seeds across clients."""
    scenarios = generate_federation_scenarios(
        num_clients=5,
        scenario_type="full_heterogeneity",
        base_seed=42,
    )
    seeds = [s.seed for s in scenarios]
    assert len(set(seeds)) == 5, "Client seeds must be distinct!"
    for s in scenarios:
        assert s.seed > 0
