"""
Tests for Weight Validation, Normalization, and Multi-Objective Scalarization.
"""
import pytest
import numpy as np
from moo.scalarization import validate_and_normalize_weights, MultiObjectiveScalarizer


def test_weight_normalization_valid():
    weights = {"w_delay": 1.0, "w_energy": 2.0, "w_overhead": 1.0, "w_reliability": 0.0}
    norm = validate_and_normalize_weights(weights)
    assert pytest.approx(sum(norm.values()), 1e-6) == 1.0
    assert pytest.approx(norm["w_delay"], 1e-6) == 0.25
    assert pytest.approx(norm["w_energy"], 1e-6) == 0.50
    assert pytest.approx(norm["w_overhead"], 1e-6) == 0.25
    assert pytest.approx(norm["w_reliability"], 1e-6) == 0.0


def test_weight_normalization_invalid():
    # Negative weight
    with pytest.raises(ValueError, match="non-negative"):
        validate_and_normalize_weights({"w_delay": -1.0, "w_energy": 1.0, "w_overhead": 0.0, "w_reliability": 0.0})

    # Zero total sum
    with pytest.raises(ValueError, match="strictly positive"):
        validate_and_normalize_weights({"w_delay": 0.0, "w_energy": 0.0, "w_overhead": 0.0, "w_reliability": 0.0})

    # Non-finite weight
    with pytest.raises(ValueError, match="finite"):
        validate_and_normalize_weights({"w_delay": float("nan"), "w_energy": 1.0, "w_overhead": 0.0, "w_reliability": 0.0})


def test_scalarization_cost_and_reward():
    weights = {"w_delay": 0.4, "w_energy": 0.3, "w_overhead": 0.2, "w_reliability": 0.1}
    ref_scales = {
        "delay_ref_s": 0.1,         # 100 ms
        "energy_ref_j": 0.005,      # 5 mJ
        "overhead_ref_bytes": 1000, # 1000 B
        "reliability_ref": 1.0,
    }
    scalarizer = MultiObjectiveScalarizer(weights=weights, reference_scales=ref_scales, mode="moo")

    # Hand-crafted test user:
    # delay = 0.05s (50ms) -> norm_delay = 0.5
    # energy = 0.0025J (2.5mJ) -> norm_energy = 0.5
    # overhead = 500B -> norm_overhead = 0.5
    # success = True (1.0) -> unreliability = 0.0
    # Expected cost = 0.4*0.5 + 0.3*0.5 + 0.2*0.5 + 0.1*0.0 = 0.20 + 0.15 + 0.10 + 0.0 = 0.45
    # Expected reward = -0.45

    delays = np.array([0.05])
    energies = np.array([0.0025])
    overheads = np.array([500.0])
    successes = np.array([True])

    rewards, diag = scalarizer.compute_agent_rewards(delays, energies, overheads, successes)
    assert pytest.approx(rewards[0, 0], 1e-4) == -0.45
    assert pytest.approx(diag["mean_cost"], 1e-4) == 0.45


def test_legacy_mode_reward():
    scalarizer = MultiObjectiveScalarizer(
        weights={"w_delay": 1.0, "w_energy": 0.0, "w_overhead": 0.0, "w_reliability": 0.0},
        mode="legacy"
    )
    # Legacy: -0.9 * delay + 0.1 * (tau_c - delay)
    # For delay = 0.02s, tau_c = 0.1s:
    # -0.9 * 0.02 + 0.1 * (0.1 - 0.02) = -0.018 + 0.008 = -0.010
    delays = np.array([0.02])
    energies = np.array([0.0])
    overheads = np.array([0.0])
    successes = np.array([True])

    rewards, diag = scalarizer.compute_agent_rewards(delays, energies, overheads, successes, tau_c=0.1)
    assert pytest.approx(rewards[0, 0], 1e-5) == -0.010
    assert diag["mode"] == "legacy"
