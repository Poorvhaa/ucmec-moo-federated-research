"""
Tests for Multi-Objective Metric Contracts, Hand-Calculated Models, and Units.
"""
import pytest
import numpy as np
from moo.metrics import MetricContract


def test_metric_contract_hand_calculated_example():
    """
    Validates delay, energy, overhead, and reliability against exact hand calculations.
    """
    params = {
        "kappa_user": 1e-28,
        "kappa_edge": 0.5e-28,
        "task_deadline_s": 0.100,  # 100 ms
        "csi_pilot_bytes": 100,
        "offload_request_bytes": 50,
        "scheduling_grant_bytes": 50,
        "fronthaul_signaling_bytes": 100,
        "ack_bytes": 20, # Total fixed signaling = 320 bytes
        "channel_mode": "legacy",
    }
    contract = MetricContract(params=params)

    # 2 users:
    # User 0: Local execution (omega=0, p=0)
    # User 1: Offloaded execution (omega=1, p=0.05 W)
    omega = np.array([0, 1])
    p = np.array([0.0, 0.05])

    # Task properties:
    # User 0: size = 100,000 bits, density = 500 cycles/bit => 50,000,000 cycles
    # User 0 user CPU = 5e8 Hz (500 MHz)
    # Expected local delay = 50e6 / 5e8 = 0.100 s (100 ms)
    # Expected local energy = 1e-28 * 50e6 * (5e8)^2 = 1e-28 * 5e7 * 25e16 = 1.25e-3 J (1.25 mJ)
    # Expected overhead = 0 B
    # Deadline: 0.100s <= 0.100s => True

    # User 1: size = 80,000 bits, density = 600 cycles/bit => 48,000,000 cycles
    # Uplink delay = 0.020 s (20 ms)
    # Fronthaul delay = 0.010 s (10 ms)
    # Process delay = 0.030 s (30 ms)
    # Expected offloaded delay = 0.020 + 0.010 + 0.030 = 0.060 s (60 ms)
    # Expected tx energy = 0.05 W * 0.020 s = 0.001 J (1.0 mJ)
    # Expected overhead = 320 B signaling
    # Expected payload = 80,000 / 8 = 10,000 B
    # Deadline: 0.060s <= 0.100s => True

    local_delay = np.array([[0.100], [0.0]])
    uplink_delay = np.array([[0.0], [0.020]])
    front_delay = np.array([[0.0], [0.010]])
    process_delay = np.array([[0.0], [0.030]])
    task_size = np.array([100000, 80000])
    task_density = np.array([500, 600])
    c_user = np.array([5e8, 5e8])
    c_actual = np.zeros((2, 3))
    c_actual[1, 0] = 16e8  # 1.6 GHz allocated to user 1 on CPU 0

    snapshot = contract.compute_step_metrics(
        omega=omega,
        p=p,
        local_delay=local_delay,
        uplink_delay=uplink_delay,
        front_delay=front_delay,
        process_delay=process_delay,
        task_size_bits=task_size,
        task_density=task_density,
        c_user=c_user,
        c_actual=c_actual,
    )

    # Delay check:
    # Mean delay = (100ms + 60ms) / 2 = 80 ms (0.080 s)
    assert pytest.approx(snapshot.mean_delay_s, 1e-5) == 0.080
    assert pytest.approx(snapshot.mean_delay_ms, 1e-3) == 80.0

    # Energy check:
    # User 0: 1.25 mJ, User 1: 1.0 mJ => Mean = 1.125 mJ (0.001125 J)
    assert pytest.approx(snapshot.mean_energy_mj, 1e-3) == 1.125
    assert pytest.approx(snapshot.mean_energy_j, 1e-6) == 0.001125

    # Overhead check:
    # User 0: 0 B, User 1: 320 B => Mean = 160 B
    assert pytest.approx(snapshot.mean_overhead_bytes, 1e-3) == 160.0
    assert pytest.approx(snapshot.mean_overhead_kb, 1e-4) == 160.0 / 1024.0

    # Reliability check:
    # Both meet 100ms deadline => 1.0 task success rate
    assert snapshot.task_success_rate == 1.0
    assert snapshot.num_local_users == 1
    assert snapshot.num_offloaded_users == 1


def test_deadline_and_channel_outage():
    """Validates that deadline violation or SINR drop results in task failure."""
    params = {
        "task_deadline_s": 0.050,  # Strict 50 ms deadline
        "channel_mode": "sinr_outage",
        "access_sinr_outage_threshold": 1.0, # 0 dB threshold
    }
    contract = MetricContract(params=params)

    # User 0: delay = 60 ms (> 50 ms deadline) -> fails deadline
    # User 1: delay = 40 ms (<= 50 ms deadline), but SINR = 0.2 (< 1.0 threshold) -> fails channel
    # User 2: delay = 30 ms (<= 50 ms deadline), SINR = 5.0 (>= 1.0) -> succeeds!
    omega = np.array([0, 1, 1])
    p = np.array([0.0, 0.05, 0.05])
    local_delay = np.array([[0.060], [0.0], [0.0]])
    uplink_delay = np.array([[0.0], [0.020], [0.010]])
    front_delay = np.array([[0.0], [0.010], [0.010]])
    process_delay = np.array([[0.0], [0.010], [0.010]])
    task_size = np.array([50000, 50000, 50000])
    task_density = np.array([500, 500, 500])
    c_user = np.array([5e8, 5e8, 5e8])
    c_actual = np.array([[0, 0, 0], [1e9, 0, 0], [1e9, 0, 0]])
    sinr_access = np.array([[10.0], [0.2], [5.0]])

    snapshot = contract.compute_step_metrics(
        omega=omega,
        p=p,
        local_delay=local_delay,
        uplink_delay=uplink_delay,
        front_delay=front_delay,
        process_delay=process_delay,
        task_size_bits=task_size,
        task_density=task_density,
        c_user=c_user,
        c_actual=c_actual,
        sinr_access=sinr_access,
    )

    # Only User 2 succeeds: 1 out of 3 = 33.33%
    assert pytest.approx(snapshot.task_success_rate, 1e-3) == 1.0 / 3.0
    # Users 1 and 2 met deadline: 2 out of 3 = 66.67%
    assert pytest.approx(snapshot.deadline_satisfaction_rate, 1e-3) == 2.0 / 3.0
    # Users 0 (local) and 2 succeeded channel: 2 out of 3 = 66.67%
    assert pytest.approx(snapshot.channel_success_rate, 1e-3) == 2.0 / 3.0
