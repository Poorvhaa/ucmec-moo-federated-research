"""
Tests for CPU Allocation Optimization, Capacity Conservation, Residuals, and Solver Diagnostics.
"""
import pytest
import numpy as np
from moo.diagnostics import solve_cpu_allocation, SolverDiagnosticsTracker


def test_cpu_allocation_conservation():
    """
    Checks that allocated capacity sum strictly conserves total edge CPU capacity:
    sum(allocated_C) <= capacity + epsilon, and each fraction is strictly positive.
    """
    capacity = 15e9  # 15 GHz
    user_ids = [0, 1, 2]
    tasks = np.array([5e7, 3e7, 2e7])
    local_delays = np.array([0.0, 0.0, 0.0])
    uplink_delays = np.array([0.01, 0.015, 0.012])
    front_delays = np.array([0.005, 0.005, 0.005])

    tracker = SolverDiagnosticsTracker()
    alloc_c, diag = solve_cpu_allocation(
        cpu_id=0,
        capacity=capacity,
        user_ids=user_ids,
        tasks=tasks,
        local_delays=local_delays,
        uplink_delays=uplink_delays,
        front_delays=front_delays,
        use_corrected_cpu_alloc=True,
        solver_name="CLARABEL",
        tracker=tracker,
    )

    assert len(alloc_c) == 3
    assert np.all(alloc_c > 0)
    # Conservation of capacity
    assert pytest.approx(float(np.sum(alloc_c)), rel=1e-5) == capacity
    assert diag.is_optimal
    assert diag.residual_capacity < 1e-4
    assert diag.duration_s >= 0.0


def test_single_user_cpu_allocation():
    """When single user offloads, it trivially receives 100% of capacity."""
    capacity = 10e9
    alloc_c, diag = solve_cpu_allocation(
        cpu_id=1,
        capacity=capacity,
        user_ids=[4],
        tasks=np.array([4e7]),
        local_delays=np.array([0.0]),
        uplink_delays=np.array([0.01]),
        front_delays=np.array([0.005]),
        solver_name="CLARABEL",
    )
    assert len(alloc_c) == 1
    assert alloc_c[0] == capacity
    assert diag.solver_name == "ANALYTICAL_SINGLE"


def test_invalid_solver_inputs():
    """Validates that non-positive capacity or tasks throw ValueError."""
    with pytest.raises(ValueError, match="capacity"):
        solve_cpu_allocation(
            cpu_id=0,
            capacity=-5.0,
            user_ids=[0],
            tasks=np.array([1e7]),
            local_delays=np.array([0.0]),
            uplink_delays=np.array([0.01]),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        solve_cpu_allocation(
            cpu_id=0,
            capacity=10e9,
            user_ids=[0, 1],
            tasks=np.array([1e7, 0.0]),
            local_delays=np.array([0.0, 0.0]),
            uplink_delays=np.array([0.01, 0.01]),
        )
