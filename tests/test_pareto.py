"""
Tests for 4-Objective Non-Dominance, Pareto Filtering, and Scenario Compatibility.
"""
import pytest
from moo.pareto import dominates, find_nondominated_set, check_scenario_compatibility


def test_four_objective_dominance():
    # Min: delay, energy, overhead. Max: task_success_rate (reliability)
    # Point A is strictly better or equal to Point B in all 4 dimensions
    a = {"delay_ms": 30.0, "energy_mj": 0.5, "overhead_kb": 0.2, "task_success_rate": 0.95}
    b = {"delay_ms": 40.0, "energy_mj": 0.6, "overhead_kb": 0.3, "task_success_rate": 0.90}
    assert dominates(a, b) is True
    assert dominates(b, a) is False

    # Trade-off between delay and energy: neither dominates
    c = {"delay_ms": 25.0, "energy_mj": 0.8, "overhead_kb": 0.2, "task_success_rate": 0.95}
    assert dominates(a, c) is False
    assert dominates(c, a) is False


def test_find_nondominated_set():
    points = [
        {"name": "p1", "delay_ms": 50.0, "energy_mj": 1.0, "overhead_kb": 0.5, "task_success_rate": 0.80},
        {"name": "p2", "delay_ms": 40.0, "energy_mj": 0.8, "overhead_kb": 0.4, "task_success_rate": 0.85}, # dominates p1
        {"name": "p3", "delay_ms": 30.0, "energy_mj": 1.2, "overhead_kb": 0.4, "task_success_rate": 0.90}, # trade-off (best delay)
    ]
    nd_set = find_nondominated_set(points)
    by_name = {p["name"]: p["is_nondominated"] for p in nd_set}
    assert by_name["p1"] is False  # dominated by p2
    assert by_name["p2"] is True   # non-dominated
    assert by_name["p3"] is True   # non-dominated


def test_scenario_compatibility():
    cfg1 = {"num_agents": 10, "episode_length": 200, "scenario_name": "MyEnv"}
    cfg2 = {"num_agents": 10, "episode_length": 200, "scenario_name": "MyEnv"}
    compat, msg = check_scenario_compatibility(cfg1, cfg2)
    assert compat is True

    cfg3 = {"num_agents": 20, "episode_length": 200, "scenario_name": "MyEnv"}
    compat_diff, msg_diff = check_scenario_compatibility(cfg1, cfg3)
    assert compat_diff is False
    assert "num_agents" in msg_diff
