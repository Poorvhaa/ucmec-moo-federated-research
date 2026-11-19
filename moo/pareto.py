"""
Pareto Optimality, 4-Objective Non-Dominance, and Empirical Frontier Analysis.

Objectives directions:
- Delay: Minimize
- Energy: Minimize
- Communication Overhead: Minimize
- Reliability (Task Success Rate): Maximize
"""
from typing import List, Dict, Any, Tuple
import numpy as np


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """
    Checks if policy/evaluation point `a` dominates point `b` across all 4 objectives:
    Minimize: delay_ms, energy_mj, overhead_kb
    Maximize: reliability (task_success_rate)
    
    `a` dominates `b` iff `a` is no worse than `b` on all 4 objectives,
    and strictly better on at least one.
    """
    # a <= b for minimization objectives
    no_worse_delay = a["delay_ms"] <= b["delay_ms"]
    no_worse_energy = a["energy_mj"] <= b["energy_mj"]
    no_worse_overhead = a["overhead_kb"] <= b["overhead_kb"]
    # a >= b for maximization objective
    no_worse_rel = a["task_success_rate"] >= b["task_success_rate"]

    all_no_worse = no_worse_delay and no_worse_energy and no_worse_overhead and no_worse_rel

    strictly_better = (
        a["delay_ms"] < b["delay_ms"]
        or a["energy_mj"] < b["energy_mj"]
        or a["overhead_kb"] < b["overhead_kb"]
        or a["task_success_rate"] > b["task_success_rate"]
    )

    return all_no_worse and strictly_better


def find_nondominated_set(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes the empirical non-dominated set among tested configurations.
    Uses all 4 objectives simultaneously.
    
    Returns a copy of the input list with an added boolean flag: 'is_nondominated'.
    """
    n = len(points)
    if n == 0:
        return []

    # Filter out invalid / incomplete points
    valid_points = []
    for p in points:
        p_copy = dict(p)
        is_valid = True
        for key in ["delay_ms", "energy_mj", "overhead_kb", "task_success_rate"]:
            if key not in p_copy or not np.isfinite(p_copy[key]):
                is_valid = False
                break
        if is_valid:
            valid_points.append(p_copy)

    m = len(valid_points)
    is_dominated = [False] * m

    for i in range(m):
        for j in range(m):
            if i != j and not is_dominated[i]:
                if dominates(valid_points[j], valid_points[i]):
                    is_dominated[i] = True
                    break

    for i, p in enumerate(valid_points):
        p["is_nondominated"] = not is_dominated[i]

    return valid_points


def check_scenario_compatibility(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify whether two runs share compatible environment and scenario configurations.
    Rejects or flags comparisons across incompatible settings.
    """
    critical_keys = ["num_agents", "episode_length", "scenario_name"]
    for key in critical_keys:
        val_a = a.get(key)
        val_b = b.get(key)
        if val_a != val_b:
            return False, f"Incompatible scenario parameter '{key}': {val_a} vs {val_b}"
    return True, "Compatible"
