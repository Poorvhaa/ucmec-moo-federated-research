"""
Solver Diagnostics, Residual Tracking, and Safe Allocation Optimization.
"""
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cvxpy as cp


@dataclass
class SolverCallDiagnostic:
    cpu_id: int
    solver_name: str
    status: str
    duration_s: float
    num_users: int
    capacity: float
    residual_capacity: float
    allocation_min: float
    allocation_max: float
    objective_value: Optional[float]
    is_optimal: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SolverDiagnosticsTracker:
    """Tracks solver executions, durations, statuses, and residuals."""

    def __init__(self, failure_dump_dir: Optional[str] = None):
        self.diagnostics_history: List[SolverCallDiagnostic] = []
        self.failure_dump_dir = Path(failure_dump_dir) if failure_dump_dir else None
        if self.failure_dump_dir:
            self.failure_dump_dir.mkdir(parents=True, exist_ok=True)

    def record(self, diagnostic: SolverCallDiagnostic):
        self.diagnostics_history.append(diagnostic)
        # Keep recent buffer manageable
        if len(self.diagnostics_history) > 1000:
            self.diagnostics_history = self.diagnostics_history[-500:]

    def get_summary(self) -> Dict[str, Any]:
        if not self.diagnostics_history:
            return {"total_solves": 0}
        
        statuses = {}
        durations = []
        residuals = []
        for d in self.diagnostics_history:
            statuses[d.status] = statuses.get(d.status, 0) + 1
            durations.append(d.duration_s)
            residuals.append(d.residual_capacity)
            
        return {
            "total_solves": len(self.diagnostics_history),
            "status_counts": statuses,
            "mean_duration_ms": float(np.mean(durations) * 1000.0) if durations else 0.0,
            "max_residual": float(np.max(residuals)) if residuals else 0.0,
            "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
        }

    def dump_failure_inputs(
        self,
        cpu_id: int,
        capacity: float,
        tasks: np.ndarray,
        local_delays: np.ndarray,
        uplink_delays: np.ndarray,
        front_delays: Optional[np.ndarray],
        status: str,
        message: str
    ) -> str:
        """Dumps problem inputs to disk on unrecoverable solve for diagnostics."""
        timestamp = int(time.time() * 1000)
        dump_data = {
            "timestamp": timestamp,
            "cpu_id": cpu_id,
            "capacity": float(capacity),
            "tasks": tasks.tolist(),
            "local_delays": local_delays.tolist(),
            "uplink_delays": uplink_delays.tolist(),
            "front_delays": front_delays.tolist() if front_delays is not None else [],
            "solver_status": status,
            "error_message": message,
        }
        dump_path = "failed_solver_inputs.json"
        if self.failure_dump_dir:
            dump_path = str(self.failure_dump_dir / f"solver_fail_cpu{cpu_id}_{timestamp}.json")
            with open(dump_path, "w") as f:
                json.dump(dump_data, f, indent=2)
        return dump_path


def solve_cpu_allocation(
    cpu_id: int,
    capacity: float,
    user_ids: List[int],
    tasks: np.ndarray,
    local_delays: np.ndarray,
    uplink_delays: np.ndarray,
    front_delays: Optional[np.ndarray] = None,
    use_corrected_cpu_alloc: bool = True,
    solver_name: str = "CLARABEL",
    tracker: Optional[SolverDiagnosticsTracker] = None,
) -> Tuple[np.ndarray, SolverCallDiagnostic]:
    """
    Solve edge CPU resource allocation:
    Minimizes sum of processing / offloading delays subject to sum(f_j) <= 1, f_j >= 0.

    Inputs are strictly validated for positivity and finiteness.
    Residuals and solver durations are logged.
    Does not fabricate allocations on failure.
    """
    k_users = len(user_ids)
    if k_users == 0:
        diag = SolverCallDiagnostic(
            cpu_id=cpu_id,
            solver_name=solver_name,
            status="EMPTY",
            duration_s=0.0,
            num_users=0,
            capacity=float(capacity),
            residual_capacity=0.0,
            allocation_min=0.0,
            allocation_max=0.0,
            objective_value=0.0,
            is_optimal=True,
        )
        return np.zeros(0, dtype=float), diag

    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError(f"CPU {cpu_id}: capacity must be finite and > 0, got {capacity}")

    tasks = np.asarray(tasks, dtype=float)
    local = np.asarray(local_delays, dtype=float)
    uplink = np.asarray(uplink_delays, dtype=float)

    if not (np.all(np.isfinite(tasks)) and np.all(np.isfinite(local)) and np.all(np.isfinite(uplink))):
        raise ValueError(f"CPU {cpu_id}: non-finite optimization inputs encountered")
    if np.any(tasks <= 0):
        raise ValueError(f"CPU {cpu_id}: task sizes must be strictly positive")

    # If single user, fraction is trivially 1.0 (100% CPU capacity)
    if k_users == 1:
        diag = SolverCallDiagnostic(
            cpu_id=cpu_id,
            solver_name="ANALYTICAL_SINGLE",
            status="OPTIMAL",
            duration_s=0.0,
            num_users=1,
            capacity=float(capacity),
            residual_capacity=0.0,
            allocation_min=1.0,
            allocation_max=1.0,
            objective_value=float(tasks[0] / capacity),
            is_optimal=True,
        )
        if tracker:
            tracker.record(diag)
        return np.array([capacity], dtype=float), diag

    # Formulation using capacity fractions f in (0, 1]
    fractions = cp.Variable(k_users, nonneg=True)
    # process_delay = tasks / (capacity * fractions)
    scaled_work = tasks / capacity
    process_delay = cp.multiply(scaled_work, cp.inv_pos(fractions))

    # Corrected vs Legacy formulation discrepancy handling:
    # Legacy: min sum(max(local, uplink + process_delay))
    # Corrected: min sum(max(local, front + uplink + process_delay))
    if use_corrected_cpu_alloc and front_delays is not None:
        front = np.asarray(front_delays, dtype=float)
        total_remote = front + uplink + process_delay
    else:
        total_remote = uplink + process_delay

    objective = cp.Minimize(cp.sum(cp.maximum(local, total_remote)))
    constraints = [cp.sum(fractions) <= 1.0]

    prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    solver_enum = getattr(cp, solver_name, cp.CLARABEL)
    try:
        prob.solve(solver=solver_enum, verbose=False)
    except Exception as e:
        duration = time.perf_counter() - t0
        err_msg = f"Solver {solver_name} threw exception: {str(e)}"
        if tracker:
            dump_file = tracker.dump_failure_inputs(
                cpu_id, capacity, tasks, local, uplink, front_delays, "EXCEPTION", err_msg
            )
            err_msg += f" (Inputs dumped to {dump_file})"
        raise RuntimeError(err_msg)

    duration = time.perf_counter() - t0

    if prob.status != cp.OPTIMAL or fractions.value is None:
        err_msg = f"CPU {cpu_id}: solver {solver_name} ended with status '{prob.status}'"
        if tracker:
            dump_file = tracker.dump_failure_inputs(
                cpu_id, capacity, tasks, local, uplink, front_delays, prob.status, err_msg
            )
            err_msg += f" (Diagnostic inputs saved to {dump_file})"
        raise RuntimeError(err_msg)

    raw_alloc = np.asarray(fractions.value, dtype=float).reshape(-1)
    
    # Residual checks
    residual_cap = abs(float(np.sum(raw_alloc)) - 1.0)
    alloc_min = float(np.min(raw_alloc))
    alloc_max = float(np.max(raw_alloc))

    # Numerical hygiene: check finiteness and within tolerance positivity
    if not np.all(np.isfinite(raw_alloc)):
        raise RuntimeError(f"CPU {cpu_id}: Non-finite allocation values returned by solver")
    if alloc_min < -1e-5 or np.sum(raw_alloc) > 1.0 + 1e-4:
        raise RuntimeError(
            f"CPU {cpu_id}: Infeasible allocation returned: min={alloc_min}, sum={np.sum(raw_alloc)}"
        )

    # Clean projection within tolerance: non-zero allocation
    clean_alloc = np.clip(raw_alloc, 1e-8, 1.0)
    clean_alloc /= float(np.sum(clean_alloc))

    diag = SolverCallDiagnostic(
        cpu_id=cpu_id,
        solver_name=solver_name,
        status=str(prob.status),
        duration_s=duration,
        num_users=k_users,
        capacity=float(capacity),
        residual_capacity=residual_cap,
        allocation_min=alloc_min,
        allocation_max=alloc_max,
        objective_value=float(prob.value) if prob.value is not None else None,
        is_optimal=True,
    )

    if tracker:
        tracker.record(diag)

    actual_allocated_cpu = capacity * clean_alloc
    return actual_allocated_cpu, diag
