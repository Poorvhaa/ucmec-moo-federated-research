"""
Multi-Objective Optimization package for UCMEC-mmWave-Fronthaul.
"""
from moo.config import PRESETS, DEFAULT_REFERENCE_SCALES, DEFAULT_PHYSICAL_PARAMS
from moo.metrics import MetricContract, StepMetricSnapshot
from moo.scalarization import MultiObjectiveScalarizer, validate_and_normalize_weights
from moo.diagnostics import SolverDiagnosticsTracker, solve_cpu_allocation
from moo.pareto import find_nondominated_set, dominates, check_scenario_compatibility

__all__ = [
    "PRESETS",
    "DEFAULT_REFERENCE_SCALES",
    "DEFAULT_PHYSICAL_PARAMS",
    "MetricContract",
    "StepMetricSnapshot",
    "MultiObjectiveScalarizer",
    "validate_and_normalize_weights",
    "SolverDiagnosticsTracker",
    "solve_cpu_allocation",
    "find_nondominated_set",
    "dominates",
    "check_scenario_compatibility",
]
