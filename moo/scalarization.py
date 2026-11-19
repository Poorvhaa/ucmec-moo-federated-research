"""
Multi-Objective Scalarization and Reward Formulation.

Implements the configurable scalarization:
cost = w_delay * (delay / delay_ref)
     + w_energy * (energy / energy_ref)
     + w_overhead * (overhead / overhead_ref)
     + w_reliability * (1 - task_success_rate)
reward = -cost
"""
from typing import Dict, Any, Tuple
import numpy as np
from moo.config import DEFAULT_REFERENCE_SCALES, PRESETS


def validate_and_normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Validate that all weights are nonnegative, finite, and have a positive sum.
    Normalizes weights to strictly sum to 1.0.
    """
    required_keys = ["w_delay", "w_energy", "w_overhead", "w_reliability"]
    normalized = {}
    for key in required_keys:
        val = weights.get(key, 0.0)
        if not np.isfinite(val):
            raise ValueError(f"Weight '{key}' must be finite, got {val}")
        if val < 0.0:
            raise ValueError(f"Weight '{key}' must be non-negative, got {val}")
        normalized[key] = float(val)

    total_weight = sum(normalized.values())
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be strictly positive")

    return {k: v / total_weight for k, v in normalized.items()}


class MultiObjectiveScalarizer:
    """
    Computes scalarized multi-objective costs and rewards using fixed reference scales.
    Preserves both raw values and normalized values.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        reference_scales: Dict[str, float] = None,
        clip_normalized_val: float = 10.0,
        mode: str = "moo"
    ):
        self.mode = mode
        self.weights = validate_and_normalize_weights(weights)
        self.reference_scales = {**DEFAULT_REFERENCE_SCALES, **(reference_scales or {})}
        self.clip_normalized_val = clip_normalized_val

    def compute_agent_rewards(
        self,
        per_user_delay_s: np.ndarray,      # [M]
        per_user_energy_j: np.ndarray,     # [M]
        per_user_overhead_b: np.ndarray,   # [M]
        per_user_success: np.ndarray,      # [M] bool/float
        tau_c: float = 0.1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute rewards for each agent.
        Returns:
            rewards: [M, 1] numpy array
            info_dict: dictionary containing normalized terms and components
        """
        M = len(per_user_delay_s)
        rewards = np.zeros([M, 1], dtype=np.float32)

        if self.mode == "legacy":
            # Original legacy formula: -0.9 * delay + 0.1 * (tau_c - delay)
            for i in range(M):
                rewards[i, 0] = -0.9 * per_user_delay_s[i] + 0.1 * (tau_c - per_user_delay_s[i])
            return rewards, {"mode": "legacy"}

        # Multi-objective normalized formulation
        t_ref = max(1e-6, float(self.reference_scales["delay_ref_s"]))
        e_ref = max(1e-6, float(self.reference_scales["energy_ref_j"]))
        o_ref = max(1e-6, float(self.reference_scales["overhead_ref_bytes"]))

        norm_delay = per_user_delay_s / t_ref
        norm_energy = per_user_energy_j / e_ref
        norm_overhead = per_user_overhead_b / o_ref
        norm_unreliability = 1.0 - np.asarray(per_user_success, dtype=float)

        # Clip normalized values for numerical safety without distorting raw metrics
        if self.clip_normalized_val > 0:
            c_delay = np.clip(norm_delay, 0.0, self.clip_normalized_val)
            c_energy = np.clip(norm_energy, 0.0, self.clip_normalized_val)
            c_overhead = np.clip(norm_overhead, 0.0, self.clip_normalized_val)
        else:
            c_delay, c_energy, c_overhead = norm_delay, norm_energy, norm_overhead

        cost = (
            self.weights["w_delay"] * c_delay
            + self.weights["w_energy"] * c_energy
            + self.weights["w_overhead"] * c_overhead
            + self.weights["w_reliability"] * norm_unreliability
        )
        
        rewards[:, 0] = -cost.astype(np.float32)

        diagnostics = {
            "mode": "moo",
            "weights": self.weights,
            "mean_norm_delay": float(np.mean(norm_delay)),
            "mean_norm_energy": float(np.mean(norm_energy)),
            "mean_norm_overhead": float(np.mean(norm_overhead)),
            "mean_norm_unreliability": float(np.mean(norm_unreliability)),
            "mean_cost": float(np.mean(cost)),
            "mean_reward": float(np.mean(rewards)),
        }

        return rewards, diagnostics
