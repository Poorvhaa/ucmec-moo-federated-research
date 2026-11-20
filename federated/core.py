"""
Federated Multi-Agent Learning: Core Aggregation, Validation, and Checkpoint Utilities.
"""
import copy
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def check_finite_tensors(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Verifies that all tensors in state_dict contain only finite values (no NaN or Inf)."""
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if not torch.isfinite(tensor).all():
            return False
    return True


def validate_tensor_schema(
    candidate_state: Dict[str, torch.Tensor],
    reference_state: Dict[str, torch.Tensor],
) -> Tuple[bool, Optional[str]]:
    """
    Validates that candidate_state strictly matches reference_state in:
    - Set of parameter names
    - Tensor dimensions/shapes
    - Floating point data types
    """
    cand_keys = set(candidate_state.keys())
    ref_keys = set(reference_state.keys())

    if cand_keys != ref_keys:
        missing = ref_keys - cand_keys
        extra = cand_keys - ref_keys
        return False, f"Key mismatch. Missing: {missing}, Extra: {extra}"

    for name, ref_tensor in reference_state.items():
        cand_tensor = candidate_state[name]
        if not torch.is_tensor(cand_tensor):
            return False, f"Value for '{name}' is not a torch.Tensor"
        if cand_tensor.shape != ref_tensor.shape:
            return (
                False,
                f"Shape mismatch for '{name}': expected {ref_tensor.shape}, got {cand_tensor.shape}",
            )
        if cand_tensor.dtype != ref_tensor.dtype:
            return (
                False,
                f"Dtype mismatch for '{name}': expected {ref_tensor.dtype}, got {cand_tensor.dtype}",
            )
        if not cand_tensor.is_floating_point():
            return False, f"Non-floating point tensor '{name}' found in trainable state dict"

    return True, None


def sample_weighted_fedavg(
    client_updates: List[Dict[str, Any]],
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    computation_dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """
    Executes sample-weighted FedAvg aggregation across validated client updates:
        theta_global = sum(n_k * theta_k) / sum(n_k)

    :param client_updates: List of dicts, each containing:
        - "actor_state": Dict[str, torch.Tensor] (client parameters)
        - "valid_samples": int (n_k > 0)
    :param reference_state: Optional reference state dict for schema verification.
    :param computation_dtype: High-precision dtype (default float64) to prevent numerical cancellation.
    :return: Aggregated global actor state dictionary in float32.
    """
    if not client_updates:
        raise ValueError("Cannot perform FedAvg on an empty list of client updates.")

    total_samples = 0
    for i, update in enumerate(client_updates):
        samples = update.get("valid_samples", 0)
        if not isinstance(samples, (int, np.integer)) or samples <= 0:
            raise ValueError(f"Client update {i} has invalid sample count: {samples}. Must be > 0.")
        total_samples += int(samples)

    if total_samples <= 0:
        raise ValueError("Total sample count across all client updates must be positive.")

    # Use first update or reference state as schema baseline
    base_state = reference_state if reference_state is not None else client_updates[0]["actor_state"]

    # Validate each client update
    for i, update in enumerate(client_updates):
        state = update["actor_state"]
        valid, err = validate_tensor_schema(state, base_state)
        if not valid:
            raise ValueError(f"Client update {i} failed schema validation: {err}")
        if not check_finite_tensors(state):
            raise ValueError(f"Client update {i} contains non-finite values (NaN or Inf).")

    # Accumulate parameters in computation_dtype
    aggregated_state: Dict[str, torch.Tensor] = {}
    for name, ref_tensor in base_state.items():
        if not ref_tensor.is_floating_point():
            # Keep non-floating buffers directly without averaging
            aggregated_state[name] = ref_tensor.clone()
            continue

        accum = torch.zeros(ref_tensor.shape, dtype=computation_dtype)
        for update in client_updates:
            n_k = float(update["valid_samples"])
            weight = n_k / float(total_samples)
            client_param = update["actor_state"][name].to(dtype=computation_dtype)
            accum += weight * client_param

        # Cast back to target float32
        aggregated_state[name] = accum.to(dtype=ref_tensor.dtype)

    return aggregated_state


def save_atomic_checkpoint(
    state: Any,
    save_path: Path,
) -> Path:
    """
    Saves a PyTorch state dictionary or object atomically using a temporary file and rename.
    Guarantees no corrupted or partially written checkpoint files upon interruption.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = save_path.with_suffix(f"{save_path.suffix}.tmp_{os.getpid()}_{np.random.randint(100000, 999999)}")

    try:
        torch.save(state, str(temp_file))
        # Atomic replace on POSIX and Windows (Python 3.3+)
        temp_file.replace(save_path)
    except Exception as e:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise IOError(f"Atomic checkpoint save failed for {save_path}: {e}") from e

    return save_path


def load_safe_checkpoint(
    checkpoint_path: Path,
    expected_schema: Optional[Dict[str, torch.Tensor]] = None,
    map_location: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Loads checkpoint from file with strict integrity checks.
    Rejects malformed or non-finite checkpoints.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Use torch.load with map_location
    state_dict = torch.load(str(checkpoint_path), map_location=map_location)

    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected state dictionary, got {type(state_dict)}")

    if not check_finite_tensors(state_dict):
        raise ValueError(f"Loaded checkpoint from {checkpoint_path} contains NaN or Inf.")

    if expected_schema is not None:
        valid, err = validate_tensor_schema(state_dict, expected_schema)
        if not valid:
            raise ValueError(f"Loaded checkpoint schema mismatch: {err}")

    return state_dict


def compute_jains_fairness(values: List[float]) -> float:
    """
    Computes Jain's Fairness Index for a list of non-negative values:
        J(x_1, ..., x_n) = (sum(x_i))^2 / (n * sum(x_i^2))
    Returns 1.0 for empty or uniform non-zero inputs, 0.0 if all zero.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 1.0
    sum_val = np.sum(arr)
    sum_sq = np.sum(arr**2)
    if sum_sq <= 1e-12:
        return 1.0 if sum_val == 0.0 else 0.0
    return float((sum_val**2) / (len(arr) * sum_sq))
