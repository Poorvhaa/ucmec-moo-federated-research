"""
Multi-Objective Optimization Metric Models and Metric Contract.

Defines the mathematical models, units, equations, assumptions, and contracts
for Delay, Energy, Communication Overhead, and Reliability.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
from moo.config import DEFAULT_PHYSICAL_PARAMS, DEFAULT_REFERENCE_SCALES


@dataclass
class StepMetricSnapshot:
    """Per-step simulator metric snapshot."""
    # Aggregated averages across users (internal units)
    mean_delay_s: float
    mean_delay_ms: float
    mean_energy_j: float
    mean_energy_mj: float
    mean_overhead_bytes: float
    mean_overhead_kb: float
    task_success_rate: float
    deadline_satisfaction_rate: float
    channel_success_rate: float
    
    # Detailed components
    mean_local_delay_s: float
    mean_uplink_delay_s: float
    mean_front_delay_s: float
    mean_process_delay_s: float
    mean_tx_energy_j: float
    mean_local_comp_energy_j: float
    mean_edge_modeled_energy_j: float
    mean_payload_bytes: float
    mean_signaling_bytes: float
    
    # Offloading distribution
    num_local_users: int
    num_offloaded_users: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricContract:
    """
    Contract and calculator for the four research objectives:
    1. Execution Delay (s / ms)
    2. Energy Consumption (J / mJ)
    3. Communication Overhead (Bytes / kB)
    4. Reliability / Simulated Task Success Rate ([0, 1])
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**DEFAULT_PHYSICAL_PARAMS, **(params or {})}

    def compute_step_metrics(
        self,
        omega: np.ndarray,              # Chosen offloading target [M] (0: local, 1..K: CPU)
        p: np.ndarray,                  # Transmit power in Watts [M]
        local_delay: np.ndarray,        # Local delay in seconds [M, 1]
        uplink_delay: np.ndarray,       # Uplink transmission delay in seconds [M, 1]
        front_delay: np.ndarray,        # Fronthaul transmission delay in seconds [M, 1]
        process_delay: np.ndarray,      # Edge computing delay in seconds [M, 1]
        task_size_bits: np.ndarray,     # Task payload size in bits [1, M] or [M]
        task_density: np.ndarray,       # Task density cycles/bit [1, M] or [M]
        c_user: np.ndarray,             # User CPU frequency in Hz [1, M] or [M]
        c_actual: np.ndarray,           # Allocated edge CPU in Hz [M, K]
        sinr_access: Optional[np.ndarray] = None, # Access link SINR linear scale [M, 1]
        sinr_front: Optional[np.ndarray] = None,  # Fronthaul link SINR linear scale [M, 1]
    ) -> StepMetricSnapshot:
        """
        Compute step metrics adhering to explicit physical models and no double-counting.
        """
        M = len(omega)
        task_size_bits = np.asarray(task_size_bits).reshape(-1)[:M]
        task_density = np.asarray(task_density).reshape(-1)[:M]
        c_user = np.asarray(c_user).reshape(-1)[:M]
        
        # ---------------------------------------------------------
        # A. Delay Model
        # Local and offloaded execution are mutually exclusive alternatives:
        # If omega[i] == 0: local delay
        # If omega[i] > 0:  uplink_delay + front_delay + process_delay
        # ---------------------------------------------------------
        per_user_delay_s = np.zeros(M, dtype=float)
        for i in range(M):
            if omega[i] == 0:
                per_user_delay_s[i] = local_delay[i, 0]
            else:
                per_user_delay_s[i] = (
                    uplink_delay[i, 0] + front_delay[i, 0] + process_delay[i, 0]
                )

        # ---------------------------------------------------------
        # B. Energy Model
        # Physical model for local computation:
        # E_local = kappa_user * cycles * (f_user)^2
        # where cycles = task_density * task_size_bits
        #
        # Physical model for transmission energy:
        # E_tx = p_tx * T_uplink (only when offloading, p_tx > 0)
        # ---------------------------------------------------------
        per_user_energy_j = np.zeros(M, dtype=float)
        local_comp_energy_j = np.zeros(M, dtype=float)
        tx_energy_j = np.zeros(M, dtype=float)
        edge_comp_energy_j = np.zeros(M, dtype=float)

        kappa_u = float(self.params["kappa_user"])
        kappa_e = float(self.params["kappa_edge"])

        for i in range(M):
            cycles = task_density[i] * task_size_bits[i]
            if omega[i] == 0:
                # Local execution on device
                e_loc = kappa_u * cycles * (c_user[i] ** 2)
                local_comp_energy_j[i] = e_loc
                per_user_energy_j[i] = e_loc
            else:
                # Offloaded: device transmission energy
                e_tx = p[i] * uplink_delay[i, 0]
                tx_energy_j[i] = e_tx
                per_user_energy_j[i] = e_tx
                
                # Modeled edge server energy (recorded for network accounting)
                cpu_allocated = np.sum(c_actual[i, :])
                if cpu_allocated > 0:
                    edge_comp_energy_j[i] = kappa_e * cycles * (cpu_allocated ** 2)

        # ---------------------------------------------------------
        # C. Communication Overhead Model
        # When offloading:
        # Payload bytes = task_size_bits / 8
        # Signaling bytes = CSI + offload request + scheduling grant + fronthaul sig + ACK
        # When local:
        # 0 bytes transmitted over the air.
        # ---------------------------------------------------------
        payload_bytes = np.zeros(M, dtype=float)
        signaling_bytes = np.zeros(M, dtype=float)
        
        fixed_sig = (
            float(self.params["csi_pilot_bytes"])
            + float(self.params["offload_request_bytes"])
            + float(self.params["scheduling_grant_bytes"])
            + float(self.params["fronthaul_signaling_bytes"])
            + float(self.params["ack_bytes"])
        )

        for i in range(M):
            if omega[i] != 0:
                payload_bytes[i] = task_size_bits[i] / 8.0
                signaling_bytes[i] = fixed_sig
            else:
                payload_bytes[i] = 0.0
                signaling_bytes[i] = 0.0

        per_user_overhead_bytes = signaling_bytes.copy()

        # ---------------------------------------------------------
        # D. Reliability / Task Success Rate Model
        # Success = (Delay <= Deadline) and (Channel Success) and (Valid CPU Allocation)
        # ---------------------------------------------------------
        deadline_s = float(self.params["task_deadline_s"])
        deadline_met = per_user_delay_s <= deadline_s
        channel_success = np.ones(M, dtype=bool)

        if self.params["channel_mode"] == "sinr_outage":
            access_th = float(self.params["access_sinr_outage_threshold"])
            front_th = float(self.params["front_sinr_outage_threshold"])
            for i in range(M):
                if omega[i] != 0:
                    sinr_a = sinr_access[i, 0] if sinr_access is not None else 100.0
                    sinr_f = sinr_front[i, 0] if sinr_front is not None else 100.0
                    if sinr_a < access_th or sinr_f < front_th:
                        channel_success[i] = False
        
        # Valid processing: CPU allocation non-zero when offloaded
        processing_valid = np.ones(M, dtype=bool)
        for i in range(M):
            if omega[i] != 0:
                if np.sum(c_actual[i, :]) <= 0:
                    processing_valid[i] = False

        task_success = deadline_met & channel_success & processing_valid

        num_local = int(np.sum(omega == 0))
        num_offload = M - num_local

        return StepMetricSnapshot(
            mean_delay_s=float(np.mean(per_user_delay_s)),
            mean_delay_ms=float(np.mean(per_user_delay_s) * 1000.0),
            mean_energy_j=float(np.mean(per_user_energy_j)),
            mean_energy_mj=float(np.mean(per_user_energy_j) * 1000.0),
            mean_overhead_bytes=float(np.mean(per_user_overhead_bytes)),
            mean_overhead_kb=float(np.mean(per_user_overhead_bytes) / 1024.0),
            task_success_rate=float(np.mean(task_success)),
            deadline_satisfaction_rate=float(np.mean(deadline_met)),
            channel_success_rate=float(np.mean(channel_success)),
            mean_local_delay_s=float(np.mean(local_delay)),
            mean_uplink_delay_s=float(np.mean(uplink_delay)),
            mean_front_delay_s=float(np.mean(front_delay)),
            mean_process_delay_s=float(np.mean(process_delay)),
            mean_tx_energy_j=float(np.mean(tx_energy_j)),
            mean_local_comp_energy_j=float(np.mean(local_comp_energy_j)),
            mean_edge_modeled_energy_j=float(np.mean(edge_comp_energy_j)),
            mean_payload_bytes=float(np.mean(payload_bytes)),
            mean_signaling_bytes=float(np.mean(signaling_bytes)),
            num_local_users=num_local,
            num_offloaded_users=num_offload,
        )
