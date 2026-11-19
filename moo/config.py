"""
Multi-Objective Optimization (MOO) Configuration, Presets, and Scales.
"""
from typing import Dict, Any, List
import numpy as np

# Metric reference scales for normalization (fixed, documented, non-dynamic)
# Internal units:
# - Delay: seconds (s)
# - Energy: joules (J)
# - Communication Overhead: bytes (B)
# - Reliability: dimensionless rate [0, 1]
DEFAULT_REFERENCE_SCALES = {
    "delay_ref_s": 0.100,        # 100 ms reference latency
    "energy_ref_j": 0.005,       # 5.0 mJ reference energy
    "overhead_ref_bytes": 1000.0, # 1,000 bytes signaling reference
    "reliability_ref": 1.0       # Normalized [0, 1]
}

# Illustrative default presets (labeled as illustrative defaults, not experimentally optimal)
PRESETS: Dict[str, Dict[str, Any]] = {
    "legacy": {
        "name": "Legacy Reward",
        "description": "Original paper reward: -0.9 * delay + 0.1 * (tau_c - delay)",
        "weights": {"w_delay": 1.0, "w_energy": 0.0, "w_overhead": 0.0, "w_reliability": 0.0},
        "mode": "legacy",
        "use_corrected_cpu_alloc": False,
    },
    "delay_only": {
        "name": "Delay Priority (Normalized)",
        "description": "Multi-objective formulation focused 100% on execution delay minimization",
        "weights": {"w_delay": 1.0, "w_energy": 0.0, "w_overhead": 0.0, "w_reliability": 0.0},
        "mode": "moo",
        "use_corrected_cpu_alloc": True,
    },
    "balanced": {
        "name": "Balanced MOO",
        "description": "Balanced trade-off across delay, energy, overhead, and reliability",
        "weights": {"w_delay": 0.35, "w_energy": 0.25, "w_overhead": 0.15, "w_reliability": 0.25},
        "mode": "moo",
        "use_corrected_cpu_alloc": True,
    },
    "energy_priority": {
        "name": "Energy Conservation Priority",
        "description": "Heavily weights mobile device energy minimization while preserving stability",
        "weights": {"w_delay": 0.15, "w_energy": 0.60, "w_overhead": 0.10, "w_reliability": 0.15},
        "mode": "moo",
        "use_corrected_cpu_alloc": True,
    },
    "reliability_priority": {
        "name": "Reliability / QoS Priority",
        "description": "Heavily penalizes deadline violations and channel outages",
        "weights": {"w_delay": 0.15, "w_energy": 0.15, "w_overhead": 0.10, "w_reliability": 0.60},
        "mode": "moo",
        "use_corrected_cpu_alloc": True,
    },
    "custom": {
        "name": "Custom Weights",
        "description": "User-specified scalarization weights normalized to sum to 1",
        "weights": {"w_delay": 0.25, "w_energy": 0.25, "w_overhead": 0.25, "w_reliability": 0.25},
        "mode": "moo",
        "use_corrected_cpu_alloc": True,
    }
}

# Physical modeling parameters (with explicit units and physical justifications)
DEFAULT_PHYSICAL_PARAMS = {
    # CMOS computation energy coefficient kappa (W / Hz^3 or J * s^2)
    # E_local = kappa * (cycles) * (freq)^2
    # Realistic mobile processor capacitance coefficient: 1e-28 J/(cycle * Hz^2)
    "kappa_user": 1e-28,
    
    # Modeled edge server computation coefficient
    "kappa_edge": 0.5e-28,
    
    # Access channel outage SINR threshold in linear scale (e.g., -3 dB = 0.501)
    "access_sinr_outage_threshold": 0.50,
    
    # Fronthaul channel outage threshold in linear scale
    "front_sinr_outage_threshold": 1.0,
    
    # Task completion deadline in seconds (default = coherence time 0.1s = 100ms)
    "task_deadline_s": 0.100,
    
    # Signaling and protocol control packet sizes (in Bytes)
    "csi_pilot_bytes": 128,          # Channel state information sounding / pilot bytes
    "offload_request_bytes": 64,      # Handshake request (task id, size, deadline)
    "scheduling_grant_bytes": 64,     # AP resource allocation grant
    "fronthaul_signaling_bytes": 128, # AP-to-CPU routing & allocation message
    "ack_bytes": 32,                 # Result acknowledgement / completion signaling
    
    # Channel failure mode: 'legacy' (always perfect channel), 'sinr_outage' (SINR threshold)
    "channel_mode": "legacy",
}
