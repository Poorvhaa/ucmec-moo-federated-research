"""
Federated Multi-Agent Learning: Configurable Client Heterogeneity Scenarios.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class ClientScenarioConfig:
    client_id: str
    site_name: str
    seed: int
    task_size_range: Tuple[float, float] = (50000.0, 100000.0)  # bits
    c_edge_range: Tuple[float, float] = (10e9, 20e9)            # CPU Hz
    epsilon: float = 6e-4                                       # Blockage density
    local_rollout_steps: int = 200
    local_epochs: int = 5
    batch_size: int = 32
    extra_env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "site_name": self.site_name,
            "seed": self.seed,
            "task_size_range": list(self.task_size_range),
            "c_edge_range": list(self.c_edge_range),
            "epsilon": self.epsilon,
            "local_rollout_steps": self.local_rollout_steps,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "extra_env_kwargs": self.extra_env_kwargs,
        }


def generate_federation_scenarios(
    num_clients: int = 3,
    scenario_type: str = "iid",  # 'iid', 'task_heterogeneity', 'channel_heterogeneity', 'compute_heterogeneity', 'full_heterogeneity'
    base_seed: int = 100,
    rollout_steps: int = 200,
    local_epochs: int = 5,
) -> List[ClientScenarioConfig]:
    """
    Generates deterministic, independently-seeded client site scenario configurations.
    Ensures per-client environment RNG independence.
    """
    clients: List[ClientScenarioConfig] = []
    rng = np.random.RandomState(base_seed)

    for i in range(num_clients):
        client_seed = int(base_seed + 1000 * (i + 1) + rng.randint(1, 999))
        site_name = f"Site-{chr(65 + (i % 26))}"

        task_range = (50000.0, 100000.0)
        c_edge_range = (10e9, 20e9)
        epsilon = 6e-4
        steps = rollout_steps

        if scenario_type == "iid":
            pass
        elif scenario_type == "task_heterogeneity":
            # Site 0: lightweight tasks; Site 1: medium; Site 2+: compute-heavy tasks
            scales = [
                (20000.0, 50000.0),
                (50000.0, 100000.0),
                (100000.0, 200000.0),
            ]
            task_range = scales[i % len(scales)]
        elif scenario_type == "channel_heterogeneity":
            # Varying mmWave blockage density
            epsilons = [1e-4, 6e-4, 15e-4]
            epsilon = epsilons[i % len(epsilons)]
        elif scenario_type == "compute_heterogeneity":
            # Heterogeneous MEC server compute resources
            capacities = [
                (5e9, 10e9),    # Constrained edge server
                (10e9, 20e9),  # Standard edge server
                (20e9, 40e9),  # High-capacity edge server
            ]
            c_edge_range = capacities[i % len(capacities)]
        elif scenario_type == "sample_budget_heterogeneity":
            # Heterogeneous local rollout budgets
            budgets = [100, 200, 300]
            steps = budgets[i % len(budgets)]
        elif scenario_type == "full_heterogeneity":
            task_scales = [(25000.0, 60000.0), (50000.0, 100000.0), (90000.0, 180000.0)]
            task_range = task_scales[i % len(task_scales)]
            epsilons = [2e-4, 6e-4, 12e-4]
            epsilon = epsilons[i % len(epsilons)]
            capacities = [(6e9, 12e9), (12e9, 24e9), (20e9, 35e9)]
            c_edge_range = capacities[i % len(capacities)]
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        clients.append(
            ClientScenarioConfig(
                client_id=f"client_{i}",
                site_name=site_name,
                seed=client_seed,
                task_size_range=task_range,
                c_edge_range=c_edge_range,
                epsilon=epsilon,
                local_rollout_steps=steps,
                local_epochs=local_epochs,
            )
        )

    return clients
