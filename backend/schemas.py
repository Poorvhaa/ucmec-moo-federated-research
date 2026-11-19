"""
Pydantic Schemas for Multi-Objective Experiment Management API.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
import re


class PresetInfo(BaseModel):
    id: str
    name: str
    description: str
    weights: Dict[str, float]
    mode: str
    use_corrected_cpu_alloc: bool


class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    algorithm_name: str = Field(default="mappo", pattern="^(mappo|rmappo)$")
    preset_name: str = Field(default="balanced")
    w_delay: Optional[float] = Field(default=None, ge=0.0)
    w_energy: Optional[float] = Field(default=None, ge=0.0)
    w_overhead: Optional[float] = Field(default=None, ge=0.0)
    w_reliability: Optional[float] = Field(default=None, ge=0.0)
    num_agents: int = Field(default=10, ge=1, le=60)
    num_env_steps: int = Field(default=400, ge=20, le=100000)
    episode_length: int = Field(default=200, ge=10, le=1000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    use_corrected_cpu_alloc: bool = True
    solver_name: str = Field(default="CLARABEL", pattern="^(CLARABEL|SCS)$")
    channel_mode: str = Field(default="legacy", pattern="^(legacy|sinr_outage)$")
    task_deadline_s: float = Field(default=0.1, gt=0.0, le=10.0)
    # Architecture and policy flags
    share_policy: bool = True
    use_centralized_V: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        # Sanitize name to prevent shell injection or directory traversal
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Experiment name must only contain alphanumeric characters, underscores, and dashes")
        return v


class ExperimentMetricRecord(BaseModel):
    step: int
    episode: int
    timestamp: float
    reward: float
    delay_ms: float
    energy_mj: float
    overhead_kb: float
    task_success_rate: float
    deadline_satisfaction_rate: float
    channel_success_rate: float
    local_users: float
    offloaded_users: float
    solver_mean_ms: float
    solver_max_residual: float


class ExperimentResponse(BaseModel):
    id: str
    name: str
    status: str  # queued, running, completed, failed, cancelled
    algorithm_name: str
    preset_name: str
    weights: Dict[str, float]
    num_agents: int
    num_env_steps: int
    episode_length: int
    seed: int
    current_step: int
    total_steps: int
    progress_percent: float
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    runtime_s: float = 0.0
    git_commit: Optional[str] = None
    git_dirty: bool = False
    solver_name: str
    use_corrected_cpu_alloc: bool
    channel_mode: str
    final_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    log_dir: str


class ExperimentListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ExperimentResponse]


class CapabilitiesResponse(BaseModel):
    available_algorithms: List[str]
    available_solvers: List[str]
    supported_channel_modes: List[str]
    max_agents: int
    max_steps: int
    system_python_version: str
    torch_version: str
    cvxpy_version: str
    clarabel_version: str
