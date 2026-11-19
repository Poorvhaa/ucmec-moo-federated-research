export interface PresetInfo {
  id: string;
  name: string;
  description: string;
  weights: {
    w_delay: number;
    w_energy: number;
    w_overhead: number;
    w_reliability: number;
  };
  mode: string;
  use_corrected_cpu_alloc: boolean;
}

export interface MetricRecord {
  step: number;
  episode: number;
  timestamp: number;
  reward: number;
  delay_ms: number;
  energy_mj: number;
  overhead_kb: number;
  task_success_rate: number;
  deadline_satisfaction_rate: number;
  channel_success_rate: number;
  local_users: number;
  offloaded_users: number;
  solver_mean_ms: number;
  solver_max_residual: number;
}

export interface Experiment {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  algorithm_name: string;
  preset_name: string;
  weights: {
    w_delay: number;
    w_energy: number;
    w_overhead: number;
    w_reliability: number;
  };
  num_agents: number;
  num_env_steps: number;
  episode_length: number;
  seed: number;
  current_step: number;
  total_steps: number;
  progress_percent: number;
  created_at: number;
  started_at?: number;
  finished_at?: number;
  runtime_s: number;
  git_commit?: string;
  git_dirty: boolean;
  solver_name: string;
  use_corrected_cpu_alloc: boolean;
  channel_mode: string;
  final_metrics?: MetricRecord;
  error_message?: string;
  log_dir: string;
}

export interface Capabilities {
  available_algorithms: string[];
  available_solvers: string[];
  supported_channel_modes: string[];
  max_agents: number;
  max_steps: number;
  system_python_version: string;
  torch_version: string;
  cvxpy_version: string;
  clarabel_version: string;
}

export interface ComparisonItem {
  id: string;
  name: string;
  preset: string;
  algorithm: string;
  seed: number;
  num_agents: number;
  episode_length: number;
  delay_ms: number;
  energy_mj: number;
  overhead_kb: number;
  task_success_rate: number;
  reward: number;
  is_nondominated?: boolean;
}

export interface ArtifactFile {
  name: string;
  size_bytes: number;
  type: string;
}
