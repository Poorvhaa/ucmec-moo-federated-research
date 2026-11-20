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

export interface FederatedCapabilities {
  supported_scenarios: string[];
  supported_aggregation: string;
  privacy_disclaimer: string;
  max_clients: number;
  max_rounds: number;
  default_smoke_config: Record<string, any>;
}

export interface FederatedClientInfo {
  client_id: string;
  site_name: string;
  seed: number;
  local_rollout_steps: number;
  local_epochs: number;
  task_size_range?: [number, number];
  c_edge_range?: [number, number];
  epsilon?: number;
}

export interface FederatedRoundRecord {
  round: number;
  runtime_s: number;
  total_valid_samples: number;
  mean_delay_ms: number;
  mean_energy_mj: number;
  mean_overhead_kb: number;
  mean_task_success_rate: number;
  worst_task_success_rate?: number;
  worst_delay_ms?: number;
  jains_fairness_index?: number;
  eval_delay_ms?: number;
  eval_energy_mj?: number;
  eval_overhead_kb?: number;
  eval_task_success_rate?: number;
  eval_reward?: number;
  download_bytes?: number;
  upload_bytes?: number;
  failed_bytes?: number;
  accepted_clients?: string[];
  dropped_clients?: string[];
}

export interface CommunicationSummary {
  total_transmissions: number;
  cumulative_upload_bytes: number;
  cumulative_download_bytes: number;
  cumulative_total_bytes: number;
  cumulative_total_mb: number;
  failed_transmissions: number;
  failed_bytes: number;
}

export interface FederatedExperiment {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  preset_name: string;
  scenario_type: string;
  num_clients: number;
  num_rounds: number;
  current_round: number;
  min_clients: number;
  participation_rate: number;
  dropout_prob: number;
  seed: number;
  progress_percent: number;
  created_at: number;
  started_at?: number;
  finished_at?: number;
  runtime_s: number;
  solver_name: string;
  channel_mode: string;
  final_metrics?: any;
  communication_summary?: CommunicationSummary;
  error_message?: string;
  log_dir: string;
}

export interface FederatedBenchmarkStudy {
  study_id: string;
  preset: string;
  num_rounds: number;
  num_clients: number;
  tradeoff_table: Array<{
    method: string;
    mean_delay_ms: number;
    mean_energy_mj: number;
    mean_overhead_kb: number;
    mean_task_success_rate: number;
    jains_fairness_index: number;
    total_communication_mb: number;
    privacy_preserving: boolean;
  }>;
}
