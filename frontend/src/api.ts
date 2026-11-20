import type {
  PresetInfo,
  Experiment,
  Capabilities,
  ComparisonItem,
  ArtifactFile,
  MetricRecord,
  FederatedCapabilities,
  FederatedExperiment,
  FederatedRoundRecord,
  FederatedClientInfo,
} from './types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<any> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to fetch system health');
  return res.json();
}

export async function fetchCapabilities(): Promise<Capabilities> {
  const res = await fetch(`${API_BASE}/capabilities`);
  if (!res.ok) throw new Error('Failed to fetch system capabilities');
  return res.json();
}

export async function fetchPresets(): Promise<PresetInfo[]> {
  const res = await fetch(`${API_BASE}/presets`);
  if (!res.ok) throw new Error('Failed to fetch presets');
  return res.json();
}

export async function fetchExperiments(page = 1, pageSize = 20): Promise<{ total: number; items: Experiment[] }> {
  const res = await fetch(`${API_BASE}/experiments?page=${page}&page_size=${pageSize}`);
  if (!res.ok) throw new Error('Failed to list experiments');
  return res.json();
}

export async function fetchExperiment(id: string): Promise<Experiment> {
  const res = await fetch(`${API_BASE}/experiments/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch experiment ${id}`);
  return res.json();
}

export async function createExperiment(payload: any): Promise<Experiment> {
  const res = await fetch(`${API_BASE}/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to create experiment');
  }
  return res.json();
}

export async function cancelExperiment(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/experiments/${id}/cancel`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to cancel experiment ${id}`);
  return res.json();
}

export async function fetchExperimentMetrics(id: string): Promise<{ total_records: number; metrics: MetricRecord[] }> {
  const res = await fetch(`${API_BASE}/experiments/${id}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics for ${id}`);
  return res.json();
}

export async function fetchExperimentLogs(id: string, tail = 200): Promise<{ stdout: string; stderr: string }> {
  const res = await fetch(`${API_BASE}/experiments/${id}/logs?tail=${tail}`);
  if (!res.ok) throw new Error(`Failed to fetch logs for ${id}`);
  return res.json();
}

export async function fetchExperimentArtifacts(id: string): Promise<{ artifacts: ArtifactFile[] }> {
  const res = await fetch(`${API_BASE}/experiments/${id}/artifacts`);
  if (!res.ok) throw new Error(`Failed to fetch artifacts for ${id}`);
  return res.json();
}

export async function fetchComparisons(): Promise<{ total_compared: number; experiments: ComparisonItem[]; nondominated_subset: ComparisonItem[] }> {
  const res = await fetch(`${API_BASE}/comparisons`);
  if (!res.ok) throw new Error('Failed to fetch comparisons');
  return res.json();
}

export function getArtifactDownloadUrl(expId: string, artifactPath: string): string {
  return `${API_BASE}/experiments/${expId}/artifacts/${encodeURIComponent(artifactPath)}`;
}

// -------------------------------------------------------------
// Federated Learning Endpoints
// -------------------------------------------------------------

export async function fetchFederatedCapabilities(): Promise<FederatedCapabilities> {
  const res = await fetch(`${API_BASE}/federation/capabilities`);
  if (!res.ok) throw new Error('Failed to fetch federation capabilities');
  return res.json();
}

export async function fetchFederatedExperiments(
  page = 1,
  pageSize = 20
): Promise<{ total: number; items: FederatedExperiment[] }> {
  const res = await fetch(`${API_BASE}/federated-experiments?page=${page}&page_size=${pageSize}`);
  if (!res.ok) throw new Error('Failed to list federated experiments');
  return res.json();
}

export async function fetchFederatedExperiment(id: string): Promise<FederatedExperiment> {
  const res = await fetch(`${API_BASE}/federated-experiments/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch federated experiment ${id}`);
  return res.json();
}

export async function createFederatedExperiment(payload: any): Promise<FederatedExperiment> {
  const res = await fetch(`${API_BASE}/federated-experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to create federated experiment');
  }
  return res.json();
}

export async function cancelFederatedExperiment(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/federated-experiments/${id}/cancel`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to cancel federated experiment ${id}`);
  return res.json();
}

export async function resumeFederatedExperiment(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/federated-experiments/${id}/resume`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to resume federated experiment ${id}`);
  return res.json();
}

export async function fetchFederatedRounds(
  id: string
): Promise<{ total_rounds: number; rounds: FederatedRoundRecord[] }> {
  const res = await fetch(`${API_BASE}/federated-experiments/${id}/rounds`);
  if (!res.ok) throw new Error(`Failed to fetch rounds for ${id}`);
  return res.json();
}

export async function fetchFederatedClients(
  id: string
): Promise<{ total_clients: number; clients: FederatedClientInfo[] }> {
  const res = await fetch(`${API_BASE}/federated-experiments/${id}/clients`);
  if (!res.ok) throw new Error(`Failed to fetch clients for ${id}`);
  return res.json();
}

export async function fetchFederatedMetrics(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/federated-experiments/${id}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics for ${id}`);
  return res.json();
}

export async function fetchFederatedArtifacts(id: string): Promise<{ artifacts: ArtifactFile[] }> {
  const res = await fetch(`${API_BASE}/federated-experiments/${id}/artifacts`);
  if (!res.ok) throw new Error(`Failed to fetch artifacts for ${id}`);
  return res.json();
}

export function getFederatedArtifactDownloadUrl(expId: string, subpath: string): string {
  return `${API_BASE}/federated-experiments/${expId}/artifacts/${encodeURIComponent(subpath)}`;
}

export async function fetchFederatedComparisons(): Promise<any> {
  const res = await fetch(`${API_BASE}/federated-comparisons`);
  if (!res.ok) throw new Error('Failed to fetch federated comparisons');
  return res.json();
}
