import type { PresetInfo, Experiment, Capabilities, ComparisonItem, ArtifactFile, MetricRecord } from './types';

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
