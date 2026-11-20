import React, { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  PlusCircle,
  Activity,
  GitCompare,
  TrendingUp,
  FolderArchive,
  RefreshCw,
  AlertCircle,
  Download,
  Terminal,
  Cpu,
  Radio,
  StopCircle,
  Network
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';

import {
  fetchCapabilities,
  fetchPresets,
  fetchExperiments,
  fetchExperiment,
  createExperiment,
  cancelExperiment,
  fetchExperimentMetrics,
  fetchExperimentLogs,
  fetchExperimentArtifacts,
  fetchComparisons,
  getArtifactDownloadUrl
} from './api';
import type { PresetInfo, Experiment, Capabilities, ComparisonItem, ArtifactFile, MetricRecord } from './types';
import { FederatedView } from './FederatedView';

export function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'create' | 'live' | 'compare' | 'tradeoff' | 'artifacts' | 'federation'>('overview');
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [selectedExp, setSelectedExp] = useState<Experiment | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<MetricRecord[]>([]);
  const [liveLogs, setLiveLogs] = useState<{ stdout: string; stderr: string }>({ stdout: '', stderr: '' });
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [comparisons, setComparisons] = useState<ComparisonItem[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactFile[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [formData, setFormData] = useState({
    name: 'moo_run',
    algorithm_name: 'mappo',
    preset_name: 'balanced',
    w_delay: 0.35,
    w_energy: 0.25,
    w_overhead: 0.15,
    w_reliability: 0.25,
    num_agents: 10,
    num_env_steps: 400,
    episode_length: 200,
    seed: 42,
    use_corrected_cpu_alloc: true,
    solver_name: 'CLARABEL',
    channel_mode: 'legacy',
    task_deadline_s: 0.1,
  });

  // Tradeoff scatter plot axes
  const [scatterX, setScatterX] = useState<'delay_ms' | 'energy_mj' | 'overhead_kb' | 'task_success_rate'>('delay_ms');
  const [scatterY, setScatterY] = useState<'delay_ms' | 'energy_mj' | 'overhead_kb' | 'task_success_rate'>('energy_mj');

  // Load initial metadata
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setError(null);
      const [presetsData, capsData, expsData, compsData] = await Promise.all([
        fetchPresets(),
        fetchCapabilities(),
        fetchExperiments(1, 50),
        fetchComparisons(),
      ]);
      setPresets(presetsData);
      setCapabilities(capsData);
      setExperiments(expsData.items);
      setComparisons(compsData.experiments);
      if (expsData.items.length > 0 && !selectedExpId) {
        setSelectedExpId(expsData.items[0].id);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to connect to backend service');
    }
  };

  // Poll live experiment state
  useEffect(() => {
    if (!selectedExpId) return;
    let isSubscribed = true;

    const poll = async () => {
      try {
        const [exp, metricsData, logsData, artsData] = await Promise.all([
          fetchExperiment(selectedExpId),
          fetchExperimentMetrics(selectedExpId),
          fetchExperimentLogs(selectedExpId, 200),
          fetchExperimentArtifacts(selectedExpId),
        ]);
        if (isSubscribed) {
          setSelectedExp(exp);
          setLiveMetrics(metricsData.metrics);
          setLiveLogs(logsData);
          setArtifacts(artsData.artifacts);
        }
      } catch (err) {
        // Silently catch transient polling errors
      }
    };

    poll();
    const interval = setInterval(poll, 2500);
    return () => {
      isSubscribed = false;
      clearInterval(interval);
    };
  }, [selectedExpId]);

  // Handle Preset change in form
  const handlePresetSelect = (presetId: string) => {
    const p = presets.find((item) => item.id === presetId);
    if (p) {
      setFormData((prev) => ({
        ...prev,
        preset_name: p.id,
        w_delay: p.weights.w_delay,
        w_energy: p.weights.w_energy,
        w_overhead: p.weights.w_overhead,
        w_reliability: p.weights.w_reliability,
        use_corrected_cpu_alloc: p.use_corrected_cpu_alloc,
      }));
    }
  };

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const created = await createExperiment(formData);
      setExperiments((prev) => [created, ...prev]);
      setSelectedExpId(created.id);
      setActiveTab('live');
    } catch (err: any) {
      setError(err.message || 'Error creating experiment');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await cancelExperiment(id);
      const updated = await fetchExperiment(id);
      setSelectedExp(updated);
    } catch (err: any) {
      setError(err.message || 'Error cancelling experiment');
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand-header">
          <div style={{ background: 'var(--teal-600)', borderRadius: '8px', padding: '6px' }}>
            <Radio size={22} color="#ffffff" />
          </div>
          <div>
            <div className="brand-title">UCMEC MOO</div>
            <div className="brand-subtitle">mmWave Edge Platform</div>
          </div>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-btn ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <LayoutDashboard size={18} /> Overview
          </button>
          <button
            className={`nav-btn ${activeTab === 'create' ? 'active' : ''}`}
            onClick={() => setActiveTab('create')}
          >
            <PlusCircle size={18} /> New Experiment
          </button>
          <button
            className={`nav-btn ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            <Activity size={18} /> Live Monitor
          </button>
          <button
            className={`nav-btn ${activeTab === 'compare' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('compare');
              fetchComparisons().then((c) => setComparisons(c.experiments));
            }}
          >
            <GitCompare size={18} /> Comparison
          </button>
          <button
            className={`nav-btn ${activeTab === 'tradeoff' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('tradeoff');
              fetchComparisons().then((c) => setComparisons(c.experiments));
            }}
          >
            <TrendingUp size={18} /> Tradeoff Analysis
          </button>
          <button
            className={`nav-btn ${activeTab === 'artifacts' ? 'active' : ''}`}
            onClick={() => setActiveTab('artifacts')}
          >
            <FolderArchive size={18} /> Artifacts
          </button>
          <button
            className={`nav-btn ${activeTab === 'federation' ? 'active' : ''}`}
            onClick={() => setActiveTab('federation')}
          >
            <Network size={18} /> Federated Learning
          </button>
        </nav>

        <div className="sidebar-footer">
          <div>Engine: MAPPO / CLARABEL</div>
          <div>Python: {capabilities?.system_python_version || '3.10'}</div>
          <div>PyTorch: CPU Mode</div>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="main-content">
        {error && (
          <div className="error-banner">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
            <button className="btn btn-secondary btn-sm" onClick={loadInitialData}>
              <RefreshCw size={14} /> Retry
            </button>
          </div>
        )}

        {/* 1. OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <div>
            <div className="page-header">
              <h1 className="page-title">Research Dashboard Overview</h1>
              <p className="page-desc">
                Multi-objective optimization simulator for user-centric mmWave mobile edge computing.
              </p>
            </div>

            <div className="grid-cards">
              <div className="stat-card">
                <div className="stat-label">Total Experiments</div>
                <div className="stat-value">{experiments.length}</div>
                <div className="stat-subtext">Persisted in SQLite</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Active Worker State</div>
                <div className="stat-value" style={{ color: 'var(--teal-600)' }}>
                  {experiments.some((e) => e.status === 'running') ? 'RUNNING' : 'IDLE'}
                </div>
                <div className="stat-subtext">Concurrency Quota: 1</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Objectives Modeled</div>
                <div className="stat-value">4</div>
                <div className="stat-subtext">Delay, Energy, Overhead, Reliability</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Default Inner Solver</div>
                <div className="stat-value" style={{ fontSize: '1.4rem' }}>
                  {capabilities?.available_solvers[0] || 'CLARABEL'}
                </div>
                <div className="stat-subtext">Conic interior point</div>
              </div>
            </div>

            <div className="card">
              <div className="card-title">
                <span>Recent Experiments</span>
                <button className="btn btn-secondary btn-sm" onClick={loadInitialData}>
                  <RefreshCw size={14} /> Refresh
                </button>
              </div>

              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Experiment</th>
                      <th>Status</th>
                      <th>Preset</th>
                      <th>Steps</th>
                      <th>Delay (ms)</th>
                      <th>Energy (mJ)</th>
                      <th>Overhead (kB)</th>
                      <th>Reliability</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {experiments.length === 0 ? (
                      <tr>
                        <td colSpan={9} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                          No experiments recorded yet. Click "New Experiment" to launch one.
                        </td>
                      </tr>
                    ) : (
                      experiments.map((exp) => (
                        <tr key={exp.id}>
                          <td>
                            <strong>{exp.name}</strong>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {exp.id} | Seed: {exp.seed}</div>
                          </td>
                          <td>
                            <span className={`badge badge-${exp.status}`}>{exp.status.toUpperCase()}</span>
                          </td>
                          <td>{exp.preset_name}</td>
                          <td>{exp.current_step} / {exp.total_steps}</td>
                          <td>{exp.final_metrics ? exp.final_metrics.delay_ms.toFixed(1) : '—'}</td>
                          <td>{exp.final_metrics ? exp.final_metrics.energy_mj.toFixed(2) : '—'}</td>
                          <td>{exp.final_metrics ? exp.final_metrics.overhead_kb.toFixed(3) : '—'}</td>
                          <td>{exp.final_metrics ? (exp.final_metrics.task_success_rate * 100).toFixed(1) + '%' : '—'}</td>
                          <td>
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => {
                                setSelectedExpId(exp.id);
                                setActiveTab('live');
                              }}
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 2. EXPERIMENT CREATION TAB */}
        {activeTab === 'create' && (
          <div>
            <div className="page-header">
              <h1 className="page-title">Launch Multi-Objective Experiment</h1>
              <p className="page-desc">
                Configure scalarization weights, reference normalizers, and edge computing physical parameters.
              </p>
            </div>

            <form onSubmit={handleCreateSubmit} style={{ maxWidth: '800px' }}>
              <div className="card">
                <div className="card-title">1. Scenario & Identity</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div className="form-group">
                    <label className="form-label">Experiment Name</label>
                    <input
                      type="text"
                      className="form-input"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      required
                    />
                    <div className="form-helper">Alphanumeric, dashes, underscores only</div>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Random Seed</label>
                    <input
                      type="number"
                      className="form-input"
                      value={formData.seed}
                      onChange={(e) => setFormData({ ...formData, seed: parseInt(e.target.value) || 0 })}
                      required
                    />
                    <div className="form-helper">Guarantees exact trajectory reproducibility</div>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
                  <div className="form-group">
                    <label className="form-label">Simulation Users (M)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={formData.num_agents}
                      min={1}
                      max={60}
                      onChange={(e) => setFormData({ ...formData, num_agents: parseInt(e.target.value) || 10 })}
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Total Steps</label>
                    <input
                      type="number"
                      className="form-input"
                      value={formData.num_env_steps}
                      min={20}
                      max={100000}
                      step={200}
                      onChange={(e) => setFormData({ ...formData, num_env_steps: parseInt(e.target.value) || 200 })}
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Episode Horizon</label>
                    <input
                      type="number"
                      className="form-input"
                      value={formData.episode_length}
                      readOnly
                      disabled
                    />
                    <div className="form-helper">Standard 200-step horizon</div>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-title">2. Multi-Objective Scalarization Weights</div>
                <div className="form-group">
                  <label className="form-label">Preset Configuration</label>
                  <select
                    className="form-select"
                    value={formData.preset_name}
                    onChange={(e) => handlePresetSelect(e.target.value)}
                  >
                    {presets.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({p.description})
                      </option>
                    ))}
                  </select>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginTop: '16px' }}>
                  <div>
                    <label className="form-label">w_delay: {formData.w_delay.toFixed(2)}</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={formData.w_delay}
                      onChange={(e) => setFormData({ ...formData, w_delay: parseFloat(e.target.value), preset_name: 'custom' })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <label className="form-label">w_energy: {formData.w_energy.toFixed(2)}</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={formData.w_energy}
                      onChange={(e) => setFormData({ ...formData, w_energy: parseFloat(e.target.value), preset_name: 'custom' })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <label className="form-label">w_overhead: {formData.w_overhead.toFixed(2)}</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={formData.w_overhead}
                      onChange={(e) => setFormData({ ...formData, w_overhead: parseFloat(e.target.value), preset_name: 'custom' })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <label className="form-label">w_reliability: {formData.w_reliability.toFixed(2)}</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={formData.w_reliability}
                      onChange={(e) => setFormData({ ...formData, w_reliability: parseFloat(e.target.value), preset_name: 'custom' })}
                      style={{ width: '100%' }}
                    />
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-title">3. Optimization & Network Settings</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div className="form-group">
                    <label className="form-label">Conic Solver</label>
                    <select
                      className="form-select"
                      value={formData.solver_name}
                      onChange={(e) => setFormData({ ...formData, solver_name: e.target.value })}
                    >
                      <option value="CLARABEL">CLARABEL (Recommended interior point)</option>
                      <option value="SCS">SCS (Legacy splitting cone solver)</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Fronthaul Delay in Inner CPU Optimization</label>
                    <select
                      className="form-select"
                      value={formData.use_corrected_cpu_alloc ? 'true' : 'false'}
                      onChange={(e) => setFormData({ ...formData, use_corrected_cpu_alloc: e.target.value === 'true' })}
                    >
                      <option value="true">Corrected (Include Fronthaul Delay)</option>
                      <option value="false">Legacy Mode (Omit Fronthaul in Objective)</option>
                    </select>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div className="form-group">
                    <label className="form-label">Reliability Channel Mode</label>
                    <select
                      className="form-select"
                      value={formData.channel_mode}
                      onChange={(e) => setFormData({ ...formData, channel_mode: e.target.value })}
                    >
                      <option value="legacy">Legacy (Channel Outage Disabled)</option>
                      <option value="sinr_outage">SINR Threshold Outage Model</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Task Latency Deadline (s)</label>
                    <input
                      type="number"
                      step="0.01"
                      className="form-input"
                      value={formData.task_deadline_s}
                      onChange={(e) => setFormData({ ...formData, task_deadline_s: parseFloat(e.target.value) || 0.1 })}
                    />
                    <div className="form-helper">Default: 0.1s (coherence window tau_c = 100ms)</div>
                  </div>
                </div>
              </div>

              <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: '14px' }} disabled={loading}>
                {loading ? 'Submitting Experiment...' : 'Launch Experiment Execution'}
              </button>
            </form>
          </div>
        )}

        {/* 3. LIVE MONITOR TAB */}
        {activeTab === 'live' && (
          <div>
            <div className="page-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <h1 className="page-title">Live Experiment Execution</h1>
                <p className="page-desc">
                  Real-time telemetry, objective curves, and inner solver diagnostic stream.
                </p>
              </div>

              {selectedExp && (
                <div style={{ display: 'flex', gap: '12px' }}>
                  {selectedExp.status === 'running' && (
                    <button className="btn btn-danger btn-sm" onClick={() => handleCancel(selectedExp.id)}>
                      <StopCircle size={16} /> Cancel Job
                    </button>
                  )}
                  <select
                    className="form-select"
                    style={{ width: '220px' }}
                    value={selectedExpId || ''}
                    onChange={(e) => setSelectedExpId(e.target.value)}
                  >
                    {experiments.map((e) => (
                      <option key={e.id} value={e.id}>
                        {e.name} ({e.status})
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {selectedExp ? (
              <div>
                {/* Progress bar */}
                <div className="card" style={{ padding: '16px 20px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span className={`badge badge-${selectedExp.status}`} style={{ marginRight: '10px' }}>
                        {selectedExp.status.toUpperCase()}
                      </span>
                      <strong>{selectedExp.name}</strong>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginLeft: '10px' }}>
                        Preset: {selectedExp.preset_name} | Seed: {selectedExp.seed}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                      {selectedExp.current_step} / {selectedExp.total_steps} Steps ({selectedExp.progress_percent.toFixed(1)}%)
                    </div>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${selectedExp.progress_percent}%` }}></div>
                  </div>
                </div>

                {/* Metric Cards */}
                <div className="grid-cards">
                  <div className="stat-card">
                    <div className="stat-label">Execution Delay</div>
                    <div className="stat-value">
                      {liveMetrics.length > 0 ? liveMetrics[liveMetrics.length - 1].delay_ms.toFixed(1) : '—'} <span style={{ fontSize: '1rem' }}>ms</span>
                    </div>
                    <div className="stat-subtext">Local vs Offloaded latency</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Energy Consumption</div>
                    <div className="stat-value">
                      {liveMetrics.length > 0 ? liveMetrics[liveMetrics.length - 1].energy_mj.toFixed(2) : '—'} <span style={{ fontSize: '1rem' }}>mJ</span>
                    </div>
                    <div className="stat-subtext">Tx + CMOS Computation</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Signaling Overhead</div>
                    <div className="stat-value">
                      {liveMetrics.length > 0 ? liveMetrics[liveMetrics.length - 1].overhead_kb.toFixed(3) : '—'} <span style={{ fontSize: '1rem' }}>kB</span>
                    </div>
                    <div className="stat-subtext">CSI, grant & handshake</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Task Success Rate</div>
                    <div className="stat-value" style={{ color: 'var(--teal-600)' }}>
                      {liveMetrics.length > 0 ? (liveMetrics[liveMetrics.length - 1].task_success_rate * 100).toFixed(1) + '%' : '—'}
                    </div>
                    <div className="stat-subtext">Deadline satisfaction</div>
                  </div>
                </div>

                {/* Learning Curves */}
                <div className="card">
                  <div className="card-title">Multi-Objective Learning Curves</div>
                  <div style={{ height: '320px', width: '100%' }}>
                    {liveMetrics.length === 0 ? (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                        Waiting for step metrics...
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={liveMetrics}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="step" label={{ value: 'Timestep', position: 'insideBottomRight', offset: -5 }} />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          <Line yAxisId="left" type="monotone" dataKey="delay_ms" name="Delay (ms)" stroke="#0d9488" dot={false} strokeWidth={2} />
                          <Line yAxisId="left" type="monotone" dataKey="energy_mj" name="Energy (mJ)" stroke="#f59e0b" dot={false} strokeWidth={2} />
                          <Line yAxisId="right" type="monotone" dataKey="task_success_rate" name="Success Rate" stroke="#10b981" dot={false} strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>

                {/* Solver Diagnostics Card */}
                <div className="card">
                  <div className="card-title">
                    <span>Inner CPU Convex Solver Diagnostics ({selectedExp.solver_name})</span>
                    <Cpu size={18} color="var(--teal-600)" />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                    <div className="stat-card">
                      <div className="stat-label">Mean Solver Duration</div>
                      <div className="stat-value" style={{ fontSize: '1.4rem' }}>
                        {liveMetrics.length > 0 ? liveMetrics[liveMetrics.length - 1].solver_mean_ms.toFixed(2) : '—'} ms
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Max Constraint Residual</div>
                      <div className="stat-value" style={{ fontSize: '1.4rem' }}>
                        {liveMetrics.length > 0 ? liveMetrics[liveMetrics.length - 1].solver_max_residual.toExponential(2) : '—'}
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Formulation Discrepancy</div>
                      <div className="stat-value" style={{ fontSize: '1.1rem', color: selectedExp.use_corrected_cpu_alloc ? 'var(--teal-600)' : 'var(--amber-500)' }}>
                        {selectedExp.use_corrected_cpu_alloc ? 'CORRECTED (Fronthaul incl.)' : 'LEGACY FORMULATION'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Subprocess Logs */}
                <div className="card">
                  <div className="card-title">
                    <span>Live Subprocess Log Stream</span>
                    <Terminal size={18} />
                  </div>
                  <div className="log-box">
                    {liveLogs.stdout ? liveLogs.stdout : 'No log output generated yet...'}
                  </div>
                </div>
              </div>
            ) : (
              <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
                <p>No experiment selected. Launch a new one or select an experiment from Overview.</p>
              </div>
            )}
          </div>
        )}

        {/* 4. COMPARISON TAB */}
        {activeTab === 'compare' && (
          <div>
            <div className="page-header">
              <h1 className="page-title">Policy & Preset Comparison</h1>
              <p className="page-desc">
                Evaluate empirical metrics across legacy baseline, delay-only, balanced, and priority presets.
              </p>
            </div>

            <div className="card">
              <div className="card-title">
                <span>Tested Policy Configurations</span>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => fetchComparisons().then((c) => setComparisons(c.experiments))}
                >
                  <RefreshCw size={14} /> Refresh
                </button>
              </div>

              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Experiment</th>
                      <th>Preset</th>
                      <th>Seed</th>
                      <th>4D Non-Dominated</th>
                      <th>Delay (ms)</th>
                      <th>Energy (mJ)</th>
                      <th>Overhead (kB)</th>
                      <th>Reliability</th>
                      <th>Reward</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisons.length === 0 ? (
                      <tr>
                        <td colSpan={9} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                          No completed runs found for comparison. Run training or evaluation to populate results.
                        </td>
                      </tr>
                    ) : (
                      comparisons.map((c) => (
                        <tr key={c.id}>
                          <td><strong>{c.name}</strong></td>
                          <td>{c.preset}</td>
                          <td>{c.seed}</td>
                          <td>
                            {c.is_nondominated ? (
                              <span className="badge badge-nd">EMPIRICAL NON-DOM</span>
                            ) : (
                              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Dominated</span>
                            )}
                          </td>
                          <td>{c.delay_ms.toFixed(1)}</td>
                          <td>{c.energy_mj.toFixed(2)}</td>
                          <td>{c.overhead_kb.toFixed(3)}</td>
                          <td>{(c.task_success_rate * 100).toFixed(1)}%</td>
                          <td>{c.reward.toFixed(4)}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 5. TRADEOFF ANALYSIS TAB */}
        {activeTab === 'tradeoff' && (
          <div>
            <div className="page-header">
              <h1 className="page-title">Objective Tradeoff & Frontier Projections</h1>
              <p className="page-desc">
                2D empirical projections of tested configurations. Gold and teal badges indicate 4D non-dominance across all four objectives.
              </p>
            </div>

            <div className="card">
              <div style={{ display: 'flex', gap: '24px', marginBottom: '20px', alignItems: 'center' }}>
                <div>
                  <label className="form-label">X Axis Projection</label>
                  <select
                    className="form-select"
                    value={scatterX}
                    onChange={(e: any) => setScatterX(e.target.value)}
                  >
                    <option value="delay_ms">Execution Delay (ms) [Min]</option>
                    <option value="energy_mj">Energy Consumption (mJ) [Min]</option>
                    <option value="overhead_kb">Overhead (kB) [Min]</option>
                    <option value="task_success_rate">Task Success Rate [Max]</option>
                  </select>
                </div>
                <div>
                  <label className="form-label">Y Axis Projection</label>
                  <select
                    className="form-select"
                    value={scatterY}
                    onChange={(e: any) => setScatterY(e.target.value)}
                  >
                    <option value="energy_mj">Energy Consumption (mJ) [Min]</option>
                    <option value="delay_ms">Execution Delay (ms) [Min]</option>
                    <option value="overhead_kb">Overhead (kB) [Min]</option>
                    <option value="task_success_rate">Task Success Rate [Max]</option>
                  </select>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '20px' }}>
                  Note: 2D visual boundary is a projection; full dominance is calculated in 4 dimensions.
                </div>
              </div>

              <div style={{ height: '420px', width: '100%' }}>
                {comparisons.length === 0 ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                    No completed comparison data available yet.
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey={scatterX} name={scatterX} />
                      <YAxis type="number" dataKey={scatterY} name={scatterY} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Legend />
                      <Scatter
                        name="4D Non-Dominated Tested Policies"
                        data={comparisons.filter((c) => c.is_nondominated)}
                        fill="#0d9488"
                        shape="circle"
                      />
                      <Scatter
                        name="Dominated Configurations"
                        data={comparisons.filter((c) => !c.is_nondominated)}
                        fill="#94a3b8"
                        shape="cross"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 6. ARTIFACTS TAB */}
        {activeTab === 'artifacts' && (
          <div>
            <div className="page-header">
              <h1 className="page-title">Experiment Artifacts & Reproducibility Files</h1>
              <p className="page-desc">
                Download structured metrics (CSV, JSON), configuration metadata, TensorBoard events, and PyTorch actor/critic checkpoints.
              </p>
            </div>

            <div className="card">
              <div className="card-title">
                <span>Artifact Explorer for {selectedExp?.name || 'Selected Run'}</span>
                <select
                  className="form-select"
                  style={{ width: '220px' }}
                  value={selectedExpId || ''}
                  onChange={(e) => setSelectedExpId(e.target.value)}
                >
                  {experiments.map((e) => (
                    <option key={e.id} value={e.id}>
                      {e.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Filename</th>
                      <th>Type</th>
                      <th>Size (KB)</th>
                      <th>Download</th>
                    </tr>
                  </thead>
                  <tbody>
                    {artifacts.length === 0 ? (
                      <tr>
                        <td colSpan={4} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                          No artifact files found for this experiment directory.
                        </td>
                      </tr>
                    ) : (
                      artifacts.map((a, idx) => (
                        <tr key={idx}>
                          <td><code>{a.name}</code></td>
                          <td><span className="badge badge-queued">{a.type}</span></td>
                          <td>{(a.size_bytes / 1024).toFixed(1)} KB</td>
                          <td>
                            {selectedExpId && (
                              <a
                                href={getArtifactDownloadUrl(selectedExpId, a.name)}
                                target="_blank"
                                rel="noreferrer"
                                className="btn btn-secondary btn-sm"
                              >
                                <Download size={14} /> Download
                              </a>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 7. FEDERATED LEARNING TAB */}
        {activeTab === 'federation' && (
          <FederatedView presets={presets} capabilities={capabilities} />
        )}
      </main>
    </div>
  );
}

export default App;
