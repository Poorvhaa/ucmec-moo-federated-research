# Changelog: Baseline Repairs & Research Enhancements

This log records all baseline repairs, structural bug fixes, and research features added to `UCMEC-mmWave-Fronthaul`.

---

## 1. Baseline Bug Fixes & Architectural Repairs

### A. Syntax & Indentation Repair
- **Issue**: A pasted edit near `capacity = float(self.C_edge[i, 0])` and `reset()` in `envs/MA_UCMEC_stat_noncoop.py` introduced mixed indentation (16-space block instead of 8-space method indent) causing `IndentationError`.
- **Fix**: Restored standard 8-space indentation across `reset()`, variable resets, and CPU allocation blocks. Verified with `py_compile`.

### B. State Reset Completeness
- **Issue**: Previous actions and previous delay tracking were not properly initialized on environment reset.
- **Fix**: `reset()` in `MA_UCMEC_stat_noncoop` now explicitly resets `step_num = 0`, `omega_last.fill(0)`, `p_last.fill(0)`, and `delay_last.fill(0)`.

### C. Seeded Reproducibility Restoration
- **Issue**: `DiscreteActionEnv.seed(seed)` had a no-op `pass` implementation, preventing random seeds passed in `train.py` from reaching the underlying simulation environment. The environment also initialized static locations with a hardcoded seed `np.random.seed(37)` without reseeding.
- **Fix**: Implemented `seed(seed)` in `DiscreteActionEnv` and `MA_UCMEC_stat_noncoop` to reseed NumPy random generators (`np.random.seed(seed)`), ensuring identical trajectories for identical seeds.

### D. Time-Limit Truncation vs Terminal Bootstrapping (`bad_masks`)
- **Issue**: In `env_runner`, reaching the 200-step horizon flagged `dones = True`, which zeroed out `masks`. In GAE, setting `masks[step+1] = 0` zeroes out $V(s_{t+1})$, incorrectly treating horizon truncation as terminal agent death/failure rather than time-limit truncation.
- **Fix**: Flagged `TimeLimit.truncated = True` and `bad_transition = True` in step info. In `env_runner.insert()`, constructed `bad_masks`. When `use_proper_time_limits` is enabled, `masks` remains 1.0 at truncation to preserve value bootstrapping while `bad_masks` cuts GAE trace recursion.

### E. Conic CPU Allocation Scaling & Residual Logging
- **Issue**: Raw task sizes in bits ($10^5$) and edge capacities in Hz ($10^{10}$) led to numerical ill-conditioning and solver stalls in SCS.
- **Fix**: Re-formulated inner allocation using normalized capacity fractions $f_j \in (0, 1]$ with $\sum f_j \le 1.0$. Adopted CLARABEL interior-point conic solver. Added constraint residual logging ($| \sum f_j - 1.0 |$) and duration tracking.

### F. Objective Discrepancy Correction
- **Issue**: The inner solver objective omitted fronthaul delay ($T_{\text{front}}$) while the outer total delay calculation included it ($T_{\text{front}} + T_{\text{uplink}} + T_{\text{process}}$).
- **Fix**: Introduced `use_corrected_cpu_alloc` flag. Corrected formulation includes $T_{\text{front}}$ in the optimization objective. Preserved uncorrected legacy mode via `--legacy_cpu_alloc` for comparison.

### G. Flag Semantics Documentation (`store_false`)
- **Finding**: In `config.py`, `--share_policy`, `--use_centralized_V`, and `--cuda` are configured with `action="store_false", default=True`. Passing `--cuda` sets `cuda=False` (CPU mode), and passing `--share_policy` disables policy sharing (IPPO mode). Documented this explicitly to prevent misleading algorithm attribution.

---

## 2. Multi-Objective Research Features

### A. 4-Objective Metric Contract
- Created `moo/metrics.py` defining:
  - Execution Delay (Local vs Offload).
  - Energy (Device transmission energy + CMOS computation energy model).
  - Communication Overhead (Task payload vs control signaling model).
  - Reliability (Latency deadline satisfaction + SINR channel outage model).
- Stored internally in SI units (s, J, Bytes), displayed in ms, mJ, kB.

### B. Configurable Scalarization & Presets
- Implemented in `moo/scalarization.py`:
  - `legacy`: Original $-0.9 T + 0.1(\tau_c - T)$.
  - `delay_only`: $w = [1.0, 0.0, 0.0, 0.0]$.
  - `balanced`: $w = [0.35, 0.25, 0.15, 0.25]$.
  - `energy_priority`: $w = [0.15, 0.60, 0.10, 0.15]$.
  - `reliability_priority`: $w = [0.15, 0.15, 0.10, 0.60]$.
  - `custom`: User-configured weights.

### C. 4D Non-Dominance & Pareto Filtering
- Implemented in `moo/pareto.py`: Computes empirical non-dominance in the 4-dimensional objective space (minimizing Delay, Energy, Overhead; maximizing Reliability).

### D. Structured Incremental Logging
- Implemented `moo/logging_helper.py`: Writes `metrics.csv`, `metrics.json`, `moo_summary.json`, and TensorBoard events incrementally.

### E. CLI Evaluation & Comparison Suite
- Created `evaluate.py`: Evaluates policies on held-out scenarios across independent seeds with deterministic actions, reporting mean $\pm$ standard deviation.

### F. Full-Stack Backend & Frontend Dashboard
- **Backend**: FastAPI app with SQLite database, durable background execution worker, cancel support, and path traversal protection.
- **Frontend**: Single-page application built with React 19, TypeScript, Vite, Recharts, and Lucide-React.
