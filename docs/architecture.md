# System Architecture: Multi-Objective UCMEC Platform

## 1. System Overview

The platform integrates a high-fidelity wireless mobile edge computing (MEC) simulator with multi-objective reinforcement learning, conic optimization solvers, an asynchronous execution engine, a REST API, and a real-time research dashboard.

```
+-------------------------------------------------------------------------------+
|                               Frontend Dashboard                              |
|   (React 19 + TypeScript + Vite + Recharts + Lucide-React - Navy/Teal Theme)  |
|   - Overview     - Experiment Creation  - Live Monitor / Charts / Logs        |
|   - Comparisons  - 2D Tradeoff Frontier - Artifact Download Explorer          |
+-------------------------------------------------------------------------------+
                                        │ HTTP / REST
                                        ▼
+-------------------------------------------------------------------------------+
|                               FastAPI Backend                                 |
|   (Endpoints: /api/health, /capabilities, /presets, /experiments, /compare)   |
|   - Input Validation (Pydantic)  - Path Traversal Shield                      |
|   - SQLite Persistence           - Interrupted Job Reconciler                 |
+-------------------------------------------------------------------------------+
                                        │ Queue
                                        ▼
+-------------------------------------------------------------------------------+
|                       Durable Background Worker Process                       |
|   - Concurrency Limiter (Default 1)  - Safe Process Trees (taskkill / SIGKILL)|
|   - Subprocess Argument Lists        - Incremental Metric Persistence (JSON/CSV)|
+-------------------------------------------------------------------------------+
                                        │ Launch
                                        ▼
+-------------------------------------------------------------------------------+
|                      MARL Simulator Engine (train/train.py)                   |
|   - Separated / Shared Policy Runners (MAPPO / IPPO)                          |
|   - DummyVecEnv + DiscreteActionEnv                                           |
|   - Proper Time-Limit Truncation vs Terminal Bootstrapping (bad_masks)        |
+---------------------------------------+---------------------------------------+
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
+---------------------------------------+ +-------------------------------------+
|        Conic Inner Solver (moo/)      | |     Multi-Objective Contract (moo/) |
|   - CLARABEL (Primal-Dual Conic)      | |   - Delay, Energy, Overhead, Rel    |
|   - Normalized Capacity Fractions     | |   - Frozen Reference Normalization  |
|   - Residual & Duration Logging       | |   - 4D Non-Dominance Evaluation     |
+---------------------------------------+ +-------------------------------------+
```

## 2. Key Subsystems

### A. Two-Level Decision Engine
1. **Level 1 (Agent Action)**:
   Each user agent observes its local state (previous offloading target, previous transmit power, previous experienced delay) and outputs a discrete action $a_i \in \{0, \dots, 9\}$. The action maps to:
   - Offloading choice $\omega_i \in \{0, 1, 2, 3\}$ (0: local, 1..3: Edge CPU).
   - Transmit power level $p_i \in \{0.025, 0.05, 0.075, 0.1\}\text{ W}$.
2. **Level 2 (Inner Conic Optimization)**:
   For each edge processor $k$, all offloaded users are grouped into $\mathcal{U}_k$. The edge CPU solves a convex allocation problem to divide its total clock capacity $C_k$ among users using normalized fractions $f_j \in (0, 1]$ such that $\sum f_j \le 1.0$.

### B. Durable Worker & Subprocess Safety
- The worker runs asynchronously from the FastAPI event loop, ensuring HTTP requests are never blocked.
- Jobs are queued durably in SQLite.
- On Windows, child processes are terminated cleanly using `taskkill /F /T /PID`, preventing orphaned processes.
- On Linux, process group termination (`os.killpg`) is utilized.
- Subprocesses are spawned exclusively via verified argument lists (`sys.executable, "train/train.py", ...`), strictly prohibiting shell string concatenation.

### C. Path Traversal & Security Protection
- Artifact endpoints verify canonical paths against allowed roots (`experiments_storage/`, `results/`) using `Path.resolve().is_relative_to(...)`. Any traversal escape (`..`) returns HTTP 404 immediately.
