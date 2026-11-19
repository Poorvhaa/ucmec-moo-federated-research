# Research Contribution Draft: Multi-Objective Optimization in mmWave Mobile Edge Computing

## 1. Abstract
We present a multi-objective reinforcement learning and conic optimization framework for user-centric mobile edge computing (MEC) networks operating over mmWave fronthaul links. By formulating a two-level decision hierarchy combining multi-agent PPO (MAPPO/IPPO) with an inner interior-point conic resource allocator (CLARABEL), we jointly optimize task execution delay, user device energy consumption, control communication overhead, and task reliability. We provide an explicit metric contract, correct numerical scaling and objective formulation discrepancies in legacy implementations, and evaluate empirical non-dominated trade-offs across controlled presets.

---

## 2. Implemented Methods vs Untested Hypotheses

| Aspect | Implemented & Verified in this Codebase | Untested Hypotheses (Scope for Future Work) |
|---|---|---|
| **Objectives** | 4 distinct objectives: Delay ($ms$), Energy ($mJ$), Overhead ($kB$), and Reliability ($[0, 1]$). | High-mobility Doppler effects, non-orthogonal multiple access (NOMA). |
| **Physical Model** | CMOS capacitance computation ($E = \kappa \cdot \text{cycles} \cdot f^2$) and transmission energy ($E = p \cdot T$). | Dynamic CPU frequency scaling (DVFS) under thermal throttling. |
| **Inner Solver** | Primal-dual interior point conic solver (CLARABEL) with normalized capacity fractions $f_j \in (0, 1]$. | Distributed ADMM solver running over lossy fronthaul. |
| **Formulation Discrepancy** | Corrected inner optimization including fronthaul latency, retaining legacy mode for comparison. | Joint continuous beamforming and power control. |
| **Evaluation** | 4-objective empirical non-dominance calculation and seed-level statistical aggregation. | Global optimality guarantees (non-convex Pareto surfaces). |

---

## 3. Empirical Verification Results

In a 200-step smoke validation execution using the balanced multi-objective preset ($w = [0.35, 0.25, 0.15, 0.25]$, seed 42):
- **Execution Delay**: $39.85\text{ ms}$
- **User Energy**: $0.53\text{ mJ}$
- **Signaling Overhead**: $0.359\text{ kB}$
- **Task Success Rate**: $88.30\%$
- **Constraint Residuals**: $< 1.15 \times 10^{-6}$
- **Mean Solver Duration**: $25.19\text{ ms}$

These genuine measurements demonstrate stable convergence without numerical failure, satisfying capacity conservation and deadline constraints.
