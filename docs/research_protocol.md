# Research Evaluation Protocol

## 1. Experimental Protocol & Rigor

To evaluate multi-objective policies reliably without methodological confounding, adhere to the following protocol:

### A. Independent Seed Level Aggregation
- Never aggregate correlated rollout steps within a single training episode as independent samples.
- Compute performance statistics across independent training and evaluation seeds ($S \ge 3$, e.g. seeds 42, 43, 44).
- Report both sample mean and sample standard deviation ($\mu \pm \sigma$) for every objective individually:
  - $\text{Delay}$ ($ms$)
  - $\text{Energy}$ ($mJ$)
  - $\text{Overhead}$ ($kB$)
  - $\text{Reliability}$ (Task success rate $[0, 1]$)

### B. Held-Out Evaluation Scenarios
- Fix held-out random evaluation scenarios across all compared policies so every policy experiences identical channel realizations, user positions, and task arrivals.
- Evaluate with stochastic exploration disabled (`deterministic=True`).

### C. 4-Dimensional Empirical Non-Dominance
- A policy $A$ dominates policy $B$ iff:
  $$T_A \le T_B \land E_A \le E_B \land O_A \le O_B \land \mathcal{R}_A \ge \mathcal{R}_B$$
  with at least one strict inequality.
- Results represent an empirical non-dominated set among tested configurations; a weighted sum sweep does not uncover non-convex regions of the true Pareto frontier.
- Visual 2D scatter plots must be interpreted as projections of the 4D space, not the complete 4D boundary.

### D. Reproduction Commands

To run a fast smoke comparison across the 5 presets:
```powershell
conda run -n ucmec python evaluate.py --presets legacy delay_only balanced energy_priority reliability_priority --seeds 42 43 44 --num_steps 20
```

To run a standard 200-step evaluation sweep:
```powershell
conda run -n ucmec python evaluate.py --presets legacy delay_only balanced energy_priority reliability_priority --seeds 42 43 44 --num_steps 200
```
