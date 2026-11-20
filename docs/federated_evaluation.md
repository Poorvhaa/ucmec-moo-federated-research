# Federated Evaluation Protocol and Benchmark Comparisons

## 1. Experimental Methodology: Matched Comparisons

To evaluate the true empirical utility of federated multi-agent learning without misleading claims, the platform implements three strictly matched methods evaluated under identical physical and transition budgets:

1. **Local-Only PPO**:
   - Each edge site trains its local user agents using its local environment and local critic with no inter-site communication ($\text{Bytes} = 0$).
   - Evaluates the baseline performance achieved by siloed local experience.
2. **Federated PPO (FedAvg-PPO)**:
   - Synchronous sample-weighted aggregation of actor parameters across edge sites, with local personalized critics.
   - Evaluates knowledge transfer across diverse edge sites while keeping raw trajectories local.
3. **Pooled-Data Reference (Centralized Trajectory PPO)**:
   - Upper-bound benchmark where fresh trajectories from all participating edge sites are aggregated in a centralized buffer.
   - Updates a single shared actor network using the exact same total transition count, minibatch count, learning rate, and PPO epochs as the federated counterpart.
   - *Unavailable flag*: If matched trajectory pooling cannot be executed for an arbitrary custom topology, it is explicitly flagged as unavailable rather than misrepresenting decentralized MAPPO as equivalent.

### 1.1 Matched Budget Invariants

For any head-to-head comparison:
- **Total Environment Transitions**: Exactly matched ($\sum_{k} T_{\text{env}}^{(k)}$).
- **Per-Site Rollout Steps**: Identical per round.
- **Optimizer Steps / Epochs**: Equal number of gradient update steps per transition.
- **Neural Network Architecture**: Exact same layer count ($2 \times 64$), activations, LayerNorm, and initialization.
- **Reward Presets & Metric Contract**: Shared reference scales and objective scalarization weights.
- **Evaluation Scenarios & Seeds**: Evaluated on identical held-out test scenarios using deterministic policy evaluation (`deterministic=True`).

---

## 2. Evaluation Metrics and Fairness Statistics

### 2.1 Multi-Objective Metrics
Evaluated on common held-out environments across independent evaluation seeds:
- **Delay (ms)**: Average end-to-end task execution delay (transmission + fronthaul + edge CPU compute).
- **Energy (mJ)**: Average device energy consumption.
- **Overhead (kB)**: Average fronthaul signaling overhead.
- **Reliability (%)**: Percentage of tasks completed within the physical deadline ($t \le 100\text{ ms}$).

### 2.2 Client Dispersion & Worst-Case Metrics
A federation must not improve average metrics at the expense of crippling individual sites. We report:
- **Per-Client Breakdown**: Objective metrics for each individual client site $k$.
- **Worst-Client Performance**:
  - $\max_{k} \text{Delay}_k$ (worst site latency).
  - $\min_{k} \text{Reliability}_k$ (worst site task success rate).

### 2.3 Jain's Fairness Index
To quantify equitable performance distribution across heterogeneous client sites, the platform computes **Jain's Fairness Index**:
$$\mathcal{J}(x_1, x_2, \dots, x_K) = \frac{\left(\sum_{k=1}^K x_k\right)^2}{K \cdot \sum_{k=1}^K x_k^2} \in \left[\frac{1}{K}, 1.0\right]$$
where $x_k$ represents client $k$'s mean task success rate (or scalar reward utility). A score of $1.0$ indicates perfect parity across sites.

---

## 3. Communication vs. Performance Trade-Offs

The platform logs and plots the Pareto front between:
- Total Model Exchange Bytes ($\text{MB}$ or $\text{GB}$) transferred over the network.
- Convergence rate and final test-set reliability/delay.
- Tradeoff comparisons between Local-Only ($0\text{ MB}$), Federated (periodic parameter transfers), and Pooled-Data references.
