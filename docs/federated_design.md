# Federated Multi-Agent Learning Architecture and System Design

## 1. Domain Entities and Hierarchy

To maintain physical and mathematical rigor in this simulated cell-free mmWave Multi-access Edge Computing (MEC) platform, domain roles are strictly distinguished as follows:

| Entity | Representation in Simulator | Role in Learning Architecture |
| :--- | :--- | :--- |
| **User Agent (UE)** | Mobile device / user equipment (simulated via $M_{\text{sim}}$ agents). | **RL Decision Agent**: Observes local state $[r_{i}(t-1)/\text{scale}, p_{i}^{\text{last}}, d_{i}^{\text{last}}]$ (dim 3) and executes discrete action $a_i \in \{0, \dots, 9\}$ choosing local execution or offloading to one of $K$ edge CPUs at transmit power levels $p \in \{1, 2, 3\}$. |
| **Access Point (AP)** | Distributed antenna infrastructure ($N_{\text{sim}} = 50$ APs) receiving wireless uplink signals and forwarding over mmWave links. | **Network Environment Infrastructure**: Forms cell-free serving clusters per user; channels experience path-loss, shadow fading, and LOS/NLOS mmWave blockage. |
| **Edge CPU (MEC Host)** | Computing servers ($K = 3$ CPUs) with bounded capacity $C_{\text{edge}}$ (e.g. 10–20 GHz). | **Resource Optimizer**: Executes convex conic optimization (CLARABEL/SCS) to allocate CPU cycles across offloaded user tasks. |
| **Edge Site (Federated Client)** | A self-contained cell-free mmWave deployment (e.g., Campus A, Factory B, Hospital C) containing its own UEs, APs, edge CPUs, channel statistics, and task workloads. | **Federated Client**: Collects fresh local rollouts, computes local PPO updates with a personalized local critic, and submits updated actor parameters plus fresh sample counts to the coordinator. |
| **Coordinator (Federation Server)** | Central aggregation service. | **Federated Aggregator**: Broadcasts versioned global actor checkpoints, validates client submissions, performs sample-weighted FedAvg aggregation, checkpoints global rounds, and tracks communication accounting. |

> [!IMPORTANT]
> User agents are **not** AP agents. The reinforcement learning agents represent mobile user devices offloading computational tasks over access and mmWave fronthaul links.

---

## 2. Policy Sharing and Critic Decomposition

In the single-site baseline:
- When `share_policy=True` (default), all user agents at the site share a single actor neural network parameter set ($\theta_{\text{actor}}$).
- When `use_centralized_V=True` (default), the critic observes the concatenated centralized observations of all local agents ($3 \times M_{\text{sim}}$ dimensions).

In the **Federated Multi-Agent Learning** system:
1. **Compatible Actor Aggregation**:
   - For homogeneous user agent roles across sites, the client actors share identical input/output schemas (observation dim 3, action dim 10).
   - Only the floating-point trainable parameters of the actor network ($\theta_{\text{actor}}$) are submitted to the coordinator and aggregated across participating sites.
   - Non-trainable buffers (such as step counters) are handled explicitly and are never averaged.
2. **Personalized Local Critics**:
   - Centralized critics ($V_{\phi}$) remain strictly **local and personalized** to each edge site.
   - They are **never** aggregated across sites.
   - *Rationale*: Centralized critics depend on site-specific agent counts, local AP topologies, and local value baselines. Aggregating critics across heterogeneous site configurations would disrupt local value estimation and advantage calculations.
3. **Local Value Normalizers**:
   - Running value statistics (`ValueNorm` / PopArt) remain local to each site's personalized critic and are never merged or averaged.

---

## 3. Mathematical Formulation: Synchronous FedAvg-PPO

### 3.1 Round Lifecycle

Each global communication round $r \in \{1, 2, \dots, R\}$ proceeds synchronously:
1. **Client Selection**: The coordinator selects a subset of eligible clients $\mathcal{S}_r \subseteq \{1, \dots, K\}$ according to the configured participation fraction $C \in (0, 1]$ or full participation. A minimum required client count $K_{\min}$ is enforced.
2. **Global Actor Broadcast**: Selected clients receive the current versioned global actor weights $\theta_{\text{global}}^{(r-1)}$ (version $r-1$).
3. **Buffer Flush & Fresh Rollouts**:
   - Each client clears all obsolete rollout buffers. Stale trajectories from previous rounds are **never** retained or pooled.
   - The client loads $\theta_{\text{global}}^{(r-1)}$ into its actor network.
   - Fresh on-policy trajectories are collected by interacting with the client's local multi-agent environment for $E_{\text{local}}$ episodes ($T_{\text{rollout}}$ steps).
4. **Local PPO Updates**:
   - The client computes generalized advantage estimates (GAE) using its local critic $V_{\phi_k}$.
   - The client performs $U_{\text{local}}$ local PPO optimization epochs using minibatch gradient descent to update $\theta_{k}$ and local critic $\phi_k$.
5. **Update Submission**:
   - Client submits the updated actor parameter dictionary $\theta_k^{(r)}$ along with $n_k^{(r)}$, defined strictly as the number of **fresh valid training transitions** generated in this round (excluding masked or discarded steps).
   - Client also reports local multi-objective metrics (delay, energy, overhead, reliability, reward, loss) and operational metadata.
6. **Coordinator Validation**:
   - Checks model version basis (rejects updates where base version $\ne r-1$).
   - Verifies client eligibility (rejects unsolicited or duplicate updates).
   - Verifies tensor names, shapes, and floating-point dtypes against the canonical actor schema.
   - Rejects updates containing non-finite values (`NaN` or `Inf`).
   - Rejects updates with $n_k^{(r)} \le 0$.
   - If accepted clients count $|\mathcal{A}_r| < K_{\min}$, the round fails according to the configured failure policy.
7. **Sample-Weighted Aggregation**:
   The new global actor parameters are computed via sample-weighted FedAvg in float32/float64:
   $$\theta_{\text{global}}^{(r)} = \sum_{k \in \mathcal{A}_r} \frac{n_k^{(r)}}{\sum_{j \in \mathcal{A}_r} n_j^{(r)}} \, \theta_k^{(r)}$$
8. **Atomic Checkpointing & Evaluation**:
   - The coordinator writes $\theta_{\text{global}}^{(r)}$ using atomic rename semantics (`.pt.tmp` $\to$ `.pt`).
   - The global actor is evaluated on held-out test scenarios common across methods.

---

## 4. Optimizer and Normalization Semantics

1. **Actor Optimizer Reset**:
   - **Default Behavior**: When receiving the aggregated global parameters $\theta_{\text{global}}^{(r-1)}$, each client resets its actor Adam optimizer state (momentum $m_t$ and second moment $v_t$ are cleared).
   - **Rationale**: FedAvg displaces parameters to a point not reached along the local optimizer trajectory. Using historical momentum from the previous local trajectory introduces stale directional bias inconsistent with the new global weights.
2. **Critic Optimizer Persistence**:
   - The local critic optimizer state persists uninterrupted across rounds, maintaining continuous estimation of the site's local value function.
3. **Observation Normalization**:
   - The network uses `MLPBase` with `nn.LayerNorm(obs_dim)` when `use_feature_normalization=True`.
   - LayerNorm parameters (`weight`, `bias`) are trainable floating-point tensors belonging to the actor network and are aggregated along with linear layer weights.
   - Raw observations have fixed physical bounds: rate $[0, 2]$, power $[0, P_{\max}]$, delay $[0, 0.2]$. Policy input semantics remain strictly compatible across sites.
4. **Learning Rate Schedule**:
   - Learning rate decays linearly across global rounds: $\eta_r = \eta_0 \cdot (1 - \frac{r-1}{R})$.

---

## 5. Theoretical Limitations & Scientific Caveats

> [!WARNING]
> **Federated PPO is an Empirical Approximation**:
> 1. Standard PPO relies on on-policy trajectory sampling with bounded policy divergence ($\text{clip}(\rho, 1-\epsilon, 1+\epsilon)$). In federated PPO with multiple local updates, local policies drift towards local site optima ($\theta_k \ne \theta_{\text{global}}$).
> 2. Averaging parameters $\sum w_k \theta_k$ in non-convex neural network parameter space does not guarantee monotonicity of policy improvement, nor does it establish theoretical convergence without unrealistic convexity or homogeneity assumptions.
> 3. FedAvg aggregation does not automatically guarantee Pareto-optimal compromises under client objective heterogeneity.
