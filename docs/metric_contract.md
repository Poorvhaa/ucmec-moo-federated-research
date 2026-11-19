# Metric Contract: Multi-Objective Wireless Edge Computing Simulator

This document establishes the formal metric contract, physical derivations, units, aggregation rules, assumptions, and limitations for the multi-objective optimization (MOO) extension in `UCMEC-mmWave-Fronthaul`.

---

## 1. Objective Definitions & Mathematical Models

The simulator optimizes four distinct system objectives across $M$ mobile users and $K$ edge computing processors connected via mmWave fronthaul:

$$\min \text{Delay}, \quad \min \text{Energy}, \quad \min \text{Overhead}, \quad \max \text{Reliability}$$

### A. Execution Delay ($T$)
- **Internal Unit**: Seconds ($s$)
- **Display Unit**: Milliseconds ($ms = s \times 10^3$)
- **Decision Variable**: Binary offloading action $\omega_i \in \{0, 1, \dots, K\}$:
  - $\omega_i = 0$: Local processing on user equipment (UE).
  - $\omega_i = k > 0$: Offloading to edge processor $k$ via mmWave access and fronthaul.
- **Formulation**:
  - **Local Delay**:
    $$T_{\text{local}, i} = \frac{\rho_i \cdot D_i}{f_{\text{user}, i}}$$
    where $\rho_i$ is task computational density (cycles/bit), $D_i$ is task size (bits), and $f_{\text{user}, i}$ is device CPU clock frequency (Hz).
  - **Offloaded Delay**:
    $$T_{\text{offload}, i} = T_{\text{uplink}, i} + T_{\text{front}, i} + T_{\text{process}, i}$$
    where:
    - Uplink transmission delay: $T_{\text{uplink}, i} = \frac{D_i}{R_{\text{access}, i}}$
    - mmWave fronthaul delay: $T_{\text{front}, i} = \max_{j \in \mathcal{A}_i} \frac{D_i}{R_{\text{front}, i, j}}$ ($\mathcal{A}_i$ is the serving AP cluster).
    - Edge processing delay: $T_{\text{process}, i} = \frac{\rho_i \cdot D_i}{C_{\text{actual}, i, k}}$ ($C_{\text{actual}, i, k}$ is allocated edge CPU in Hz).
  - **Total Per-User Delay**:
    $$T_i = \begin{cases} T_{\text{local}, i}, & \text{if } \omega_i = 0 \\ T_{\text{offload}, i}, & \text{if } \omega_i > 0 \end{cases}$$
- **Aggregation**: Episode mean delay $\bar{T} = \frac{1}{M} \sum_{i=1}^M T_i$.

---

### B. Energy Consumption ($E$)
- **Internal Unit**: Joules ($J$)
- **Display Unit**: Millijoules ($mJ = J \times 10^3$)
- **Physical Models**:
  - **Mobile Device Local Computation Energy**:
    Follows standard CMOS dynamic capacitance dissipation ($P = \kappa f^3$):
    $$E_{\text{local}, i} = \kappa_{\text{user}} \cdot (\rho_i D_i) \cdot (f_{\text{user}, i})^2$$
    where $\kappa_{\text{user}} = 10^{-28} \text{ J} / (\text{cycle} \cdot \text{Hz}^2)$.
  - **Mobile Device Wireless Transmission Energy**:
    $$E_{\text{tx}, i} = p_i \cdot T_{\text{uplink}, i}$$
    where $p_i$ is the actual allocated transmit power (Watts).
  - **User Device Total Energy**:
    $$E_i = \begin{cases} E_{\text{local}, i}, & \text{if } \omega_i = 0 \\ E_{\text{tx}, i}, & \text{if } \omega_i > 0 \end{cases}$$
  - **Edge & Fronthaul Network Modeled Energy (Informational)**:
    - AP Fronthaul Tx: $E_{\text{front}, i} = p_{\text{ap}} \cdot T_{\text{front}, i}$ ($p_{\text{ap}} = 1.0\text{ W}$).
    - Edge Server Comp: $E_{\text{edge}, i} = \kappa_{\text{edge}} \cdot (\rho_i D_i) \cdot (C_{\text{actual}, i, k})^2$.
- **Aggregation**: Episode mean user energy $\bar{E} = \frac{1}{M} \sum_{i=1}^M E_i$.

---

### C. Communication Overhead ($O$)
- **Internal Unit**: Bytes ($B$)
- **Display Unit**: Kilobytes ($kB = B / 1024$)
- **Definitions**:
  - **Task Payload**: Transmitted raw task bits $S_{\text{payload}, i} = D_i / 8$ Bytes (only when offloading).
  - **Signaling / Control Overhead**:
    Signaling bytes exchanged for channel sounding, scheduling, and synchronization:
    $$S_{\text{sig}, i} = B_{\text{csi}} + B_{\text{req}} + B_{\text{grant}} + B_{\text{front\_sig}} + B_{\text{ack}}$$
    Default values:
    - $B_{\text{csi}} = 128\text{ B}$ (pilot sounding / CSI feedback)
    - $B_{\text{req}} = 64\text{ B}$ (offloading request)
    - $B_{\text{grant}} = 64\text{ B}$ (scheduling grant)
    - $B_{\text{front\_sig}} = 128\text{ B}$ (AP-to-CPU routing)
    - $B_{\text{ack}} = 32\text{ B}$ (download ACK)
    - Total per offload event: $416\text{ B}$.
  - **Overhead Handling**: Local execution requires 0 over-the-air communication ($O_i = 0$).

---

### D. Reliability / Task Success Rate ($\mathcal{R}$)
- **Internal Unit**: Dimensionless fraction in $[0, 1]$
- **Display Unit**: Percentage ($\%$)
- **Success Criteria**:
  A task succeeds if and only if three joint conditions are fulfilled:
  1. **Latency Deadline Satisfaction**: $T_i \le T_{\text{deadline}}$ (default $T_{\text{deadline}} = \tau_c = 100\text{ ms}$).
  2. **Wireless Channel Feasibility**:
     - Local tasks: Always $1.0$ (no wireless channel).
     - Offloaded tasks: Instantaneous access SINR $\gamma_{\text{access}, i} \ge \gamma_{\text{th}}$ and fronthaul SINR $\gamma_{\text{front}, i} \ge \gamma_{\text{front\_th}}$.
  3. **Valid Edge Computation**: Allocation $C_{\text{actual}, i} > 0$ and finite.
- **Aggregation**:
  $$\text{Task Success Rate} = \frac{1}{M} \sum_{i=1}^M \mathbb{I}(\text{Success}_i)$$

---

## 2. Multi-Objective Scalarization & Reward Formulation

The multi-objective reward is structured as the negative weighted normalized cost:

$$\text{cost}_i = w_D \left(\frac{T_i}{\bar{T}_{\text{ref}}}\right) + w_E \left(\frac{E_i}{\bar{E}_{\text{ref}}}\right) + w_O \left(\frac{O_i}{\bar{O}_{\text{ref}}}\right) + w_R (1 - \text{Success}_i)$$

$$\text{reward}_i = -\text{cost}_i$$

### Scalarization Constraints:
- Nonnegativity: $w_D, w_E, w_O, w_R \ge 0$.
- Unity normalization: $w_D + w_E + w_O + w_R = 1.0$.
- Fixed reference scales:
  - $\bar{T}_{\text{ref}} = 0.100\text{ s}$ ($100\text{ ms}$)
  - $\bar{E}_{\text{ref}} = 0.005\text{ J}$ ($5.0\text{ mJ}$)
  - $\bar{O}_{\text{ref}} = 1000\text{ B}$
- **No per-episode min-max normalization**: Reference scales are frozen constants to prevent non-stationary reward scaling.

---

## 3. Two-Level Decision & Optimization Architecture

The overall optimization decomposes into two coupled decision levels:

| Level | Decision Maker | Controlled Variables | Influencing Objectives | Formulation |
|---|---|---|---|---|
| **Level 1** | MARL Agent Policy (MAPPO/IPPO) | Offloading choice $\omega_i \in \{0..K\}$, Transmit power $p_i$ | Delay, Energy, Overhead, Reliability | Discrete action mapping |
| **Level 2** | Edge CPU Solver (CLARABEL) | Capacity fraction allocation $f_{i, k} \in (0, 1]$ | Processing Delay, Offload Deadline | Convex interior-point optimization |

### Critical Derivation:
- Energy ($E_{\text{tx}}$) and Overhead ($O$) depend purely on Level 1 decisions ($p_i$ and offloading state $\omega_i$). They are invariant with respect to Level 2 CPU fractions $f_{i, k}$.
- Level 2 solves:
  $$\min \sum_{j \in \mathcal{U}_k} \left( T_{\text{front}, j} + T_{\text{uplink}, j} + \frac{\rho_j D_j}{C_k f_{j, k}} \right) \quad \text{s.t.} \quad \sum_{j \in \mathcal{U}_k} f_{j, k} \le 1, \quad f_{j, k} > 0$$
  Since $T_{\text{front}}$ and $T_{\text{uplink}}$ are constant with respect to $f$, this convex problem is strictly convex with guaranteed global optimality solved numerically by CLARABEL.
