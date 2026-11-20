# Federated Coordination Protocol and Communication Accounting

## 1. Protocol Message Exchange Specification

The communication protocol operates across synchronized rounds $r = 1, \dots, R$:

```
Coordinator                             Selected Client k
    │                                          │
    │────── Broadcast Global Model ───────────>│ (Global weights theta_global^(r-1), Round r)
    │       (Download bytes tracked)           │
    │                                          │── Local Trajectory Collection
    │                                          │── Local PPO Updates
    │                                          │── Parameter Delta & Sample Count Extraction
    │<───── Submit Model Update ───────────────│ (theta_k^(r), n_k^(r), Metrics, ClientID)
    │       (Upload bytes tracked)             │
    │                                          │
    │── Validate Update (Schema, NaN, Stale)   │
    │── Verify Participation (|A_r| >= K_min)  │
    │── Aggregate (Sample-Weighted FedAvg)     │
    │── Atomic Global Checkpoint Save          │
    │── Held-Out Global Evaluation             │
```

### 1.1 Message Schemas

#### Global Model Broadcast Message (Coordinator $\to$ Client)
```json
{
  "round": 1,
  "global_version": 0,
  "config": {
    "local_epochs": 5,
    "local_batch_size": 32,
    "lr": 0.0005,
    "reset_optimizer_on_sync": true
  },
  "actor_state_dict": "<serialized_floating_tensors>"
}
```

#### Client Update Submission Message (Client $\to$ Coordinator)
```json
{
  "experiment_id": "fed_exp_123",
  "client_id": "client_0",
  "round": 1,
  "base_global_version": 0,
  "valid_samples": 200,
  "runtime_s": 3.42,
  "local_metrics": {
    "reward": -0.24,
    "delay_ms": 18.5,
    "energy_mj": 4.2,
    "overhead_kb": 12.0,
    "task_success_rate": 0.95,
    "policy_loss": 0.012,
    "value_loss": 0.045
  },
  "actor_state_dict": "<serialized_floating_tensors>"
}
```

---

## 2. Communication Accounting Specification

### 2.1 Serialization and Byte Accounting

1. **Global Model Download**:
   - For each selected client receiving the broadcast:
     $$\text{Bytes}_{\text{download}} = \sum_{p \in \text{actor parameters}} \text{size}(p) \times \text{dtype\_bytes}(p) + \text{protocol\_header\_bytes}$$
   - Only actor network weights are transmitted; critics remain local and contribute zero bytes.
2. **Client Model Upload**:
   - For each client successfully uploading its update:
     $$\text{Bytes}_{\text{upload}} = \sum_{p \in \text{actor parameters}} \text{size}(p) \times \text{dtype\_bytes}(p) + \text{metadata\_json\_bytes}$$
3. **Cumulative Accounting**:
   - The coordinator maintains an immutable communication ledger logging:
     - `round`: Communication round index.
     - `client_id`: Target/source client.
     - `direction`: `download` or `upload`.
     - `payload_bytes`: Raw serialized tensor bytes.
     - `protocol_bytes`: Framing and metadata bytes.
     - `success`: Boolean indicating if transmission succeeded or dropped.

### 2.2 Domain Separation of Overheads

- **Fronthaul Wireless Signaling Overhead**: Measured in kilobytes ($\text{kB}$) as part of the physical mmWave transmission simulation per task.
- **Federated Training Communication Overhead**: Measured in megabytes ($\text{MB}$) representing serialized network parameter exchanges between edge site servers and the coordinator.
- **Rule**: Federated model training bytes are strictly tracked in the communication ledger and are **never** added directly to the per-task reward scalarization without an explicit amortization model.

---

## 3. Failure Handling and Dropout Semantics

1. **Configurable Minimum Participation ($K_{\min}$)**:
   - If the number of valid submitted updates $|\mathcal{A}_r| < K_{\min}$, the coordinator marks the round as failed.
   - Behavior: If `on_insufficient_clients == "abort"`, the experiment stops with status `failed`. If `on_insufficient_clients == "retry"`, the round is re-attempted up to `max_round_retries` times.
2. **Simulated Dropout / Stragglers**:
   - Clients can be configured with a dropout probability $p_{\text{drop}} \in [0, 1)$ or simulated network delays.
   - Dropped updates contribute to failed transmission byte tracking but are excluded from the FedAvg parameter aggregation.
3. **Atomic Checkpointing & Warm Resumes**:
   - Global checkpoints are saved atomically: written first to `global_actor_round_{r}.pt.tmp`, then atomically moved to `global_actor_round_{r}.pt`.
   - On worker restart or experiment resume, the coordinator inspects committed round manifests and resumes from round $r_{\text{last\_committed}} + 1$.
