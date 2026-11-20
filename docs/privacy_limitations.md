# Privacy Scope and Security Limitations

## 1. Trajectory Boundary

In this federated multi-agent learning system, raw user interaction data strictly remains within each client's local execution boundary:

- **Retained Locally at Client**:
  - Raw environment observations ($s_t$, $o_t^i$).
  - Individual and joint agent actions ($a_t^i$).
  - Per-user delay, energy, and overhead transition samples ($r_t^i$).
  - Task arrival sizes, densities, and user positions.
  - Local critic parameters and critic value trajectories.
  - Replay and rollout buffers.
- **Transmitted to Coordinator**:
  - Validated floating-point actor model parameter tensors ($\theta_k$).
  - Total valid sample counts ($n_k$).
  - Scalar operational metadata (round ID, client ID, base model version).
  - Aggregate scalar summary metrics (mean delay, mean energy, task success rate).

---

## 2. Formal Privacy Disclaimer

> [!CAUTION]
> **Formal Privacy Disclaimer**:
> **Raw trajectories remain local in this simulation. Model updates may still reveal information. Differential privacy and secure aggregation are not implemented.**
>
> This platform must **not** be described as "privacy-preserving" in any formal cryptographic or information-theoretic sense (such as $(\epsilon, \delta)$-differential privacy). Gradient inversion attacks, membership inference attacks, and model reconstruction techniques can potentially infer properties of training environments from raw neural network weight updates.

---

## 3. Serialization and Deserialization Security

To prevent remote code execution vulnerabilities commonly associated with Python object serialization:
1. **No Insecure Unpickling**:
   - `pickle.loads` and unrestricted `torch.load` on untrusted binary buffers are **strictly prohibited**.
2. **Safe Model Deserialization**:
   - Updates are transferred either as named floating-point NumPy/torch tensor dictionaries or structured safe tensor formats.
   - Every tensor in the incoming payload is checked against a strict schema:
     - Exact key names matching the canonical actor module.
     - Exact dimension tuple matching expected layer shapes.
     - Explicit float32 or float64 dtype.
     - Finite numeric validation (`torch.isfinite(t).all()`).
3. **Validated JSON Metadata**:
   - Non-tensor payload components (client metadata, transition counts, scalar metrics) must conform to strict Pydantic schemas.
