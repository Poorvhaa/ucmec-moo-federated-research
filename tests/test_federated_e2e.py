"""
End-to-End Federated Integration Test: API submission, worker execution, checkpoint evaluation, and artifacts.
"""
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import torch

from backend.app import app
from evaluate import evaluate_policy_on_scenario


def test_federated_e2e_execution():
    client = TestClient(app)

    # 1. Submit functional smoke federation job: 3 clients, 2 rounds, short rollouts
    payload = {
        "name": "fed_smoke_e2e",
        "preset_name": "balanced",
        "scenario_type": "iid",
        "num_clients": 3,
        "num_rounds": 2,
        "min_clients": 1,
        "participation_rate": 1.0,
        "dropout_prob": 0.0,
        "seed": 42,
    }
    create_res = client.post("/api/federated-experiments", json=payload)
    assert create_res.status_code == 200
    exp_id = create_res.json()["id"]

    # 2. Poll until completed or failed
    start_time = time.time()
    final_status = None
    while time.time() - start_time < 300:
        get_res = client.get(f"/api/federated-experiments/{exp_id}")
        assert get_res.status_code == 200
        st = get_res.json()
        if st["status"] in ["completed", "failed"]:
            final_status = st
            break
        time.sleep(2)

    assert final_status is not None, "Timed out waiting for federated experiment execution"
    assert final_status["status"] == "completed", f"Federated run failed: {final_status.get('error_message')}"
    assert final_status["progress_percent"] == 100.0
    assert final_status["current_round"] == 2

    # 3. Verify rounds endpoint
    rounds_res = client.get(f"/api/federated-experiments/{exp_id}/rounds")
    assert rounds_res.status_code == 200
    r_data = rounds_res.json()
    assert r_data["total_rounds"] == 2
    r2 = r_data["rounds"][-1]
    assert r2["round"] == 2
    assert len(r2["accepted_clients"]) >= 1
    assert r2["round_download_bytes"] > 0
    assert r2["round_upload_bytes"] > 0
    assert r2["mean_delay_ms"] > 0.0
    assert r2["mean_energy_mj"] > 0.0

    # 4. Verify artifacts endpoint
    art_res = client.get(f"/api/federated-experiments/{exp_id}/artifacts")
    assert art_res.status_code == 200
    art_data = art_res.json()
    names = [a["name"] for a in art_data["artifacts"]]
    assert any("checkpoints/global_actor_round_2.pt" in n for n in names)
    assert any("checkpoints/round_2_manifest.json" in n for n in names)

    # 5. Genuine checkpoint loading for held-out evaluation
    ckpt_path = Path(final_status["log_dir"]) / "checkpoints" / "global_actor_round_2.pt"
    assert ckpt_path.exists()

    eval_score = evaluate_policy_on_scenario(
        preset_name="balanced",
        seed=999,
        num_steps=50,
        num_agents=10,
        model_dir=str(ckpt_path),
    )
    assert eval_score["delay_ms"] > 0.0
    assert eval_score["energy_mj"] > 0.0
    assert eval_score["num_steps_completed"] == 50


if __name__ == "__main__":
    test_federated_e2e_execution()
    print("Federated E2E test passed successfully!")
