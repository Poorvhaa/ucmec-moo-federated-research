"""
End-to-End Integration Test: API experiment creation, durable worker execution, and artifact verification.
"""
import time
import pytest
from fastapi.testclient import TestClient
from backend.app import app


def test_e2e_api_and_worker_execution():
    client = TestClient(app)
    
    # 1. Submit experiment through API
    payload = {
        "name": "e2e_test_run",
        "algorithm_name": "mappo",
        "preset_name": "balanced",
        "num_agents": 10,
        "num_env_steps": 200,
        "episode_length": 200,
        "seed": 101,
        "use_corrected_cpu_alloc": True,
        "solver_name": "CLARABEL",
        "channel_mode": "legacy",
    }
    create_res = client.post("/api/experiments", json=payload)
    assert create_res.status_code == 200
    exp_id = create_res.json()["id"]

    # 2. Wait for worker process to finish the 200 steps
    start_time = time.time()
    final_status = None
    while time.time() - start_time < 150:
        get_res = client.get(f"/api/experiments/{exp_id}")
        assert get_res.status_code == 200
        st = get_res.json()
        if st["status"] in ["completed", "failed"]:
            final_status = st
            break
        time.sleep(2)

    assert final_status is not None, "Timed out waiting for experiment execution"
    assert final_status["status"] == "completed", f"Experiment failed: {final_status.get('error_message')}"
    assert final_status["progress_percent"] == 100.0

    # 3. Verify metrics endpoint
    metrics_res = client.get(f"/api/experiments/{exp_id}/metrics")
    assert metrics_res.status_code == 200
    metrics_data = metrics_res.json()
    assert metrics_data["total_records"] > 0
    latest_metric = metrics_data["metrics"][-1]
    assert latest_metric["delay_ms"] > 0.0
    assert latest_metric["energy_mj"] > 0.0
    assert latest_metric["task_success_rate"] > 0.0

    # 4. Verify artifacts endpoint
    art_res = client.get(f"/api/experiments/{exp_id}/artifacts")
    assert art_res.status_code == 200
    art_data = art_res.json()
    assert len(art_data["artifacts"]) > 0


if __name__ == "__main__":
    test_e2e_api_and_worker_execution()
    print("End-to-end integration test PASSED successfully!")
