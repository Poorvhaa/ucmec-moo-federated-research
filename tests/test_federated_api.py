"""
Unit and Integration tests for Federated Learning FastAPI Endpoints and Validation.
"""
import pytest
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_federation_capabilities_endpoint(client):
    res = client.get("/api/federation/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert "supported_scenarios" in data
    assert "iid" in data["supported_scenarios"]
    assert "task_heterogeneity" in data["supported_scenarios"]
    assert "privacy_disclaimer" in data
    assert "Raw trajectories remain local" in data["privacy_disclaimer"]
    assert data["max_clients"] == 20


def test_create_federated_experiment_validation(client):
    # Valid creation payload
    payload = {
        "name": "fed_valid_test",
        "preset_name": "balanced",
        "scenario_type": "iid",
        "num_clients": 3,
        "num_rounds": 2,
        "min_clients": 1,
        "participation_rate": 1.0,
        "dropout_prob": 0.0,
        "seed": 42,
    }
    res = client.post("/api/federated-experiments", json=payload)
    assert res.status_code == 200
    exp = res.json()
    exp_id = exp["id"]
    assert exp["status"] == "queued"
    assert exp["num_clients"] == 3
    assert exp["num_rounds"] == 2

    # Verify detail endpoint
    get_res = client.get(f"/api/federated-experiments/{exp_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == exp["name"]

    # Verify clients endpoint
    clients_res = client.get(f"/api/federated-experiments/{exp_id}/clients")
    assert clients_res.status_code == 200
    c_data = clients_res.json()
    assert c_data["total_clients"] == 3
    # Clean up so the background queue is not clogged
    client.post(f"/api/federated-experiments/{exp_id}/cancel")

    # Invalid name with shell injection characters
    bad_payload = {**payload, "name": "bad;name/rm -rf"}
    bad_res = client.post("/api/federated-experiments", json=bad_payload)
    assert bad_res.status_code == 422

    # Invalid client count (> 20)
    bad_clients = {**payload, "num_clients": 50}
    bad_c_res = client.post("/api/federated-experiments", json=bad_clients)
    assert bad_c_res.status_code == 422


def test_federated_artifact_path_traversal_safety(client):
    # Create experiment
    payload = {
        "name": "fed_traversal_test",
        "preset_name": "balanced",
        "scenario_type": "iid",
        "num_clients": 2,
        "num_rounds": 1,
        "seed": 99,
    }
    res = client.post("/api/federated-experiments", json=payload)
    exp_id = res.json()["id"]

    # Attempt path traversal
    res_escape1 = client.get(f"/api/federated-experiments/{exp_id}/artifacts/../../../../windows/system32/cmd.exe")
    assert res_escape1.status_code in [404, 400]

    res_escape2 = client.get(f"/api/federated-experiments/{exp_id}/artifacts/..%2f..%2fconfig.py")
    assert res_escape2.status_code in [404, 400]
    # Clean up so the background queue is not clogged
    client.post(f"/api/federated-experiments/{exp_id}/cancel")


def test_federated_cancel_and_resume_endpoints(client):
    payload = {
        "name": "fed_cancel_test",
        "preset_name": "balanced",
        "scenario_type": "iid",
        "num_clients": 2,
        "num_rounds": 1,
        "seed": 100,
    }
    create_res = client.post("/api/federated-experiments", json=payload)
    exp_id = create_res.json()["id"]

    # Cancel
    cancel_res = client.post(f"/api/federated-experiments/{exp_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "cancelled"

    # Resume
    resume_res = client.post(f"/api/federated-experiments/{exp_id}/resume")
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "queued"
    # Clean up after testing resume
    client.post(f"/api/federated-experiments/{exp_id}/cancel")
