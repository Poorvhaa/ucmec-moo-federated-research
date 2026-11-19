"""
Tests for FastAPI Endpoints, Input Validation, and Artifact Path Traversal Safety.
"""
import pytest
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "active_jobs" in data


def test_api_capabilities(client):
    res = client.get("/api/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert "mappo" in data["available_algorithms"]
    assert "CLARABEL" in data["available_solvers"]
    assert data["max_agents"] == 60


def test_api_presets(client):
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()
    ids = [p["id"] for p in presets]
    assert "legacy" in ids
    assert "balanced" in ids
    assert "energy_priority" in ids
    assert "reliability_priority" in ids


def test_experiment_creation_validation(client):
    # Valid creation
    payload = {
        "name": "valid_test_run",
        "algorithm_name": "mappo",
        "preset_name": "balanced",
        "num_agents": 10,
        "num_env_steps": 200,
        "episode_length": 200,
        "seed": 42
    }
    res = client.post("/api/experiments", json=payload)
    assert res.status_code == 200
    exp = res.json()
    assert exp["status"] == "queued"
    assert exp["preset_name"] == "balanced"
    assert exp["num_agents"] == 10

    # Invalid name (contains forbidden characters)
    bad_payload = {**payload, "name": "bad;name/rm -rf"}
    res_bad = client.post("/api/experiments", json=bad_payload)
    assert res_bad.status_code == 422

    # Invalid agent count (> 60)
    bad_agents = {**payload, "num_agents": 100}
    res_agents = client.post("/api/experiments", json=bad_agents)
    assert res_agents.status_code == 422


def test_artifact_path_traversal_safety(client):
    """Verifies that requests attempting to escape the experiment root are strictly rejected."""
    # Create experiment to get valid ID
    payload = {
        "name": "traversal_test",
        "algorithm_name": "mappo",
        "preset_name": "balanced",
        "num_agents": 10,
        "num_env_steps": 200,
        "episode_length": 200,
        "seed": 99
    }
    res = client.post("/api/experiments", json=payload)
    exp_id = res.json()["id"]

    # Traversal attempts
    res_escape1 = client.get(f"/api/experiments/{exp_id}/artifacts/../../../../windows/system32/cmd.exe")
    assert res_escape1.status_code in [404, 400]

    res_escape2 = client.get(f"/api/experiments/{exp_id}/artifacts/..%2f..%2fconfig.py")
    assert res_escape2.status_code in [404, 400]


def test_comparisons_endpoint(client):
    res = client.get("/api/comparisons")
    assert res.status_code == 200
    data = res.json()
    assert "total_compared" in data
    assert "nondominated_subset" in data
