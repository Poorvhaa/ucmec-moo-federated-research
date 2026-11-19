"""
Tests for Environment Reset, Episode Boundaries, Truncation vs Termination, and Seeded Reproducibility.
"""
import pytest
import numpy as np
from envs.env_discrete import DiscreteActionEnv


def test_reset_clears_state():
    env = DiscreteActionEnv(max_steps=200)
    obs = env.reset()
    assert env.env.step_num == 0
    assert np.all(env.env.omega_last == 0)
    assert np.all(env.env.p_last == 0)
    assert np.all(env.env.delay_last == 0)

    # Take 5 steps
    action = np.zeros((10, 10))
    action[:, 1] = 1.0 # Offload action
    for _ in range(5):
        obs, rews, dones, infos = env.step(action)
    
    assert env.env.step_num == 5
    assert np.any(env.env.omega_last != 0)

    # Reset again
    obs_after = env.reset()
    assert env.env.step_num == 0
    assert np.all(env.env.omega_last == 0)
    assert np.all(env.env.p_last == 0)
    assert np.all(env.env.delay_last == 0)


def test_seeded_reproducibility():
    """Confirms that two environment instances with identical seed generate identical trajectories."""
    env1 = DiscreteActionEnv(preset_name="balanced", max_steps=50)
    env1.seed(12345)
    obs1 = env1.reset()

    env2 = DiscreteActionEnv(preset_name="balanced", max_steps=50)
    env2.seed(12345)
    obs2 = env2.reset()

    np.testing.assert_allclose(obs1, obs2)

    action = np.zeros((10, 10))
    action[:, 2] = 1.0

    for _ in range(10):
        o1, r1, d1, i1 = env1.step(action)
        o2, r2, d2, i2 = env2.step(action)
        np.testing.assert_allclose(o1, o2)
        np.testing.assert_allclose(r1, r2)
        assert d1.tolist() == d2.tolist()


def test_exact_episode_horizon_and_truncation():
    """Confirms that at step 20, done=True and info flags TimeLimit.truncated=True."""
    env = DiscreteActionEnv(max_steps=20)
    env.reset()
    action = np.zeros((10, 10))
    action[:, 0] = 1.0

    for step in range(19):
        obs, rews, dones, infos = env.step(action)
        assert not np.any(dones)
        assert not infos[0]["TimeLimit.truncated"]

    # 20th step reaches max_steps
    obs, rews, dones, infos = env.step(action)
    assert np.all(dones)
    assert infos[0]["TimeLimit.truncated"] is True
    assert infos[0]["bad_transition"] is True
