from __future__ import annotations

from copy import deepcopy

import numpy as np

from onpolicy.envs.mec.config_loader import load_scenario
from onpolicy.envs.mec.finite_k_env import FiniteKHAPUAVMECEnv
from onpolicy.envs.mec.observation import PHYSICAL_PUBLIC_STATE_DIM


def _actions(k: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    return {
        "hap_velocity_mps": rng.uniform(-30, 30, 2),
        "uav_velocity_mps": rng.uniform(-40, 40, (k, 2)),
        "beta": rng.uniform(0, 1, k),
    }


def _sorted_rows(rows: np.ndarray) -> np.ndarray:
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


def test_v6_random_step_smoke_and_metrics():
    cfg = load_scenario()
    env = FiniteKHAPUAVMECEnv(cfg)
    obs, info = env.reset(seed=123)
    assert obs["normalized"]["uavs"].shape == (16, 3)
    assert obs["normalized"]["hap"].shape == (PHYSICAL_PUBLIC_STATE_DIM,)
    assert info["demand_center_m"].shape == (2,)
    assert info["hotspot_centers_m"].shape == (4, 2)

    obs, reward, terminated, truncated, info = env.step(_actions(16))
    assert terminated is False
    assert truncated is False
    assert np.isfinite(reward)
    assert info["phi_sum_error"] < 1e-6
    assert "access_diagnostics" in info
    assert obs["normalized"]["uavs"].shape == (16, 3)


def test_horizon_truncates_without_true_termination():
    cfg = load_scenario()
    cfg["base"]["episode_horizon_slots"] = 3
    env = FiniteKHAPUAVMECEnv(cfg)
    env.reset(seed=5)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(_actions(16))
        assert terminated is False
    assert truncated is True


def test_random_reset_permutation_preserves_initial_physical_set():
    cfg_perm = load_scenario()
    cfg_fixed = deepcopy(cfg_perm)
    cfg_fixed["env"]["uav"]["initial_deploy"]["random_uav_permutation"] = False

    env_fixed = FiniteKHAPUAVMECEnv(cfg_fixed)
    env_perm = FiniteKHAPUAVMECEnv(cfg_perm)
    env_fixed.reset(seed=1)
    env_perm.reset(seed=1)

    np.testing.assert_allclose(env_perm.state.hap_xy_m, env_fixed.state.hap_xy_m)
    np.testing.assert_allclose(
        _sorted_rows(env_perm.state.uav_xy_m),
        _sorted_rows(env_fixed.state.uav_xy_m),
    )


def test_fleet_size_override_recomputes_action_shape():
    k = 8
    cfg = load_scenario(fleet_size_k=k)
    env = FiniteKHAPUAVMECEnv(cfg)
    obs, _ = env.reset(seed=11)
    assert obs["normalized"]["uavs"].shape == (k, 3)
    _, reward, terminated, truncated, info = env.step(_actions(k))
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["projected_action"]["uav_velocity_mps"].shape == (k, 2)


def test_v7_random_split_hotspots_reset_and_density():
    cfg = load_scenario("v7_random_split_hotspots")
    env = FiniteKHAPUAVMECEnv(cfg)
    obs, info = env.reset(seed=101)
    centers = info["hotspot_centers_m"][:2]
    assert obs["normalized"]["hap"].shape == (PHYSICAL_PUBLIC_STATE_DIM,)
    assert np.all(centers >= np.array([0.15 * env.lx, 0.15 * env.ly]))
    assert np.all(centers <= np.array([0.85 * env.lx, 0.85 * env.ly]))
    assert 1800.0 <= np.linalg.norm(centers[0] - centers[1]) <= 2700.0
    np.testing.assert_allclose(info["hotspot_velocities_mps"], 0.0)
    _, density = env._demand_density()
    np.testing.assert_allclose(np.sum(density) * env.cell_area, 150e6, rtol=1e-12)
    features = obs["normalized"]["hap"][3:].reshape(4, 5)
    np.testing.assert_allclose(features[2:], 0.0)
