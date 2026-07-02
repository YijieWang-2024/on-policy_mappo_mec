"""Canonical observation layouts for the HAP/UAV MEC task."""

from __future__ import annotations

import numpy as np

ROLE_INDEX = 0
OWN_SLICE = slice(1, 4)

UAV_STATE_DIM = 3
PHYSICAL_PUBLIC_STATE_DIM = 7
RESOURCE_CONTEXT_DIM = 6
PUBLIC_STATE_DIM = PHYSICAL_PUBLIC_STATE_DIM + RESOURCE_CONTEXT_DIM
PHYSICAL_PUBLIC_SLICE = slice(4, 4 + PHYSICAL_PUBLIC_STATE_DIM)
RESOURCE_CONTEXT_SLICE = slice(
    PHYSICAL_PUBLIC_SLICE.stop,
    PHYSICAL_PUBLIC_SLICE.stop + RESOURCE_CONTEXT_DIM,
)
PUBLIC_SLICE = slice(4, 4 + PUBLIC_STATE_DIM)
AGENT_OBS_DIM = 1 + UAV_STATE_DIM + PUBLIC_STATE_DIM
LEGACY_AGENT_OBS_DIM = 14


def build_resource_context(cfg: dict) -> np.ndarray:
    """Return dimensionless fleet/resource parameters shared by every agent.

    The context makes cardinality and K-dependent resource scaling explicit to
    fixed-width population policies. The reference fleet size is the scenario's
    native K before any fleet-size override.
    """
    k = int(cfg["env"]["fleet_size_k"])
    norm = cfg["normalization"]["resource_context"]
    k_ref = max(float(norm["fleet_size_reference"]), 1.0)

    access = cfg["communication"]["access"]
    access_per = float(access["bandwidth_per_uav_hz"])
    access_total = float(access.get("total_bandwidth_hz", access_per * k))
    mmwave = cfg["communication"]["backhaul"]["mmwave"]
    backhaul_per = float(mmwave["beam_bandwidth_hz"])
    backhaul_total = float(
        mmwave.get("total_bandwidth_hz", backhaul_per * k)
    )

    uav_compute = (
        k * float(cfg["derived"]["uav_compute_capacity_bits"])
    )
    hap_compute = float(cfg["derived"]["hap_compute_capacity_bits"])
    total_compute = max(uav_compute + hap_compute, 1e-12)
    loss_ref = max(
        float(cfg["cost"]["references"]["loss_ref_bits_per_slot"]),
        1e-12,
    )

    return np.asarray(
        [
            k / k_ref,
            access_per / max(access_total, 1e-12),
            backhaul_per / max(backhaul_total, 1e-12),
            uav_compute / total_compute,
            hap_compute / total_compute,
            total_compute / loss_ref,
        ],
        dtype=np.float32,
    )


def team_state_dim(num_agents: int) -> int:
    """Return the dimension of ``[public, s_1, ..., s_K]``."""
    if num_agents < 2:
        raise ValueError("MEC requires one HAP and at least one UAV")
    return PUBLIC_STATE_DIM + UAV_STATE_DIM * (num_agents - 1)


def build_team_state(agent_obs: np.ndarray) -> np.ndarray:
    """Build one canonical centralized state from grouped local rows.

    ``agent_obs`` may be shaped ``[N, D]`` or ``[..., N, D]``. Agent zero is
    the HAP and agents one through K are UAVs.
    """
    obs = np.asarray(agent_obs)
    if obs.shape[-1] != AGENT_OBS_DIM:
        raise ValueError(
            f"expected MEC local observation dim {AGENT_OBS_DIM}, "
            f"got {obs.shape[-1]}"
        )
    if obs.shape[-2] < 2:
        raise ValueError("MEC grouped observations require K+1 agent rows")

    public = obs[..., 0, PUBLIC_SLICE]
    uavs = obs[..., 1:, OWN_SLICE].reshape(*obs.shape[:-2], -1)
    return np.concatenate([public, uavs], axis=-1)


def repeat_team_state(agent_obs: np.ndarray) -> np.ndarray:
    """Build and repeat the canonical state for every agent buffer row."""
    obs = np.asarray(agent_obs)
    state = build_team_state(obs)
    return np.repeat(state[..., None, :], obs.shape[-2], axis=-2)
