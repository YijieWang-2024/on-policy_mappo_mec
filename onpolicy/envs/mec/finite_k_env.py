"""Finite-K hierarchical aerial MEC environment (v2, spec docs/mec_env_port_spec.md).

Port of ``Mean Field Mec/src/mfmec/env/finite_k_env.py`` with the v2 deltas:

- backhaul: 60 GHz mmWave dedicated beam per UAV, link-budget SNR with O2
  absorption and a demod gate (outside the ~2.2 km service circle rate = 0);
- cost: lambda_src / lambda_ovf split (source expiry vs queue overflow);
- demand: random-walk hotspot (fixed start, per-episode random heading,
  per-slot noise, boundary reflection);
- hub: low-altitude mobile compute hub (major agent), Fan-style surrogate
  flight power.

Unchanged physics (access channel, LoS/NLoS, service split, queues, DVFS,
rotary-wing energy, safety) is copied verbatim from the origin for 1:1 parity;
the parity test diffs those methods against the origin file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EnvState:
    hap_xy_m: np.ndarray
    hap_queue_bits: float
    uav_xy_m: np.ndarray
    uav_queue_bits: np.ndarray
    demand_center_m: np.ndarray
    demand_velocity_mps: np.ndarray
    step_index: int


class FiniteKHAPUAVMECEnv:
    """``hap`` in code/config = the low-altitude mobile compute hub (major)."""

    def __init__(self, config: dict[str, Any], *, enforce_horizon: bool = True):
        self.cfg = config
        self.enforce_horizon = bool(enforce_horizon)
        self.rng = np.random.default_rng(int(config["base"]["seed"]))
        self.state: EnvState | None = None

        self.k = int(config["env"]["fleet_size_k"])
        self.delta = float(config["base"]["slot_length_s"])
        self.horizon = int(config["base"]["episode_horizon_slots"])
        self.lx = float(config["env"]["region"]["lx_m"])
        self.ly = float(config["env"]["region"]["ly_m"])
        self.eps = float(config["numerics"]["eps"])

        nx, ny = config["numerics"]["spatial_integral"]["grid_shape"]
        xs = (np.arange(nx, dtype=float) + 0.5) * self.lx / nx
        ys = (np.arange(ny, dtype=float) + 0.5) * self.ly / ny
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        self.grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self.cell_area = self.lx * self.ly / (nx * ny)

    # ------------------------------------------------------------------ API

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        center, velocity = self._initial_demand_motion()
        hub_xy, uav_xy = self._initial_deployment()
        self.state = EnvState(
            hap_xy_m=hub_xy,
            hap_queue_bits=float(self.cfg["env"]["hap"]["initial_queue_bits"]),
            uav_xy_m=uav_xy,
            uav_queue_bits=np.array(self.cfg["env"]["uav"]["initial_queue_bits"], dtype=float),
            demand_center_m=center,
            demand_velocity_mps=velocity,
            step_index=0,
        )
        return self._observation(), {
            "demand_center_m": center.copy(),
            "demand_velocity_mps": velocity.copy(),
        }

    def _initial_deployment(self) -> tuple[np.ndarray, np.ndarray]:
        """Hub + UAV start positions.

        If env.uav.initial_deploy.random_centroid is set, each episode samples the
        swarm centroid (= hub) uniformly in a central frac box, then lays the K UAVs
        on a square grid spanning a fixed side around that centroid. This removes the
        fixed-corner directional bias (the swarm no longer learns a constant offset
        vector). If env.uav.initial_deploy.random_uav_permutation is set, the
        homogeneous UAV rows are randomly assigned to those grid points at reset.
        Otherwise the static yaml positions are used (back-compatible).
        """
        dep = self.cfg["env"]["uav"].get("initial_deploy")
        if not dep or not dep.get("random_centroid"):
            hub = np.array(self.cfg["env"]["hap"]["initial_xy_m"], dtype=float)
            uav = np.array(self.cfg["env"]["uav"]["initial_xy_m"], dtype=float)
            return hub, uav
        (lo_x, hi_x), (lo_y, hi_y) = dep["centroid_frac_range"]
        cx = self.rng.uniform(float(lo_x), float(hi_x)) * self.lx
        cy = self.rng.uniform(float(lo_y), float(hi_y)) * self.ly
        side = float(dep.get("grid_side_m", 1000.0))
        g = int(math.ceil(math.sqrt(self.k)))
        offs = (np.arange(g) / (g - 1) - 0.5) * side if g > 1 else np.array([0.0])
        pts = [[cx + offs[i], cy + offs[j]] for i in range(g) for j in range(g)]
        uav = self._clip_xy(np.array(pts[: self.k], dtype=float))
        if dep.get("random_uav_permutation"):
            uav = uav[self.rng.permutation(self.k)]
        hub = self._clip_xy(np.array([cx, cy], dtype=float))
        return hub, uav

    def step(self, action: dict[str, Any]):
        if self.state is None:
            raise RuntimeError("reset() must be called before step().")
        st = self.state
        pre_hap_queue = float(st.hap_queue_bits)
        pre_uav_queue = st.uav_queue_bits.copy()

        hap_v, uav_v, beta = self._project_action(action)
        next_hap_xy = self._clip_xy(st.hap_xy_m + self.delta * hap_v)
        next_uav_xy = self._clip_xy(st.uav_xy_m + self.delta * uav_v)
        service_hap_xy, service_uav_xy = self._service_positions(
            st.hap_xy_m, st.uav_xy_m, next_hap_xy, next_uav_xy)
        active_density, fresh_density = self._demand_density(st.demand_center_m)
        access_gain = self._access_gain(service_uav_xy)
        phi0, phi = self._service_share(access_gain)
        outside_loss = float(np.sum(phi0 * fresh_density) * self.cell_area)
        if self._uses_continuous_workload():
            demand_i = np.sum(phi * fresh_density[None, :], axis=1) * self.cell_area
            access_rate = self._continuous_access_rate(access_gain, phi, fresh_density, demand_i)
            accepted = np.minimum(demand_i, self.delta * access_rate)
            source_loss_uav = np.maximum(demand_i - accepted, 0.0)
            n_srv = demand_i
        else:
            n_srv = np.sum(phi * active_density[None, :], axis=1) * self.cell_area
            access_rate = self._access_rate(access_gain, n_srv)
            b0 = float(self.cfg["demand"]["packet_size_bits"])
            accepted = np.sum(
                phi * active_density[None, :] * np.minimum(b0, self.delta * access_rate),
                axis=1) * self.cell_area
            source_loss_uav = np.sum(
                phi * active_density[None, :] * np.maximum(b0 - self.delta * access_rate, 0.0),
                axis=1) * self.cell_area
            demand_i = accepted + source_loss_uav
        source_loss = float(outside_loss + np.sum(source_loss_uav))
        access_diag = self._access_diagnostics(
            st.demand_center_m, service_hap_xy, service_uav_xy,
            fresh_density, phi0, phi, access_gain, access_rate,
            demand_i, accepted, source_loss_uav)

        backhaul_rate = self._backhaul_rate(service_hap_xy, service_uav_xy)
        c_u = float(self.cfg["derived"]["uav_compute_capacity_bits"])
        s_u = np.minimum((1.0 - beta) * pre_uav_queue, c_u)
        b_i = np.minimum(beta * pre_uav_queue, self.delta * backhaul_rate)
        residual_uav = np.maximum(pre_uav_queue - s_u - b_i, 0.0)
        uav_unclipped = residual_uav + accepted
        q_u_max = float(self.cfg["env"]["uav"]["queue_max_bits"])
        d_u = np.maximum(uav_unclipped - q_u_max, 0.0)
        next_uav_queue = np.minimum(uav_unclipped, q_u_max)

        c_h = float(self.cfg["derived"]["hap_compute_capacity_bits"])
        s_h = min(pre_hap_queue, c_h)
        h_unclipped = max(pre_hap_queue - s_h, 0.0) + float(np.sum(b_i))
        q_h_max = float(self.cfg["env"]["hap"]["queue_max_bits"])
        d_h = max(h_unclipped - q_h_max, 0.0)
        next_hap_queue = min(h_unclipped, q_h_max)

        uav_energy, uav_energy_parts = self._uav_energy(uav_v, s_u, b_i, backhaul_rate)
        hap_energy, hap_energy_parts = self._hap_energy(hap_v, s_h)
        cost_parts = self._stage_cost(
            pre_uav_queue, pre_hap_queue, source_loss, d_u, d_h,
            uav_energy, hap_energy, next_uav_xy)

        next_center, next_velocity = self._advance_demand(st)
        self.state = EnvState(
            next_hap_xy, next_hap_queue, next_uav_xy, next_uav_queue,
            next_center, next_velocity, st.step_index + 1)

        phi_sum_error = float(np.max(np.abs(phi0 + np.sum(phi, axis=0) - 1.0)))
        info = {
            **cost_parts,
            "queue_bits_pre": {"uav": pre_uav_queue.copy(), "hap": pre_hap_queue},
            "queue_bits_post": {"uav": next_uav_queue.copy(), "hap": next_hap_queue},
            "A_i": accepted,
            "B_i": b_i,
            "S_i_U": s_u,
            "S_H": s_h,
            "D_i_U": d_u,
            "D_H": d_h,
            "U_src": source_loss,
            "source_loss_outside_bits": outside_loss,
            "source_loss_capacity_bits": float(np.sum(source_loss_uav)),
            "source_loss_uav_bits": source_loss_uav,
            "uav_energy_j": uav_energy,
            "hap_energy_j": hap_energy,
            "uav_energy_parts": uav_energy_parts,
            "hap_energy_parts": hap_energy_parts,
            "phi_sum_error": phi_sum_error,
            "service_outside_share": float(np.mean(phi0)),
            "projected_action": {"hap_velocity_mps": hap_v, "uav_velocity_mps": uav_v, "beta": beta},
            "hap_xy_m": st.hap_xy_m.copy(),
            "uav_xy_m": st.uav_xy_m.copy(),
            "service_hap_xy_m": service_hap_xy.copy(),
            "service_uav_xy_m": service_uav_xy.copy(),
            "next_hap_xy_m": next_hap_xy.copy(),
            "next_uav_xy_m": next_uav_xy.copy(),
            "demand_center_m": st.demand_center_m.copy(),
            "next_demand_center_m": next_center.copy(),
            "demand_velocity_mps": st.demand_velocity_mps.copy(),
            "backhaul_rate_bps": backhaul_rate,
            "backhaul_in_range": backhaul_rate > 0.0,
            "access_rate_bps": access_rate,
            "A_dem_i": demand_i,
            "access_diagnostics": access_diag,
            "n_srv": n_srv,
        }
        truncated = self.enforce_horizon and self.state.step_index >= self.horizon
        return self._observation(), -cost_parts["training_cost"], False, truncated, info

    # ------------------------------------------------------- observation

    def _observation(self) -> dict[str, Any]:
        assert self.state is not None
        st = self.state
        demand_features = self._demand_features(st)
        norm = self.cfg["normalization"]
        hap_norm = np.concatenate([
            st.hap_xy_m / np.array(norm["position"]["divide_by_m"], dtype=float),
            np.array([st.hap_queue_bits / float(norm["hap_queue"]["divide_by_bits"])]),
            demand_features,
        ])
        uav_norm = np.concatenate([
            st.uav_xy_m / np.array(norm["position"]["divide_by_m"], dtype=float),
            (st.uav_queue_bits / float(norm["uav_queue"]["divide_by_bits"]))[:, None],
        ], axis=1)
        return {
            "hap": {
                "xy_m": st.hap_xy_m.copy(),
                "queue_bits": float(st.hap_queue_bits),
                "demand_center_m": st.demand_center_m.copy(),
                "demand_velocity_mps": st.demand_velocity_mps.copy(),
                "demand_features": demand_features.copy(),
            },
            "demand": {
                "hotspot_center_m": st.demand_center_m.copy(),
                "hotspot_velocity_mps": st.demand_velocity_mps.copy(),
                "features": demand_features.copy(),
            },
            "uavs": {"xy_m": st.uav_xy_m.copy(), "queue_bits": st.uav_queue_bits.copy()},
            "normalized": {"hap": hap_norm, "uavs": uav_norm},
        }

    def _demand_features(self, state: EnvState) -> np.ndarray:
        pos_norm = state.demand_center_m / np.array(
            self.cfg["normalization"]["position"]["divide_by_m"], dtype=float)
        velocity_scale = max(float(self.cfg["env"]["uav"]["velocity_max_mps"]), self.eps)
        vel_norm = state.demand_velocity_mps / velocity_scale
        return np.concatenate([pos_norm, vel_norm]).astype(float, copy=False)

    # ------------------------------------------------- demand (v2 random walk)

    def _initial_demand_motion(self) -> tuple[np.ndarray, np.ndarray]:
        proc = self.cfg["demand"]["process"]
        rng = proc.get("initial_center_frac_range")
        if rng is not None:
            # per-episode random hotspot centre, sampled in a central box (lo,hi per
            # axis) so it is never near a boundary; policy must observe & respond.
            (lo_x, hi_x), (lo_y, hi_y) = rng
            fx = self.rng.uniform(float(lo_x), float(hi_x))
            fy = self.rng.uniform(float(lo_y), float(hi_y))
            frac = np.array([fx, fy], dtype=float)
        else:
            frac = np.array(proc["initial_center_frac"], dtype=float)
        center = frac * np.array([self.lx, self.ly], dtype=float)
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        speed = float(proc["speed_mps"])
        velocity = np.array([speed * math.cos(theta), speed * math.sin(theta)])
        return self._clip_xy(center), velocity

    def _advance_demand(self, state: EnvState) -> tuple[np.ndarray, np.ndarray]:
        noise_std = float(self.cfg["demand"]["process"]["noise_std_m"])
        drift = state.demand_velocity_mps * self.delta + self.rng.normal(0.0, noise_std, 2)
        next_center = np.array(state.demand_center_m, dtype=float) + drift
        next_velocity = np.array(state.demand_velocity_mps, dtype=float).copy()
        limits = np.array([self.lx, self.ly], dtype=float)
        for axis, upper in enumerate(limits):
            if next_center[axis] < 0.0:
                next_center[axis] = -next_center[axis]
                next_velocity[axis] *= -1.0
            if next_center[axis] > upper:
                next_center[axis] = 2.0 * upper - next_center[axis]
                next_velocity[axis] *= -1.0
        return self._clip_xy(next_center), next_velocity

    def _demand_density(self, center: np.ndarray):
        demand = self.cfg["demand"]
        field = demand.get("workload_field")
        if field is not None and field.get("model") == "normalized_background_gaussian":
            dist2 = np.sum((self.grid_xy - center) ** 2, axis=1)
            sigma = float(field["hotspot_sigma_m"])
            hot = np.exp(-dist2 / (2.0 * sigma**2))
            hot_norm = np.sum(hot) * self.cell_area
            area = self.lx * self.ly
            total = float(field["total_workload_bits_per_slot"])
            zeta = float(field["hotspot_fraction"])
            density = total * ((1.0 - zeta) / area + zeta * hot / max(hot_norm, self.eps))
            return density, density
        rho = float(demand["device_density"]["rho_g_devices_per_m2"])
        ap = demand["activity_probability"]
        dist2 = np.sum((self.grid_xy - center) ** 2, axis=1)
        p = float(ap["base_probability"]) + float(ap["hotspot_peak_increment"]) * np.exp(
            -dist2 / (2.0 * float(ap["hotspot_sigma_m"]) ** 2))
        p = np.clip(p, float(ap["clip_probability"][0]), float(ap["clip_probability"][1]))
        active = rho * p
        return active, float(demand["packet_size_bits"]) * active

    def _uses_continuous_workload(self) -> bool:
        field = self.cfg["demand"].get("workload_field")
        return bool(field and field.get("model") == "normalized_background_gaussian")

    # ------------------------------------------------ actions & kinematics

    def _project_action(self, action: dict[str, Any]):
        hap_v = np.array(action.get("hap_velocity_mps", np.zeros(2)), dtype=float)
        uav_v = np.array(action.get("uav_velocity_mps", np.zeros((self.k, 2))), dtype=float)
        beta = np.array(action.get("beta", np.zeros(self.k)), dtype=float)
        hap_v = _project_l2(hap_v, float(self.cfg["env"]["hap"]["velocity_max_mps"]))
        uav_v = np.vstack([
            _project_l2(v, float(self.cfg["env"]["uav"]["velocity_max_mps"])) for v in uav_v])
        return hap_v, uav_v, np.clip(beta, 0.0, 1.0)

    def _service_positions(self, current_hap_xy, current_uav_xy, next_hap_xy, next_uav_xy):
        timing = str(self.cfg["env"]["slot_timing"]["service_position"])
        if timing == "pre_move":
            return current_hap_xy, current_uav_xy
        if timing == "mid_move":
            return 0.5 * (current_hap_xy + next_hap_xy), 0.5 * (current_uav_xy + next_uav_xy)
        if timing == "post_move":
            return next_hap_xy, next_uav_xy
        raise ValueError(f"unsupported env.slot_timing.service_position: {timing}")

    def _clip_xy(self, xy: np.ndarray):
        arr = np.array(xy, dtype=float)
        arr[..., 0] = np.clip(arr[..., 0], 0.0, self.lx)
        arr[..., 1] = np.clip(arr[..., 1], 0.0, self.ly)
        return arr

    # --------------------------------------------------- access (unchanged)

    def _access_gain(self, uav_xy: np.ndarray):
        access = self.cfg["communication"]["access"]
        pl = access["los_probability"]
        path = access["pathloss"]
        h = float(self.cfg["env"]["uav"]["altitude_m"])
        gains = []
        for xy in uav_xy:
            horizontal = np.linalg.norm(self.grid_xy - xy, axis=1)
            d = np.sqrt(horizontal**2 + h**2)
            theta_deg = np.degrees(np.arctan2(h, horizontal))
            p_los = 1.0 / (1.0 + float(pl["a1"]) * np.exp(-float(pl["a2"]) * (theta_deg - float(pl["a1"]))))
            loss_l = float(path["reference_loss_linear_at_1m"]) * d ** float(path["alpha_los"]) * float(path["eta_los_linear"])
            loss_n = float(path["reference_loss_linear_at_1m"]) * d ** float(path["alpha_nlos"]) * float(path["eta_nlos_linear"])
            gains.append(p_los / loss_l + (1.0 - p_los) / loss_n)
        return np.array(gains)

    def _service_share(self, access_gain: np.ndarray):
        tau = float(self.cfg["communication"]["access"]["smooth_service_temperature_tau"])
        score = np.maximum(access_gain / float(self.cfg["derived"]["access_gain_threshold"]), self.eps) ** (1.0 / tau)
        denom = 1.0 + np.sum(score, axis=0)
        return 1.0 / denom, score / denom[None, :]

    def _access_rate(self, access_gain: np.ndarray, n_srv: np.ndarray):
        access = self.cfg["communication"]["access"]
        bandwidth = float(access["bandwidth_per_uav_hz"])
        b_max = float(self.cfg["derived"]["max_per_device_bandwidth_hz"])
        per_device_bw = np.where(
            n_srv > self.eps, np.minimum(bandwidth / np.maximum(n_srv, self.eps), b_max), 0.0)
        snr = float(access["user_target_psd_w_per_hz"]) * access_gain / float(
            self.cfg["derived"]["noise_psd_eff_w_per_hz"])
        return per_device_bw[:, None] * np.log2(1.0 + snr)

    def _continuous_access_rate(
            self, access_gain: np.ndarray, phi: np.ndarray,
            workload_density: np.ndarray, demand_i: np.ndarray):
        access = self.cfg["communication"]["access"]
        bandwidth = float(access["bandwidth_per_uav_hz"])
        snr = float(access["user_target_psd_w_per_hz"]) * access_gain / float(
            self.cfg["derived"]["noise_psd_eff_w_per_hz"])
        eta = np.log2(1.0 + snr)
        weights = phi * workload_density[None, :]
        numerator = np.sum(weights * eta, axis=1) * self.cell_area
        eta_bar = numerator / np.maximum(demand_i, self.eps)
        return bandwidth * eta_bar

    def _access_diagnostics(
            self, center: np.ndarray, hap_xy: np.ndarray, uav_xy: np.ndarray,
            fresh_density: np.ndarray, phi0: np.ndarray, phi: np.ndarray,
            access_gain: np.ndarray, access_rate: np.ndarray, demand_i: np.ndarray,
            accepted: np.ndarray, source_loss_uav: np.ndarray) -> dict[str, Any]:
        """Scalar diagnostics for source-loss anatomy and spatial service quality."""
        total_workload = float(np.sum(fresh_density) * self.cell_area)
        assigned = phi * fresh_density[None, :]
        outside_density = phi0 * fresh_density
        demand_safe = np.maximum(demand_i, self.eps)
        accepted_density = assigned * (accepted / demand_safe)[:, None]
        capacity_source_density = assigned * (source_loss_uav / demand_safe)[:, None]

        sigma = self._hotspot_sigma_m()
        dist_grid = np.linalg.norm(self.grid_xy - center[None, :], axis=1)
        masks = {
            "hotspot": dist_grid <= 1.5 * sigma,
            "background": dist_grid > 1.5 * sigma,
        }
        regions: dict[str, dict[str, float]] = {}
        for name, mask in masks.items():
            offered = float(np.sum(fresh_density[mask]) * self.cell_area)
            outside = float(np.sum(outside_density[mask]) * self.cell_area)
            cap_src = float(np.sum(capacity_source_density[:, mask]) * self.cell_area)
            acc = float(np.sum(accepted_density[:, mask]) * self.cell_area)
            regions[name] = {
                "offered_bits": offered,
                "accepted_bits": acc,
                "source_bits": outside + cap_src,
                "source_outside_bits": outside,
                "source_capacity_bits": cap_src,
            }

        snr = float(self.cfg["communication"]["access"]["user_target_psd_w_per_hz"]) * access_gain / float(
            self.cfg["derived"]["noise_psd_eff_w_per_hz"])
        eta_grid = np.log2(1.0 + snr)
        eta_i = (np.sum(assigned * eta_grid, axis=1) * self.cell_area
                 / np.maximum(demand_i, self.eps))
        active = demand_i > self.eps
        if np.any(active):
            eta_active = eta_i[active]
            weights = demand_i[active]
            eta_mean = float(np.average(eta_active, weights=weights))
            eta_p05 = _weighted_quantile(eta_active, weights, 0.05)
            eta_p50 = _weighted_quantile(eta_active, weights, 0.50)
            eta_p95 = _weighted_quantile(eta_active, weights, 0.95)
        else:
            eta_mean = eta_p05 = eta_p50 = eta_p95 = 0.0

        eta_numer = float(np.sum(assigned * eta_grid) * self.cell_area)
        assigned_total = float(np.sum(assigned) * self.cell_area)
        eta_served_weighted = eta_numer / max(assigned_total, self.eps)
        eta_all_workload_weighted = eta_numer / max(total_workload, self.eps)

        uav_dist = np.linalg.norm(uav_xy - center[None, :], axis=1)
        hub_to_hotspot = float(np.linalg.norm(hap_xy - center))
        uav_to_hub = np.linalg.norm(uav_xy - hap_xy[None, :], axis=1)
        return {
            "total_workload_bits": total_workload,
            "regions": regions,
            "eta_mean": eta_mean,
            "eta_p05": eta_p05,
            "eta_p50": eta_p50,
            "eta_p95": eta_p95,
            "eta_served_workload_weighted": float(eta_served_weighted),
            "eta_all_workload_weighted": float(eta_all_workload_weighted),
            "hotspot_radius_m": float(1.5 * sigma),
            "n_core_uav": int(np.sum(uav_dist <= sigma)),
            "n_hotspot_uav": int(np.sum(uav_dist <= 1.5 * sigma)),
            "n_background_uav": int(np.sum(uav_dist > 1.5 * sigma)),
            "hub_to_hotspot_m": hub_to_hotspot,
            "mean_uav_to_hub_m": float(np.mean(uav_to_hub)),
            "max_uav_to_hub_m": float(np.max(uav_to_hub)),
        }

    def _hotspot_sigma_m(self) -> float:
        field = self.cfg["demand"].get("workload_field")
        if field is not None and "hotspot_sigma_m" in field:
            return float(field["hotspot_sigma_m"])
        return float(self.cfg["demand"]["activity_probability"]["hotspot_sigma_m"])

    # ---------------------------------------------- backhaul (v2: 60 GHz beam)

    def _backhaul_rate(self, hap_xy: np.ndarray, uav_xy: np.ndarray):
        d = self.cfg["derived"]
        dz = float(self.cfg["env"]["hap"]["altitude_m"]) - float(self.cfg["env"]["uav"]["altitude_m"])
        horizontal = np.linalg.norm(uav_xy - hap_xy[None, :], axis=1)
        dist = np.sqrt(horizontal**2 + dz**2)
        snr_db = (d["bh_link_budget_const_db"]
                  - 10.0 * float(d.get("bh_pathloss_exponent", 2.0))
                  * np.log10(np.maximum(dist, 1.0))
                  - d["bh_kappa_o2_db_per_km"] * dist / 1000.0)
        rate = d["bh_beam_bandwidth_hz"] * np.log2(1.0 + 10.0 ** (snr_db / 10.0))
        if not d.get("bh_use_hard_cutoff", True):
            return rate
        return np.where(snr_db >= d["bh_demod_snr_min_db"], rate, 0.0)

    # ----------------------------------------------------- energy (unchanged)

    def _uav_energy(self, uav_v: np.ndarray, s_u: np.ndarray, b_i: np.ndarray, backhaul_rate: np.ndarray):
        speed = np.linalg.norm(uav_v, axis=1)
        e = self.cfg["energy"]["uav_rotary_wing"]
        p0 = float(e["p0_blade_profile_w"])
        p_ind = float(e["p_induced_w"])
        u_tip = float(e["u_tip_mps"])
        v0 = float(e["v0_hover_mps"])
        d0 = float(e["fuselage_drag_ratio"])
        rho = float(e["air_density_kg_per_m3"])
        solidity = float(e["rotor_solidity"])
        area = float(e["rotor_disc_area_m2"])
        power = p0 * (1.0 + 3.0 * speed**2 / u_tip**2)
        power += p_ind * (np.sqrt(1.0 + speed**4 / (4.0 * v0**4)) - speed**2 / (2.0 * v0**2)) ** 0.5
        power += 0.5 * d0 * rho * solidity * area * speed**3
        fly = self.delta * power
        cmp_e = compute_energy(s_u, float(self.cfg["compute"]["kappa_uav"]),
                               float(self.cfg["compute"]["cycles_per_bit"]), self.delta)
        tx_power_w = 10.0 ** ((float(self.cfg["communication"]["backhaul"]["mmwave"]["tx_power_dbm"]) - 30.0) / 10.0)
        tx = np.where(backhaul_rate > self.eps,
                      tx_power_w * b_i / np.maximum(backhaul_rate, self.eps), 0.0)
        return fly + cmp_e + tx, {"fly": fly, "compute": cmp_e, "tx": tx}

    def _hap_energy(self, hap_v: np.ndarray, s_h: float):
        hap = self.cfg["env"]["hap"]
        speed = float(np.linalg.norm(hap_v))
        static_power = float(hap.get("static_power_w", 0.0))
        eps_speed = float(hap.get("speed_smoothing_eps_mps", 0.0))
        if eps_speed > 0.0:
            speed_factor = (speed**2 + eps_speed**2) ** (1.0 / 3.0)
        else:
            speed_factor = speed ** (2.0 / 3.0) if speed > 0 else 0.0
        fly_power = static_power + (
            float(hap["wind_speed_mps"]) ** 3
            * float(hap["air_density_kg_per_m3"])
            * speed_factor
            * float(hap["drag_coefficient"]))
        fly = self.delta * fly_power
        cmp_e = float(compute_energy(np.array([s_h]), float(self.cfg["compute"]["kappa_hap"]),
                                     float(self.cfg["compute"]["cycles_per_bit"]), self.delta)[0])
        return fly + cmp_e, {"fly": fly, "compute": cmp_e}

    # ------------------------------------------------- cost (v2: src/ovf split)

    def _stage_cost(self, pre_uav_queue, pre_hap_queue, source_loss, d_u, d_h,
                    uav_energy, hap_energy, uav_xy):
        weights = self.cfg["cost"]["weights"]
        raw_queue_bits = float(np.sum(pre_uav_queue)) + float(pre_hap_queue)
        raw_src_bits = float(source_loss)
        raw_ovf_bits = float(np.sum(d_u)) + float(d_h)
        raw_energy_j = float(np.sum(uav_energy)) + float(hap_energy)
        safety_raw, safety_violations, min_distance = self._separation_penalty(uav_xy)

        queue = float(weights["omega_queue_per_bit"]) * raw_queue_bits
        src = float(weights["omega_src_per_bit"]) * raw_src_bits
        ovf = float(weights["omega_ovf_per_bit"]) * raw_ovf_bits
        energy = float(weights["omega_energy_per_j"]) * raw_energy_j
        safety = float(weights["omega_safety"]) * safety_raw
        total = queue + src + ovf + energy + safety
        return {
            "raw_cost": total,
            "training_cost": total,
            "queue_cost_component": queue,
            "src_cost_component": src,
            "ovf_cost_component": ovf,
            "energy_cost_component": energy,
            "safety_cost_component": safety,
            "raw_queue_bits": raw_queue_bits,
            "raw_src_bits": raw_src_bits,
            "raw_ovf_bits": raw_ovf_bits,
            "raw_energy_j": raw_energy_j,
            "raw_safety_penalty": safety_raw,
            "safety_violation_count": safety_violations,
            "min_uav_distance_m": min_distance,
        }

    def _separation_penalty(self, uav_xy: np.ndarray):
        if self.k < 2:
            return 0.0, 0, float("inf")
        min_distance = float(self.cfg["cost"]["safety"]["min_uav_distance_m"])
        penalty = 0.0
        violations = 0
        observed_min = float("inf")
        for i in range(self.k):
            for j in range(i + 1, self.k):
                distance = float(np.linalg.norm(uav_xy[i] - uav_xy[j]))
                observed_min = min(observed_min, distance)
                if min_distance <= 0.0:
                    continue
                if distance < min_distance:
                    violations += 1
                    penalty += ((min_distance - distance) / min_distance) ** 2
        return penalty, violations, observed_min


def compute_energy(bits: np.ndarray, kappa: float, cycles_per_bit: float, delta: float):
    return kappa * cycles_per_bit**3 / delta**2 * bits**3


def _project_l2(vector: np.ndarray, radius: float):
    norm = float(np.linalg.norm(vector))
    if norm <= radius or norm == 0.0:
        return vector.copy()
    return vector * (radius / norm)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or float(np.sum(weights)) <= 0.0:
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[min(int(np.searchsorted(cdf, q, side="left")), values.size - 1)])
