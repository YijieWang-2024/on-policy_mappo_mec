"""Self-contained scenario loader for the MEC v2 environment.

Single source of truth: docs/mec_env_port_spec.md (v2). Mirrors the origin
repo's ``mfmec.config`` for unchanged pieces (unit unwrapping, access-side
derived constants) and adds the v2-specific parts: grid-generated UAV
deployment, mmWave backhaul block, and formula-derived cost weights (R2 -
omegas must never be hardcoded in the YAML).
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
DEFAULT_SCENARIO = "v6_hap_loadbearing"

_META_KEYS = {"unit", "source", "rationale", "generator", "margin_frac"}


def load_scenario(name_or_path: str | Path = DEFAULT_SCENARIO,
                  fleet_size_k: int | None = None) -> dict[str, Any]:
    """Load a scenario YAML into a plain config dict ready for the env.

    fleet_size_k overrides the YAML's K (for fleet sweeps 8/16/24/32); every
    K-dependent derived quantity (per-UAV access bandwidth, UAV deployment,
    omega_queue, omega_safety) is recomputed accordingly.
    """
    raw = _read_yaml(_resolve(name_or_path))
    cfg = unwrap_values(raw)
    reference_fleet_size = int(cfg["env"]["fleet_size_k"])
    if fleet_size_k is not None:
        cfg["env"]["fleet_size_k"] = int(fleet_size_k)
    _apply_bandwidth_derivation(cfg)
    _generate_uav_deployment(cfg)
    _broadcast_initial_queues(cfg)
    cfg["normalization"] = _derive_normalization(
        cfg, fleet_size_reference=reference_fleet_size
    )
    cfg["cost"]["weights"] = derive_cost_weights(cfg)
    cfg["derived"] = derive_constants(cfg)
    validate_scenario(cfg)
    return cfg


def unwrap_values(obj: Any) -> Any:
    """Strip {value, unit, source, rationale} wrappers, keep generator specs."""
    if isinstance(obj, dict):
        if "value" in obj and (set(obj) - {"value"}) <= _META_KEYS:
            return unwrap_values(obj["value"])
        if "generator" in obj:
            return {k: unwrap_values(v) for k, v in obj.items()}
        return {k: unwrap_values(v) for k, v in obj.items() if k not in _META_KEYS}
    if isinstance(obj, list):
        return [unwrap_values(v) for v in obj]
    return obj


def derive_cost_weights(cfg: dict[str, Any]) -> dict[str, float]:
    """R2: all omegas formula-derived; K-dependent ones recomputed per K."""
    lam = cfg["cost"]["lambdas"]
    ref = cfg["cost"]["references"]
    k = int(cfg["env"]["fleet_size_k"])
    queue_ref_bits = k * float(cfg["env"]["uav"]["queue_max_bits"]) + float(
        cfg["env"]["hap"]["queue_max_bits"])
    d_ref = float(ref["loss_ref_bits_per_slot"])
    e_ref = float(ref["energy_ref_j_per_slot"])
    return {
        "omega_queue_per_bit": float(lam["queue"]) / queue_ref_bits,
        "omega_src_per_bit": float(lam["src"]) / d_ref,
        "omega_ovf_per_bit": float(lam["ovf"]) / d_ref,
        "omega_energy_per_j": float(lam["energy"]) / e_ref,
        "omega_safety": float(lam["safety"]) / max(1.0, k * (k - 1) / 2.0),
        "queue_ref_bits": queue_ref_bits,
    }


def derive_constants(cfg: dict[str, Any]) -> dict[str, float]:
    """Access-side constants mirror the origin repo (1:1 parity, R7).

    Backhaul is v2 mmWave: precompute the dB-domain link-budget constant so the
    env's per-step work is a single distance->SNR evaluation.
    """
    delta = float(cfg["base"]["slot_length_s"])
    access = cfg["communication"]["access"]
    noise = cfg["communication"]["noise"]
    mmw = cfg["communication"]["backhaul"]["mmwave"]
    cycles_per_bit = float(cfg["compute"]["cycles_per_bit"])
    env = cfg["env"]

    thermal_w_per_hz = 10.0 ** ((float(noise["thermal_noise_psd_dbm_per_hz"]) - 30.0) / 10.0)
    noise_eff_w_per_hz = thermal_w_per_hz * 10.0 ** (float(noise["receiver_noise_figure_db"]) / 10.0)
    user_psd = float(access["user_target_psd_w_per_hz"])
    g_th = noise_eff_w_per_hz * float(access["snr_threshold_linear"]) / user_psd

    w_beam = float(mmw["beam_bandwidth_hz"])
    bh_alpha = float(mmw.get("pathloss_exponent", 2.0))
    bh_kappa = float(mmw.get("kappa_o2_db_per_km", 0.0))
    bh_cutoff = bool(mmw.get("use_hard_cutoff", True))
    bh_demod = float(mmw.get("demod_snr_min_db", -math.inf))
    # SNR_dB(d) = bh_link_budget_const_db - 20log10(d) - kappa_o2*(d/1000)
    fspl_const_db = 20.0 * math.log10(float(mmw["carrier_frequency_hz"])) + 20.0 * math.log10(
        4.0 * math.pi / 2.99792458e8)
    noise_dbm = -174.0 + 10.0 * math.log10(w_beam) + float(mmw["noise_figure_db"])
    # Keep the implementation/pointing/fade margin explicit instead of hiding
    # it inside the antenna gain. Existing scenarios omit it and retain their
    # original link budget through the zero default.
    link_margin_db = float(mmw.get("link_margin_db", 0.0))
    bh_const_db = (
        float(mmw["tx_power_dbm"])
        + float(mmw["antenna_gain_total_db"])
        - link_margin_db
        - fspl_const_db
        - noise_dbm
    )

    return {
        "slot_length_s": delta,
        "thermal_noise_psd_w_per_hz": thermal_w_per_hz,
        "noise_psd_eff_w_per_hz": noise_eff_w_per_hz,
        "access_gain_threshold": g_th,
        "max_per_device_bandwidth_hz": float(access["user_max_power_w"]) / user_psd,
        "access_noise_power_w": noise_eff_w_per_hz * float(access["bandwidth_per_uav_hz"]),
        "uav_compute_capacity_bits": float(env["uav"]["cpu_frequency_hz"]) * delta / cycles_per_bit,
        "hap_compute_capacity_bits": float(env["hap"]["cpu_frequency_hz"]) * delta / cycles_per_bit,
        "bh_link_budget_const_db": bh_const_db,
        "bh_pathloss_exponent": bh_alpha,
        "bh_kappa_o2_db_per_km": bh_kappa,
        "bh_link_margin_db": link_margin_db,
        "bh_demod_snr_min_db": bh_demod,
        "bh_use_hard_cutoff": bh_cutoff,
        "bh_beam_bandwidth_hz": w_beam,
    }


def access_coverage_radius_m(cfg: dict[str, Any]) -> float:
    """Horizontal radius where the access gain crosses g_th (viz/analysis aid)."""
    import numpy as _np
    acc = cfg["communication"]["access"]
    path = acc["pathloss"]
    pl = acc["los_probability"]
    h = float(cfg["env"]["uav"]["altitude_m"])
    g_th = cfg["derived"]["access_gain_threshold"]
    a1, a2 = float(pl["a1"]), float(pl["a2"])
    refl = float(path["reference_loss_linear_at_1m"])
    al_los, al_nl = float(path["alpha_los"]), float(path["alpha_nlos"])
    el, en = float(path["eta_los_linear"]), float(path["eta_nlos_linear"])

    def gain(rho: float) -> float:
        d = math.hypot(rho, h)
        theta = math.degrees(math.atan2(h, max(rho, 1e-9)))
        p_los = 1.0 / (1.0 + a1 * math.exp(-a2 * (theta - a1)))
        return p_los / (refl * d ** al_los * el) + (1.0 - p_los) / (refl * d ** al_nl * en)

    lo, hi = 1.0, max(float(cfg["env"]["region"]["lx_m"]), float(cfg["env"]["region"]["ly_m"]))
    if gain(lo) < g_th:
        return 0.0
    if gain(hi) >= g_th:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gain(mid) >= g_th else (lo, mid)
    return lo


def backhaul_service_radius_m(cfg: dict[str, Any]) -> float:
    """Horizontal radius where beam SNR crosses the demod threshold (analysis aid)."""
    if not cfg["derived"].get("bh_use_hard_cutoff", True):
        return math.inf
    d = cfg["derived"]
    dz = float(cfg["env"]["hap"]["altitude_m"]) - float(cfg["env"]["uav"]["altitude_m"])

    def snr_db(horiz: float) -> float:
        dist = math.hypot(horiz, dz)
        return (d["bh_link_budget_const_db"]
                - 10.0 * float(d.get("bh_pathloss_exponent", 2.0)) * math.log10(dist)
                - d["bh_kappa_o2_db_per_km"] * dist / 1000.0)

    lo, hi = 0.0, 50000.0
    if snr_db(lo) < d["bh_demod_snr_min_db"]:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if snr_db(mid) >= d["bh_demod_snr_min_db"] else (lo, mid)
    return lo


def validate_scenario(cfg: dict[str, Any]) -> None:
    k = int(cfg["env"]["fleet_size_k"])
    uav = cfg["env"]["uav"]
    if len(uav["initial_xy_m"]) != k:
        raise ValueError("uav.initial_xy_m length must equal fleet_size_k")
    if len(uav["initial_queue_bits"]) != k:
        raise ValueError("uav.initial_queue_bits length must equal fleet_size_k")
    if cfg["communication"]["backhaul"]["link_model"] not in {"mmwave_beam", "continuous_mmwave"}:
        raise ValueError("MEC loader supports backhaul.link_model=mmwave_beam or continuous_mmwave")
    proc = cfg["demand"]["process"]
    if proc["model"] != "random_walk_hotspot":
        raise ValueError("v2 loader only supports demand.process.model=random_walk_hotspot")
    frac = proc["initial_center_frac"]
    if not (isinstance(frac, list) and len(frac) == 2 and all(0.0 <= float(f) <= 1.0 for f in frac)):
        raise ValueError("demand.process.initial_center_frac must be two fractions in [0,1]")
    if cfg["env"]["slot_timing"]["service_position"] not in {"pre_move", "mid_move", "post_move"}:
        raise ValueError("unsupported env.slot_timing.service_position")
    states = cfg["demand"]["markov_chain"]["states"]
    rows = cfg["demand"]["markov_chain"]["transition_matrix"]
    if len(rows) != len(states) or any(
            len(r) != len(states) or abs(sum(r) - 1.0) > 1e-10 for r in rows):
        raise ValueError("demand transition matrix must be square and row-stochastic")
    weights = cfg["cost"]["weights"]
    expected = derive_cost_weights(cfg)
    for key, val in expected.items():
        if abs(weights[key] - val) > 1e-15 * max(1.0, abs(val)):
            raise ValueError(f"cost weight {key} is not formula-derived (R2)")


def _generate_uav_deployment(cfg: dict[str, Any]) -> None:
    uav = cfg["env"]["uav"]
    spec = uav["initial_xy_m"]
    if isinstance(spec, list):
        return
    gen = spec.get("generator")
    k = int(cfg["env"]["fleet_size_k"])
    lx = float(cfg["env"]["region"]["lx_m"])
    ly = float(cfg["env"]["region"]["ly_m"])
    g = int(np.ceil(np.sqrt(k)))
    if gen == "grid":
        margin = float(spec.get("margin_frac", 0.2))
        span = 1.0 - 2.0 * margin
        pts: list[list[float]] = []
        for i in range(g):
            for j in range(g):
                if len(pts) < k:
                    fx = margin + (span * i / (g - 1) if g > 1 else span / 2.0)
                    fy = margin + (span * j / (g - 1) if g > 1 else span / 2.0)
                    pts.append([lx * fx, ly * fy])
        uav["initial_xy_m"] = pts
    elif gen == "cluster":
        # Compact staging formation centered at center_frac, side = cluster_frac*region.
        # Mutual spacing = side/(g-1) is kept >> d_min so no initial collision penalty;
        # used to place the swarm AWAY from the demand start so positioning is forced.
        cfrac = float(spec.get("cluster_frac", 0.10))
        cx, cy = (float(v) for v in spec.get("center_frac", [0.75, 0.75]))
        side_x, side_y = cfrac * lx, cfrac * ly
        pts = []
        for i in range(g):
            for j in range(g):
                if len(pts) < k:
                    ox = (i / (g - 1) - 0.5 if g > 1 else 0.0) * side_x
                    oy = (j / (g - 1) - 0.5 if g > 1 else 0.0) * side_y
                    px = min(max(lx * cx + ox, 0.0), lx)
                    py = min(max(ly * cy + oy, 0.0), ly)
                    pts.append([px, py])
        uav["initial_xy_m"] = pts
    else:
        raise ValueError(f"unsupported uav.initial_xy_m generator: {gen}")
    _apply_hub_initial(cfg)


def _apply_hub_initial(cfg: dict[str, Any]) -> None:
    """Hub start = UAV swarm centroid when configured 'swarm_centroid' (the hub and
    fleet deploy together from one staging point); else keep the explicit xy."""
    hap = cfg["env"]["hap"]
    spec = hap.get("initial_xy_m")
    if isinstance(spec, dict) and spec.get("generator") == "swarm_centroid":
        pts = np.asarray(cfg["env"]["uav"]["initial_xy_m"], dtype=float)
        hap["initial_xy_m"] = [float(pts[:, 0].mean()), float(pts[:, 1].mean())]


def _broadcast_initial_queues(cfg: dict[str, Any]) -> None:
    uav = cfg["env"]["uav"]
    q0 = uav["initial_queue_bits"]
    if not isinstance(q0, list):
        uav["initial_queue_bits"] = [float(q0)] * int(cfg["env"]["fleet_size_k"])


def _apply_bandwidth_derivation(cfg: dict[str, Any]) -> None:
    access = cfg["communication"]["access"]
    if access.get("bandwidth_allocation") == "fixed_total" or (
            "total_bandwidth_hz" in access and "bandwidth_per_uav_hz" not in access):
        access["bandwidth_per_uav_hz"] = float(access["total_bandwidth_hz"]) / float(
            cfg["env"]["fleet_size_k"])
        access["bandwidth_allocation"] = "fixed_total"
    bh = cfg["communication"].get("backhaul", {})
    mmw = bh.get("mmwave", {})
    if mmw.get("bandwidth_allocation") == "fixed_total" or (
            "total_bandwidth_hz" in mmw and "beam_bandwidth_hz" not in mmw):
        mmw["beam_bandwidth_hz"] = float(mmw["total_bandwidth_hz"]) / float(
            cfg["env"]["fleet_size_k"])
        mmw["bandwidth_allocation"] = "fixed_total"


def _derive_normalization(
    cfg: dict[str, Any], *, fleet_size_reference: int
) -> dict[str, Any]:
    out = dict(cfg.get("normalization", {}))
    out["position"] = {"divide_by_m": [float(cfg["env"]["region"]["lx_m"]),
                                       float(cfg["env"]["region"]["ly_m"])]}
    out["uav_queue"] = {"divide_by_bits": float(cfg["env"]["uav"]["queue_max_bits"])}
    out["hap_queue"] = {"divide_by_bits": float(cfg["env"]["hap"]["queue_max_bits"])}
    out["resource_context"] = {
        "fleet_size_reference": int(fleet_size_reference)
    }
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    candidate = SCENARIO_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    if p.is_absolute():
        return p
    raise FileNotFoundError(f"scenario not found: {name_or_path} (looked in {SCENARIO_DIR})")
