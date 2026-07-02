from __future__ import annotations

import math

from onpolicy.envs.mec.config_loader import load_scenario


def test_default_scenario_is_v6_hap_loadbearing():
    cfg = load_scenario()
    assert cfg["name"] == "v6_hap_loadbearing"
    assert cfg["env"]["fleet_size_k"] == 16
    assert cfg["base"]["episode_horizon_slots"] == 200
    assert cfg["communication"]["backhaul"]["link_model"] == "continuous_mmwave"


def test_v6_derived_constants_and_resource_scaling():
    cfg = load_scenario()
    assert abs(cfg["derived"]["uav_compute_capacity_bits"] - 4.0e6) < 1e-6
    assert abs(cfg["derived"]["hap_compute_capacity_bits"] - 90e6) < 1e-6
    assert math.isinf(cfg["derived"]["bh_demod_snr_min_db"])

    small = load_scenario(fleet_size_k=8)
    assert small["env"]["fleet_size_k"] == 8
    assert len(small["env"]["uav"]["initial_xy_m"]) == 8
    assert small["normalization"]["resource_context"]["fleet_size_reference"] == 16
    assert small["communication"]["access"]["bandwidth_per_uav_hz"] == 40e6 / 8


def test_cost_weights_are_formula_derived():
    cfg = load_scenario()
    w = cfg["cost"]["weights"]
    assert w["omega_ovf_per_bit"] <= w["omega_src_per_bit"]
    assert w["queue_ref_bits"] == (
        16 * cfg["env"]["uav"]["queue_max_bits"]
        + cfg["env"]["hap"]["queue_max_bits"]
    )


def test_continuous_workload_reference_scenario_still_loads():
    cfg = load_scenario("v6_continuous_workload")
    assert cfg["name"] == "v6_continuous_workload"
    assert cfg["env"]["fleet_size_k"] == 16
    assert cfg["demand"]["workload_field"]["total_workload_bits_per_slot"] == 150e6
