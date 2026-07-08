"""
Compose a full AeroSim sim-config from modular SHIFT-missile config files.

Rationale
---------
The engagement is described by small, reusable JSON files so many scenarios can
be built without duplicating a huge monolithic config:

    master_*.json      high-level run: which scenario + Monte-Carlo settings
      └─ scenario_*.json   world/clock/engagement + which missile & target
           ├─ missile_*.json   interceptor design + GNC parameters (per-FMU)
           └─ target_*.json    threat missile parameters

The **topology** (the FMU graph: which aux topics wire to which inputs) is fixed
and lives here in Python; the modular JSON files only carry **parameters**
(``fmu_initial_vals``), the world/clock, and the engagement geometry.  This
module merges them into the exact ``fmu_models`` structure the orchestrator
expects and writes the generated sim-config next to the other example configs.

Usage:
    python examples/shift_missile/compose_sim_config.py \
        [config/shift_missile/master_intercept.json]
"""

from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(EXAMPLES, "config")
MODULAR_DIR = os.path.join(CONFIG_DIR, "shift_missile")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base, path)


def load_scenario(master_path: str) -> dict:
    """Resolve a master file into a flat dict of all engagement parameters."""
    master = _load(master_path)
    base = os.path.dirname(master_path)
    scenario = _load(_resolve(master["scenario"], base))
    sbase = os.path.dirname(_resolve(master["scenario"], base))
    missile = _load(_resolve(scenario["missile"], sbase))
    target = _load(_resolve(scenario["target"], sbase))

    mc = None
    if master.get("monte_carlo"):
        mc = _load(_resolve(master["monte_carlo"], base))

    return {
        "master": master,
        "scenario": scenario,
        "missile": missile,
        "target": target,
        "monte_carlo": mc,
    }


def _merged_params(bundle: dict) -> dict:
    """Merge missile/target per-FMU initial_vals with engagement + overrides."""
    missile = bundle["missile"]
    target = bundle["target"]
    scenario = bundle["scenario"]

    # Per-FMU parameter dictionaries (deep-copied so overrides don't mutate).
    p: dict = {}
    for fmu_id, vals in missile.get("fmus", {}).items():
        p[fmu_id] = copy.deepcopy(vals)
    p["threat_missile"] = copy.deepcopy(target.get("fmus", {}).get("threat_missile", {}))

    # Engagement geometry overrides (ego initial conditions, target launch geom).
    eng = scenario.get("engagement", {})
    for dotted, value in eng.get("ego", {}).items():
        p.setdefault("aerodynamics_sm", {})[dotted] = value
    for dotted, value in eng.get("target", {}).items():
        p.setdefault("threat_missile", {})[dotted] = value

    # World origin altitude propagates to every FMU that needs it.
    origin_alt = scenario.get("world", {}).get("origin", {}).get("altitude", 0.0)
    for fmu_id in ("aerodynamics_sm", "baro_sensor", "ego_nav_ekf", "threat_missile"):
        p.setdefault(fmu_id, {})["world_origin_altitude"] = origin_alt

    # Free-form scenario overrides, e.g. {"guidance.guidance_law": "mpc"}.
    for dotted, value in scenario.get("overrides", {}).items():
        fmu_id, key = dotted.split(".", 1)
        p.setdefault(fmu_id, {})[key] = value
    return p


# ── FMU graph (topology) ─────────────────────────────────────────────────────
def _fmu_models(p: dict) -> list:
    """Build the fmu_models list, injecting per-FMU initial_vals from ``p``."""

    def iv(fmu_id):
        return p.get(fmu_id, {})

    A1 = "aerosim.actor1"
    A2 = "aerosim.actor2"
    return [
        {
            "id": "threat_missile",
            "fmu_model_path": "fmu/threat_missile_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [
                {"msg_type": "aerosim::types::VehicleState", "topic": "aerosim.actor2.vehicle_state"}
            ],
            "fmu_aux_input_mapping": {},
            "fmu_aux_output_mapping": {
                f"{A2}.threat.aux_out": {
                    "threat_pos_n": "threat_pos_n", "threat_pos_e": "threat_pos_e",
                    "threat_pos_d": "threat_pos_d", "threat_vel_n": "threat_vel_n",
                    "threat_vel_e": "threat_vel_e", "threat_vel_d": "threat_vel_d",
                    "threat_speed_mps": "threat_speed_mps", "threat_altitude_m": "threat_altitude_m",
                }
            },
            "fmu_initial_vals": iv("threat_missile"),
        },
        {
            "id": "aerodynamics_sm",
            "fmu_model_path": "fmu/aerodynamics_sm_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [
                {"msg_type": "aerosim::types::VehicleState", "topic": "aerosim.actor1.vehicle_state"}
            ],
            "fmu_aux_input_mapping": {
                f"{A1}.servo.aux_out": {
                    "elevator_rad": "elevator_rad", "aileron_rad": "aileron_rad",
                    "rudder_rad": "rudder_rad",
                },
                f"{A1}.propulsion.aux_out": {"thrust_n": "thrust_n"},
                f"{A1}.structures.aux_out": {
                    "mass_kg": "mass_kg", "Ixx": "Ixx", "Iyy": "Iyy", "Izz": "Izz"
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.aero.aux_out": {
                    "true_fx_n": "true_fx_n", "true_fy_n": "true_fy_n", "true_fz_n": "true_fz_n",
                    "true_mx_nm": "true_mx_nm", "true_my_nm": "true_my_nm", "true_mz_nm": "true_mz_nm",
                    "sm_fx_n": "sm_fx_n", "sm_fy_n": "sm_fy_n", "sm_fz_n": "sm_fz_n",
                    "sm_mx_nm": "sm_mx_nm", "sm_my_nm": "sm_my_nm", "sm_mz_nm": "sm_mz_nm",
                    "sm_valid_fx": "sm_valid_fx", "sm_valid_fy": "sm_valid_fy", "sm_valid_fz": "sm_valid_fz",
                    "sm_valid_mx": "sm_valid_mx", "sm_valid_my": "sm_valid_my", "sm_valid_mz": "sm_valid_mz",
                    "sm_sigma_fx": "sm_sigma_fx", "sm_sigma_fy": "sm_sigma_fy", "sm_sigma_fz": "sm_sigma_fz",
                    "sm_sigma_mx": "sm_sigma_mx", "sm_sigma_my": "sm_sigma_my", "sm_sigma_mz": "sm_sigma_mz",
                    "alpha_deg": "alpha_deg", "beta_deg": "beta_deg",
                    "mach_number": "mach_number", "dynamic_pressure_pa": "dynamic_pressure_pa",
                    "air_density_kgm3": "air_density_kgm3", "air_pressure_pa": "air_pressure_pa",
                    "speed_of_sound_mps": "speed_of_sound_mps", "altitude_msl_m": "altitude_msl_m",
                    "ego_pos_n": "ego_pos_n", "ego_pos_e": "ego_pos_e", "ego_pos_d": "ego_pos_d",
                    "ego_vel_n": "ego_vel_n", "ego_vel_e": "ego_vel_e", "ego_vel_d": "ego_vel_d",
                    "ego_qw": "ego_qw", "ego_qx": "ego_qx", "ego_qy": "ego_qy", "ego_qz": "ego_qz",
                    "airspeed_mps": "airspeed_mps", "model_source": "model_source",
                }
            },
            "fmu_initial_vals": iv("aerodynamics_sm"),
        },
        {
            "id": "servo_sm",
            "fmu_model_path": "fmu/servo_sm_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.autopilot.aux_out": {
                    "elevator_cmd_rad": "elevator_cmd_rad", "aileron_cmd_rad": "aileron_cmd_rad",
                    "rudder_cmd_rad": "rudder_cmd_rad", "throttle_cmd": "throttle_cmd",
                }
            },
            "fmu_aux_output_mapping": {
                f"{A1}.servo.aux_out": {
                    "elevator_rad": "elevator_rad", "aileron_rad": "aileron_rad",
                    "rudder_rad": "rudder_rad", "throttle": "throttle",
                }
            },
            "fmu_initial_vals": iv("servo_sm"),
        },
        {
            "id": "propulsion_sm",
            "fmu_model_path": "fmu/propulsion_sm_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.servo.aux_out": {"throttle": "throttle"},
                f"{A1}.aero.aux_out": {"air_pressure_pa": "ambient_pressure_pa"},
            },
            "fmu_aux_output_mapping": {
                f"{A1}.propulsion.aux_out": {
                    "thrust_n": "thrust_n", "thrust_sigma_n": "thrust_sigma_n",
                    "propellant_fraction": "propellant_fraction", "mass_flow_kg_s": "mass_flow_kg_s",
                    "chamber_pressure_pa": "chamber_pressure_pa", "isp_s": "isp_s",
                    "total_impulse_ns": "total_impulse_ns", "burn_time_s": "burn_time_s",
                    "thrust_coeff": "thrust_coeff", "motor_burning": "motor_burning",
                }
            },
            "fmu_initial_vals": iv("propulsion_sm"),
        },
        {
            "id": "structures_sm",
            "fmu_model_path": "fmu/structures_sm_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.propulsion.aux_out": {
                    "propellant_fraction": "propellant_fraction", "thrust_n": "thrust_n"
                },
                f"{A1}.aero.aux_out": {"true_fz_n": "true_fz_n", "true_fy_n": "true_fy_n"},
            },
            "fmu_aux_output_mapping": {
                f"{A1}.structures.aux_out": {
                    "mass_kg": "mass_kg", "Ixx": "Ixx", "Iyy": "Iyy", "Izz": "Izz",
                    "cg_x_m": "cg_x_m", "load_factor_g": "load_factor_g",
                    "max_bending_moment_nm": "max_bending_moment_nm",
                    "max_bending_stress_pa": "max_bending_stress_pa",
                    "axial_stress_pa": "axial_stress_pa", "combined_stress_pa": "combined_stress_pa",
                    "margin_of_safety": "margin_of_safety",
                    "first_bending_freq_hz": "first_bending_freq_hz",
                    "prob_structural_failure": "prob_structural_failure",
                }
            },
            "fmu_initial_vals": iv("structures_sm"),
        },
        {
            "id": "corrector",
            "fmu_model_path": "fmu/corrector_fmu.fmu",
            "component_input_topics": [
                {"msg_type": "aerosim::types::VehicleState", "topic": "aerosim.actor1.vehicle_state"}
            ],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.aero.aux_out": {
                    "sm_fx_n": "sm_fx_n", "sm_fy_n": "sm_fy_n", "sm_fz_n": "sm_fz_n",
                    "sm_mx_nm": "sm_mx_nm", "sm_my_nm": "sm_my_nm", "sm_mz_nm": "sm_mz_nm",
                    "sm_valid_fx": "sm_valid_fx", "sm_valid_fy": "sm_valid_fy", "sm_valid_fz": "sm_valid_fz",
                    "sm_valid_mx": "sm_valid_mx", "sm_valid_my": "sm_valid_my", "sm_valid_mz": "sm_valid_mz",
                    "sm_sigma_fx": "sm_sigma_fx", "sm_sigma_fy": "sm_sigma_fy", "sm_sigma_fz": "sm_sigma_fz",
                    "sm_sigma_mx": "sm_sigma_mx", "sm_sigma_my": "sm_sigma_my", "sm_sigma_mz": "sm_sigma_mz",
                },
                f"{A1}.propulsion.aux_out": {"thrust_n": "thrust_n"},
                f"{A1}.structures.aux_out": {
                    "mass_kg": "mass_kg", "Ixx": "Ixx", "Iyy": "Iyy", "Izz": "Izz"
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.corrector.aux_out": {
                    "fx_corrected_n": "fx_corrected_n", "fy_corrected_n": "fy_corrected_n",
                    "fz_corrected_n": "fz_corrected_n", "mx_corrected_nm": "mx_corrected_nm",
                    "my_corrected_nm": "my_corrected_nm", "mz_corrected_nm": "mz_corrected_nm",
                    "std_fx_n": "std_fx_n", "std_fy_n": "std_fy_n", "std_fz_n": "std_fz_n",
                    "std_mx_nm": "std_mx_nm", "std_my_nm": "std_my_nm", "std_mz_nm": "std_mz_nm",
                    "innovation_norm": "innovation_norm", "ensemble_spread": "ensemble_spread",
                    "ground_truth_valid": "ground_truth_valid",
                }
            },
            "fmu_initial_vals": iv("corrector"),
        },
        _sensor("imu_sensor", "imu_fmu", f"{A1}.imu.aux_out", {
            "accel_x_mps2": "accel_x_mps2", "accel_y_mps2": "accel_y_mps2", "accel_z_mps2": "accel_z_mps2",
            "gyro_x_rps": "gyro_x_rps", "gyro_y_rps": "gyro_y_rps", "gyro_z_rps": "gyro_z_rps",
            "measurement_ready": "measurement_ready", "imu_time_s": "imu_time_s",
        }, iv("imu_sensor")),
        _sensor("gnss_sensor", "gnss_fmu", f"{A1}.gnss.aux_out", {
            "pos_n_m": "pos_n_m", "pos_e_m": "pos_e_m", "pos_d_m": "pos_d_m",
            "vel_n_mps": "vel_n_mps", "vel_e_mps": "vel_e_mps", "vel_d_mps": "vel_d_mps",
            "measurement_ready": "measurement_ready", "gnss_time_s": "gnss_time_s",
        }, iv("gnss_sensor")),
        _sensor("baro_sensor", "baro_fmu", f"{A1}.baro.aux_out", {
            "pressure_pa": "pressure_pa", "baro_alt_m": "baro_alt_m",
            "measurement_ready": "measurement_ready", "baro_time_s": "baro_time_s",
        }, iv("baro_sensor")),
        {
            "id": "ir_seeker",
            "fmu_model_path": "fmu/ir_seeker_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.aero.aux_out": {
                    "ego_pos_n": "ego_pos_n", "ego_pos_e": "ego_pos_e", "ego_pos_d": "ego_pos_d",
                    "ego_qw": "ego_qw", "ego_qx": "ego_qx", "ego_qy": "ego_qy", "ego_qz": "ego_qz",
                },
                f"{A2}.threat.aux_out": {
                    "threat_pos_n": "tgt_pos_n", "threat_pos_e": "tgt_pos_e", "threat_pos_d": "tgt_pos_d"
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.ir.aux_out": {
                    "ir_az_rad": "ir_az_rad", "ir_el_rad": "ir_el_rad", "ir_locked": "ir_locked",
                    "measurement_ready": "measurement_ready", "ir_time_s": "ir_time_s",
                }
            },
            "fmu_initial_vals": iv("ir_seeker"),
        },
        {
            "id": "radar_seeker",
            "fmu_model_path": "fmu/semi_active_radar_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.aero.aux_out": {
                    "ego_pos_n": "ego_pos_n", "ego_pos_e": "ego_pos_e", "ego_pos_d": "ego_pos_d",
                    "ego_vel_n": "ego_vel_n", "ego_vel_e": "ego_vel_e", "ego_vel_d": "ego_vel_d",
                    "ego_qw": "ego_qw", "ego_qx": "ego_qx", "ego_qy": "ego_qy", "ego_qz": "ego_qz",
                },
                f"{A2}.threat.aux_out": {
                    "threat_pos_n": "tgt_pos_n", "threat_pos_e": "tgt_pos_e", "threat_pos_d": "tgt_pos_d",
                    "threat_vel_n": "tgt_vel_n", "threat_vel_e": "tgt_vel_e", "threat_vel_d": "tgt_vel_d",
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.radar.aux_out": {
                    "radar_range_m": "radar_range_m", "radar_az_rad": "radar_az_rad",
                    "radar_el_rad": "radar_el_rad", "radar_range_rate_mps": "radar_range_rate_mps",
                    "radar_locked": "radar_locked", "measurement_ready": "measurement_ready",
                    "radar_time_s": "radar_time_s",
                }
            },
            "fmu_initial_vals": iv("radar_seeker"),
        },
        {
            "id": "ego_nav_ekf",
            "fmu_model_path": "fmu/ego_nav_ekf_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.imu.aux_out": {
                    "accel_x_mps2": "accel_x_mps2", "accel_y_mps2": "accel_y_mps2", "accel_z_mps2": "accel_z_mps2",
                    "gyro_x_rps": "gyro_x_rps", "gyro_y_rps": "gyro_y_rps", "gyro_z_rps": "gyro_z_rps",
                    "measurement_ready": "imu_measurement_ready",
                },
                f"{A1}.gnss.aux_out": {
                    "pos_n_m": "pos_n_m", "pos_e_m": "pos_e_m", "pos_d_m": "pos_d_m",
                    "vel_n_mps": "vel_n_mps", "vel_e_mps": "vel_e_mps", "vel_d_mps": "vel_d_mps",
                    "measurement_ready": "gnss_measurement_ready",
                },
                f"{A1}.baro.aux_out": {
                    "baro_alt_m": "baro_alt_m", "measurement_ready": "baro_measurement_ready"
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.egonav.aux_out": {
                    "nav_pos_n": "nav_pos_n", "nav_pos_e": "nav_pos_e", "nav_pos_d": "nav_pos_d",
                    "nav_vel_n": "nav_vel_n", "nav_vel_e": "nav_vel_e", "nav_vel_d": "nav_vel_d",
                    "nav_qw": "nav_qw", "nav_qx": "nav_qx", "nav_qy": "nav_qy", "nav_qz": "nav_qz",
                    "nav_p": "nav_p", "nav_q": "nav_q", "nav_r": "nav_r",
                    "nav_ax": "nav_ax", "nav_ay": "nav_ay", "nav_az": "nav_az", "nav_valid": "nav_valid",
                }
            },
            "fmu_initial_vals": iv("ego_nav_ekf"),
        },
        {
            "id": "target_nav_ekf",
            "fmu_model_path": "fmu/target_nav_ekf_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.egonav.aux_out": {
                    "nav_pos_n": "ego_pos_n", "nav_pos_e": "ego_pos_e", "nav_pos_d": "ego_pos_d",
                    "nav_vel_n": "ego_vel_n", "nav_vel_e": "ego_vel_e", "nav_vel_d": "ego_vel_d",
                },
                f"{A1}.radar.aux_out": {
                    "radar_range_m": "radar_range_m", "radar_az_rad": "radar_az_rad",
                    "radar_el_rad": "radar_el_rad", "radar_range_rate_mps": "radar_range_rate_mps",
                    "measurement_ready": "radar_ready",
                },
                f"{A1}.ir.aux_out": {
                    "ir_az_rad": "ir_az_rad", "ir_el_rad": "ir_el_rad", "measurement_ready": "ir_ready"
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.tgtnav.aux_out": {
                    "tgt_pos_n": "tgt_pos_n", "tgt_pos_e": "tgt_pos_e", "tgt_pos_d": "tgt_pos_d",
                    "tgt_vel_n": "tgt_vel_n", "tgt_vel_e": "tgt_vel_e", "tgt_vel_d": "tgt_vel_d",
                    "tgt_acc_n": "tgt_acc_n", "tgt_acc_e": "tgt_acc_e", "tgt_acc_d": "tgt_acc_d",
                    "tgt_valid": "tgt_valid", "est_range_m": "est_range_m",
                }
            },
            "fmu_initial_vals": iv("target_nav_ekf"),
        },
        {
            "id": "guidance",
            "fmu_model_path": "fmu/guidance_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.egonav.aux_out": {
                    "nav_pos_n": "nav_pos_n", "nav_pos_e": "nav_pos_e", "nav_pos_d": "nav_pos_d",
                    "nav_vel_n": "nav_vel_n", "nav_vel_e": "nav_vel_e", "nav_vel_d": "nav_vel_d",
                    "nav_valid": "nav_valid",
                },
                f"{A1}.tgtnav.aux_out": {
                    "tgt_pos_n": "tgt_pos_n", "tgt_pos_e": "tgt_pos_e", "tgt_pos_d": "tgt_pos_d",
                    "tgt_vel_n": "tgt_vel_n", "tgt_vel_e": "tgt_vel_e", "tgt_vel_d": "tgt_vel_d",
                    "tgt_acc_n": "tgt_acc_n", "tgt_acc_e": "tgt_acc_e", "tgt_acc_d": "tgt_acc_d",
                    "tgt_valid": "tgt_valid",
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.guidance.aux_out": {
                    "a_cmd_n": "a_cmd_n", "a_cmd_e": "a_cmd_e", "a_cmd_d": "a_cmd_d",
                    "range_m": "range_m", "closing_speed_mps": "closing_speed_mps",
                    "t_go_s": "t_go_s", "los_rate_rps": "los_rate_rps", "zem_m": "zem_m",
                    "pip_n": "pip_n", "pip_e": "pip_e", "pip_d": "pip_d",
                    "guidance_phase": "guidance_phase", "guidance_active": "guidance_active",
                }
            },
            "fmu_initial_vals": iv("guidance"),
        },
        {
            "id": "autopilot",
            "fmu_model_path": "fmu/autopilot_fmu.fmu",
            "component_input_topics": [],
            "component_output_topics": [],
            "fmu_aux_input_mapping": {
                f"{A1}.guidance.aux_out": {
                    "a_cmd_n": "a_cmd_n", "a_cmd_e": "a_cmd_e", "a_cmd_d": "a_cmd_d",
                    "guidance_active": "guidance_active",
                },
                f"{A1}.egonav.aux_out": {
                    "nav_qw": "nav_qw", "nav_qx": "nav_qx", "nav_qy": "nav_qy", "nav_qz": "nav_qz",
                    "nav_p": "nav_p", "nav_q": "nav_q", "nav_r": "nav_r",
                    "nav_vel_n": "nav_vel_n", "nav_vel_e": "nav_vel_e", "nav_vel_d": "nav_vel_d",
                    "nav_valid": "nav_valid",
                },
                f"{A1}.aero.aux_out": {
                    "dynamic_pressure_pa": "qbar_pa", "airspeed_mps": "airspeed_mps"
                },
                f"{A1}.structures.aux_out": {"mass_kg": "mass_kg", "Iyy": "Iyy", "Izz": "Izz"},
                f"{A1}.imu.aux_out": {
                    "gyro_x_rps": "gyro_p", "gyro_y_rps": "gyro_q", "gyro_z_rps": "gyro_r",
                    "accel_x_mps2": "accel_x", "accel_y_mps2": "accel_y", "accel_z_mps2": "accel_z",
                },
            },
            "fmu_aux_output_mapping": {
                f"{A1}.autopilot.aux_out": {
                    "elevator_cmd_rad": "elevator_cmd_rad", "aileron_cmd_rad": "aileron_cmd_rad",
                    "rudder_cmd_rad": "rudder_cmd_rad", "throttle_cmd": "throttle_cmd",
                    "az_cmd_mps2": "az_cmd_mps2", "ay_cmd_mps2": "ay_cmd_mps2",
                    "az_ach_mps2": "az_ach_mps2", "ay_ach_mps2": "ay_ach_mps2",
                }
            },
            "fmu_initial_vals": iv("autopilot"),
        },
    ]


def _sensor(fmu_id, fmu_file, topic, out_map, initial_vals):
    return {
        "id": fmu_id,
        "fmu_model_path": f"fmu/{fmu_file}.fmu",
        "component_input_topics": [
            {"msg_type": "aerosim::types::VehicleState", "topic": "aerosim.actor1.vehicle_state"}
        ],
        "component_output_topics": [],
        "fmu_aux_input_mapping": {},
        "fmu_aux_output_mapping": {topic: out_map},
        "fmu_initial_vals": initial_vals,
    }


def compose(master_path: str) -> dict:
    bundle = load_scenario(master_path)
    scenario = bundle["scenario"]
    p = _merged_params(bundle)

    clock = scenario.get("clock", {"step_size_ms": 10, "pace_1x_scale": True})
    world = scenario.get("world", {})
    origin = world.get("origin", {"latitude": 35.0, "longitude": -117.0, "altitude": 0.0})
    weather = world.get("weather", {"preset": "ClearSky"})

    def _actor(name, desc):
        return {
            "actor_name": name,
            "actor_asset": "vehicles/generic_airplane/generic_airplane",
            "parent": "", "description": desc,
            "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                          "scale": [1.0, 1.0, 1.0]},
            "state": {"msg_type": "aerosim::types::VehicleState",
                      "topic": f"aerosim.{name}.vehicle_state"},
            "effectors": [],
        }

    return {
        "description": bundle["master"].get("description", "SHIFT missile interception (composed)"),
        "clock": clock,
        "orchestrator": {
            "sync_topics": [
                {"topic": "aerosim.actor1.vehicle_state", "interval_ms": clock.get("step_size_ms", 10)},
                {"topic": "aerosim.actor2.vehicle_state", "interval_ms": clock.get("step_size_ms", 10)},
            ]
        },
        "world": {
            "update_interval_ms": 20,
            "origin": origin,
            "weather": weather,
            "actors": [
                _actor("actor1", "Ego SHIFT interceptor"),
                _actor("actor2", "Threat missile (interception target)"),
            ],
            "sensors": [],
        },
        "renderers": [],
        "fmu_models": _fmu_models(p),
    }


def main(argv=None) -> str:
    argv = argv if argv is not None else sys.argv[1:]
    master = argv[0] if argv else os.path.join(MODULAR_DIR, "master_intercept.json")
    master = _resolve(master, os.getcwd())
    bundle = load_scenario(master)
    out_name = bundle["master"].get("output_sim_config",
                                    "sim_config_shift_missile_intercept.generated.json")
    out_path = os.path.join(CONFIG_DIR, out_name)
    cfg = compose(master)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=4)
    print(f"Composed sim-config -> {out_path}")
    return out_path


if __name__ == "__main__":
    main()
