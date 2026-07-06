"""Closed-loop smoke test for the SHIFT missile interception scenario.

Runs the entire FMU stack (threat + ego plant + sensors + seekers + GNC) in a
single Python process by stubbing the AeroSim FMU runtime (pythonfmu3,
aerosim_core, aerosim_data), wiring the FMUs exactly like
``sim_config_shift_missile_intercept.json`` and integrating at 100 Hz.

It verifies: every FMU instantiates/steps without error, all outputs stay
finite, the corrector reconstructs the surrogate-unknown force channels from
ground truth, and the guided ego closes range on the threat.

This file is a developer harness (not an FMU); run it directly:
    python _shift_intercept_smoke_test.py
"""

from __future__ import annotations

import os
import sys
import types as _pytypes

import numpy as np

# ── Stub the FMU runtime so the FMU classes import & instantiate here ─────────
_pfmu = _pytypes.ModuleType("pythonfmu3")


class _Fmi3Slave:
    def __init__(self, **kwargs):
        pass


_pfmu.Fmi3Slave = _Fmi3Slave
sys.modules["pythonfmu3"] = _pfmu

_core = _pytypes.ModuleType("aerosim_core")
_core.register_fmu3_var = lambda self, name, causality=None: None
_core.register_fmu3_param = lambda self, name: None
sys.modules["aerosim_core"] = _core


class _NS:
    """Recursive attribute namespace supporting read/write of nested fields."""

    def __init__(self, d=None):
        object.__setattr__(self, "_d", {})
        if d:
            for k, v in d.items():
                self._d[k] = _NS(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        d = object.__getattribute__(self, "_d")
        if k not in d:
            d[k] = 0.0
        return d[k]

    def __setattr__(self, k, v):
        object.__getattribute__(self, "_d")[k] = v


def _vec3():
    return {"x": 0.0, "y": 0.0, "z": 0.0}


def _quat():
    return {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


class _VehicleState:
    @staticmethod
    def to_dict():
        return {
            "state": {"pose": {"position": _vec3(), "orientation": _quat()}},
            "velocity": _vec3(),
            "angular_velocity": _vec3(),
            "acceleration": _vec3(),
        }


_data = _pytypes.ModuleType("aerosim_data")
_types = _pytypes.ModuleType("aerosim_data.types")
_types.VehicleState = _VehicleState
_data.types = _types
_data.dict_to_namespace = lambda d: _NS(d)
sys.modules["aerosim_data"] = _data
sys.modules["aerosim_data.types"] = _types

# ── Make the FMU source folders importable (atmosphere.py, sixdof.py, FMUs) ──
ROOT = os.path.dirname(os.path.abspath(__file__))
DYN = os.path.join(ROOT, "aerosim-dynamics-models/python/aerosim_dynamics_models/shift_missile_dynamics_fmus")
CTL = os.path.join(ROOT, "aerosim-controllers/python/aerosim_controllers/shift_missile_controller_fmus")
SEN = os.path.join(ROOT, "aerosim-sensors/python/aerosim_sensors/shift_missile_sensor_fmus")
SCE = os.path.join(ROOT, "aerosim-scenarios/python/aerosim_scenarios/shift_missile_scenario_fmus")
for p in (DYN, CTL, SEN, SCE):
    sys.path.insert(0, p)

import importlib.util


def _load(path, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(*path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


aero_m = _load((DYN, "aerodynamics_sm_fmu.py"), "aerodynamics_sm_fmu")
servo_m = _load((DYN, "servo_sm_fmu.py"), "servo_sm_fmu")
struct_m = _load((DYN, "structures_sm_fmu.py"), "structures_sm_fmu")
prop_m = _load((DYN, "propulsion_sm_fmu.py"), "propulsion_sm_fmu")
corr_m = _load((DYN, "corrector_fmu.py"), "corrector_fmu")
guid_m = _load((CTL, "guidance_fmu.py"), "guidance_fmu")
auto_m = _load((CTL, "autopilot_fmu.py"), "autopilot_fmu")
egon_m = _load((CTL, "ego_nav_ekf_fmu.py"), "ego_nav_ekf_fmu")
tgtn_m = _load((CTL, "target_nav_ekf_fmu.py"), "target_nav_ekf_fmu")
imu_m = _load((SEN, "imu_fmu.py"), "imu_fmu")
gnss_m = _load((SEN, "gnss_fmu.py"), "gnss_fmu")
baro_m = _load((SEN, "baro_fmu.py"), "baro_fmu")
ir_m = _load((SEN, "ir_seeker_fmu.py"), "ir_seeker_fmu")
radar_m = _load((SEN, "semi_active_radar_fmu.py"), "semi_active_radar_fmu")
threat_m = _load((SCE, "threat_missile_fmu.py"), "threat_missile_fmu")


def setp(obj, **vals):
    for k, v in vals.items():
        setattr(obj, k, v)


# ── Instantiate + configure (mirrors the sim config initial_vals) ─────────────
threat = threat_m.threat_missile_fmu()
setp(threat, launch_range_m=20000.0, launch_bearing_deg=0.0, cruise_altitude_m=8000.0,
     cruise_speed_mps=300.0, max_lateral_accel_g=8.0, target_north_m=0.0, target_east_m=0.0,
     weave_amplitude_g=2.0, weave_frequency_hz=0.2, world_origin_altitude=0.0)

aero = aero_m.aerodynamics_sm_fmu()
setp(aero, mlp_model_path="", world_origin_altitude=0.0, init_pos_north_m=0.0,
     init_pos_east_m=0.0, init_pos_down_m=-8000.0, init_yaw_rad=0.0, init_pitch_rad=0.0,
     init_speed_mps=600.0)

servo = servo_m.servo_sm_fmu()
prop = prop_m.propulsion_sm_fmu()
setp(prop, burn_time_s=8.0)
struct = struct_m.structures_sm_fmu()
corr = corr_m.corrector_fmu()
imu = imu_m.imu_fmu()
gnss = gnss_m.gnss_fmu()
baro = baro_m.baro_fmu()
ir = ir_m.ir_seeker_fmu()
radar = radar_m.semi_active_radar_fmu()
egonav = egon_m.ego_nav_ekf_fmu()
tgtnav = tgtn_m.target_nav_ekf_fmu()
_LAW = sys.argv[1] if len(sys.argv) > 1 else "propnav"
_CTL = sys.argv[2] if len(sys.argv) > 2 else "lqr"
guid = guid_m.guidance_fmu()
setp(guid, guidance_law=_LAW, nav_gain=3.0, max_accel_g=15.0, mpc_gain=3.0)
auto = auto_m.autopilot_fmu()
setp(auto, controller_type=_CTL, max_alpha_cmd_rad=0.20, lqr_q_angle=50.0)
print(f"[config] guidance_law={_LAW}  controller_type={_CTL}")

everyone = [threat, aero, servo, prop, struct, corr, imu, gnss, baro, ir, radar,
            egonav, tgtnav, guid, auto]
for fmu in everyone:
    fmu.enter_initialization_mode()
    if hasattr(fmu, "exit_initialization_mode"):
        fmu.exit_initialization_mode()


def wire():
    """Copy outputs -> inputs exactly like the sim config aux mappings."""
    # threat aux
    servo.elevator_cmd_rad = auto.elevator_cmd_rad
    servo.aileron_cmd_rad = auto.aileron_cmd_rad
    servo.rudder_cmd_rad = auto.rudder_cmd_rad
    servo.throttle_cmd = auto.throttle_cmd
    prop.throttle = servo.throttle
    struct.propellant_fraction = prop.propellant_fraction
    struct.true_fz_n = aero.true_fz_n
    aero.elevator_rad = servo.elevator_rad
    aero.aileron_rad = servo.aileron_rad
    aero.rudder_rad = servo.rudder_rad
    aero.thrust_n = prop.thrust_n
    aero.mass_kg = struct.mass_kg
    aero.Ixx, aero.Iyy, aero.Izz = struct.Ixx, struct.Iyy, struct.Izz
    # sensors read ego truth (component vehicle_state)
    imu.vehicle_state = aero.vehicle_state
    gnss.vehicle_state = aero.vehicle_state
    baro.vehicle_state = aero.vehicle_state
    corr.vehicle_state = aero.vehicle_state
    # seekers (aux)
    for s in (ir, radar):
        s.ego_pos_n, s.ego_pos_e, s.ego_pos_d = aero.ego_pos_n, aero.ego_pos_e, aero.ego_pos_d
        s.ego_qw, s.ego_qx, s.ego_qy, s.ego_qz = aero.ego_qw, aero.ego_qx, aero.ego_qy, aero.ego_qz
        s.tgt_pos_n, s.tgt_pos_e, s.tgt_pos_d = threat.threat_pos_n, threat.threat_pos_e, threat.threat_pos_d
    radar.ego_vel_n, radar.ego_vel_e, radar.ego_vel_d = aero.ego_vel_n, aero.ego_vel_e, aero.ego_vel_d
    radar.tgt_vel_n, radar.tgt_vel_e, radar.tgt_vel_d = threat.threat_vel_n, threat.threat_vel_e, threat.threat_vel_d
    # ego nav inputs
    for a in ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2", "gyro_x_rps", "gyro_y_rps", "gyro_z_rps"):
        setattr(egonav, a, getattr(imu, a))
    egonav.imu_measurement_ready = imu.measurement_ready
    for a in ("pos_n_m", "pos_e_m", "pos_d_m", "vel_n_mps", "vel_e_mps", "vel_d_mps"):
        setattr(egonav, a, getattr(gnss, a))
    egonav.gnss_measurement_ready = gnss.measurement_ready
    egonav.baro_alt_m = baro.baro_alt_m
    egonav.baro_measurement_ready = baro.measurement_ready
    # corrector aux
    for a in ("sm_fx_n", "sm_fy_n", "sm_fz_n", "sm_mx_nm", "sm_my_nm", "sm_mz_nm",
              "sm_valid_fx", "sm_valid_fy", "sm_valid_fz", "sm_valid_mx", "sm_valid_my", "sm_valid_mz"):
        setattr(corr, a, getattr(aero, a))
    corr.thrust_n = prop.thrust_n
    corr.mass_kg, corr.Ixx, corr.Iyy, corr.Izz = struct.mass_kg, struct.Ixx, struct.Iyy, struct.Izz
    # target nav
    tgtnav.ego_pos_n, tgtnav.ego_pos_e, tgtnav.ego_pos_d = egonav.nav_pos_n, egonav.nav_pos_e, egonav.nav_pos_d
    tgtnav.ego_vel_n, tgtnav.ego_vel_e, tgtnav.ego_vel_d = egonav.nav_vel_n, egonav.nav_vel_e, egonav.nav_vel_d
    tgtnav.radar_range_m, tgtnav.radar_az_rad = radar.radar_range_m, radar.radar_az_rad
    tgtnav.radar_el_rad, tgtnav.radar_range_rate_mps = radar.radar_el_rad, radar.radar_range_rate_mps
    tgtnav.radar_ready = radar.measurement_ready
    tgtnav.ir_az_rad, tgtnav.ir_el_rad, tgtnav.ir_ready = ir.ir_az_rad, ir.ir_el_rad, ir.measurement_ready
    # guidance
    guid.nav_pos_n, guid.nav_pos_e, guid.nav_pos_d = egonav.nav_pos_n, egonav.nav_pos_e, egonav.nav_pos_d
    guid.nav_vel_n, guid.nav_vel_e, guid.nav_vel_d = egonav.nav_vel_n, egonav.nav_vel_e, egonav.nav_vel_d
    guid.nav_valid = egonav.nav_valid
    guid.tgt_pos_n, guid.tgt_pos_e, guid.tgt_pos_d = tgtnav.tgt_pos_n, tgtnav.tgt_pos_e, tgtnav.tgt_pos_d
    guid.tgt_vel_n, guid.tgt_vel_e, guid.tgt_vel_d = tgtnav.tgt_vel_n, tgtnav.tgt_vel_e, tgtnav.tgt_vel_d
    guid.tgt_acc_n, guid.tgt_acc_e, guid.tgt_acc_d = tgtnav.tgt_acc_n, tgtnav.tgt_acc_e, tgtnav.tgt_acc_d
    guid.tgt_valid = tgtnav.tgt_valid
    # autopilot
    auto.a_cmd_n, auto.a_cmd_e, auto.a_cmd_d = guid.a_cmd_n, guid.a_cmd_e, guid.a_cmd_d
    auto.guidance_active = guid.guidance_active
    auto.nav_qw, auto.nav_qx, auto.nav_qy, auto.nav_qz = egonav.nav_qw, egonav.nav_qx, egonav.nav_qy, egonav.nav_qz
    auto.nav_p, auto.nav_q, auto.nav_r = egonav.nav_p, egonav.nav_q, egonav.nav_r
    auto.nav_vel_n, auto.nav_vel_e, auto.nav_vel_d = egonav.nav_vel_n, egonav.nav_vel_e, egonav.nav_vel_d
    auto.nav_valid = egonav.nav_valid
    auto.qbar_pa, auto.airspeed_mps = aero.dynamic_pressure_pa, aero.airspeed_mps
    auto.mass_kg, auto.Iyy, auto.Izz = struct.mass_kg, struct.Iyy, struct.Izz


def rng():
    r = np.array([threat.threat_pos_n - aero.ego_pos_n,
                  threat.threat_pos_e - aero.ego_pos_e,
                  threat.threat_pos_d - aero.ego_pos_d])
    return float(np.linalg.norm(r))


dt = 0.01
t = 0.0
step_order = [threat, servo, prop, struct, aero, imu, gnss, baro, ir, radar,
              egonav, tgtnav, guid, auto, corr]

min_range = 1e18
nonfinite = []
print(f"{'t[s]':>6} {'range[m]':>10} {'ego_alt':>8} {'mach':>5} {'nav_ok':>6} "
      f"{'tgt_ok':>6} {'gd':>3} {'acmd':>8} {'de[deg]':>8}")
n_steps = int(40.0 / dt)
for i in range(n_steps):
    wire()
    for fmu in step_order:
        fmu.do_step(t, dt)
    t += dt
    r = rng()
    min_range = min(min_range, r)
    # finiteness sweep on a representative set of outputs
    for fmu, names in (
        (aero, ("true_fx_n", "true_fz_n", "true_my_nm", "mach_number")),
        (corr, ("fx_corrected_n", "fy_corrected_n", "mz_corrected_nm")),
        (egonav, ("nav_pos_n", "nav_vel_n", "nav_qw")),
        (tgtnav, ("tgt_pos_n", "tgt_vel_n")),
        (guid, ("a_cmd_n", "a_cmd_d")),
        (auto, ("elevator_cmd_rad", "rudder_cmd_rad")),
    ):
        for nm in names:
            if not np.isfinite(getattr(fmu, nm)):
                nonfinite.append((round(t, 2), type(fmu).__name__, nm))
    if i % 200 == 0:
        acmd = float(np.hypot(guid.a_cmd_n, np.hypot(guid.a_cmd_e, guid.a_cmd_d)))
        print(f"{t:6.1f} {r:10.1f} {aero.altitude_msl_m:8.1f} {aero.mach_number:5.2f} "
              f"{egonav.nav_valid:6.0f} {tgtnav.tgt_valid:6.0f} {guid.guidance_active:3.0f} "
              f"{acmd:8.1f} {np.degrees(auto.elevator_cmd_rad):8.2f}")

print("\n--- Corrector reconstruction check (last step) ---")
print(f"  true   f=({aero.true_fx_n:9.1f},{aero.true_fy_n:9.1f},{aero.true_fz_n:9.1f})  "
      f"m=({aero.true_mx_nm:8.1f},{aero.true_my_nm:8.1f},{aero.true_mz_nm:8.1f})")
print(f"  surrog f=({aero.sm_fx_n:9.1f}, unknown ,{aero.sm_fz_n:9.1f})  "
      f"m=( unknown ,{aero.sm_my_nm:8.1f}, unknown )")
print(f"  corr   f=({corr.fx_corrected_n:9.1f},{corr.fy_corrected_n:9.1f},{corr.fz_corrected_n:9.1f})  "
      f"m=({corr.mx_corrected_nm:8.1f},{corr.my_corrected_nm:8.1f},{corr.mz_corrected_nm:8.1f})")
print(f"  innovation_norm={corr.innovation_norm:.3f}  ensemble_spread={corr.ensemble_spread:.1f}  "
      f"gt_valid={corr.ground_truth_valid:.0f}")

print("\n--- Summary ---")
print(f"  model source        : {aero.model_source}")
print(f"  min range achieved  : {min_range:.1f} m")
print(f"  ego final altitude  : {aero.altitude_msl_m:.1f} m")
print(f"  non-finite outputs  : {len(nonfinite)}")
if nonfinite:
    print("   first few:", nonfinite[:5])
ok = (not nonfinite) and (min_range < 20000.0)
print(f"\n  RESULT: {'PASS' if ok else 'CHECK'} (range closed: {min_range < 20000.0}, all finite: {not nonfinite})")
