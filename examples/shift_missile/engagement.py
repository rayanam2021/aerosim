"""
Standalone closed-loop SHIFT-missile engagement simulator.

Runs the *entire* FMU stack (threat + ego 6-DOF plant + propulsion + structures +
sensors + seekers + GNC + corrector) in a single Python process by stubbing the
AeroSim FMU runtime, wired exactly like the composed sim-config.  It exposes a
callable ``run_engagement(overrides=...)`` that returns the miss distance and
telemetry for one engagement — fast enough (faster than real time) to drive a
Monte-Carlo P_kill campaign without the distributed Kafka/orchestrator stack.

This module is the shared engine for ``run_monte_carlo.py`` and can also be run
directly as a single-shot smoke test::

    python examples/shift_missile/engagement.py [propnav|mpc] [lqr|pid]
"""

from __future__ import annotations

import importlib.util
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
sys.modules.setdefault("pythonfmu3", _pfmu)

_core = _pytypes.ModuleType("aerosim_core")
_core.register_fmu3_var = lambda self, name, causality=None: None
_core.register_fmu3_param = lambda self, name: None
sys.modules.setdefault("aerosim_core", _core)


class _NS:
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
sys.modules.setdefault("aerosim_data", _data)
sys.modules.setdefault("aerosim_data.types", _types)

# ── Import the FMU source modules ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DYN = os.path.join(ROOT, "aerosim-dynamics-models/python/aerosim_dynamics_models/shift_missile_dynamics_fmus")
CTL = os.path.join(ROOT, "aerosim-controllers/python/aerosim_controllers/shift_missile_controller_fmus")
SEN = os.path.join(ROOT, "aerosim-sensors/python/aerosim_sensors/shift_missile_sensor_fmus")
SCE = os.path.join(ROOT, "aerosim-scenarios/python/aerosim_scenarios/shift_missile_scenario_fmus")
for _p in (DYN, CTL, SEN, SCE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(folder, fname, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(folder, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_MODS = {
    "aerodynamics_sm": _load(DYN, "aerodynamics_sm_fmu.py", "aerodynamics_sm_fmu").aerodynamics_sm_fmu,
    "servo_sm": _load(DYN, "servo_sm_fmu.py", "servo_sm_fmu").servo_sm_fmu,
    "structures_sm": _load(DYN, "structures_sm_fmu.py", "structures_sm_fmu").structures_sm_fmu,
    "propulsion_sm": _load(DYN, "propulsion_sm_fmu.py", "propulsion_sm_fmu").propulsion_sm_fmu,
    "corrector": _load(DYN, "corrector_fmu.py", "corrector_fmu").corrector_fmu,
    "guidance": _load(CTL, "guidance_fmu.py", "guidance_fmu").guidance_fmu,
    "autopilot": _load(CTL, "autopilot_fmu.py", "autopilot_fmu").autopilot_fmu,
    "ego_nav_ekf": _load(CTL, "ego_nav_ekf_fmu.py", "ego_nav_ekf_fmu").ego_nav_ekf_fmu,
    "target_nav_ekf": _load(CTL, "target_nav_ekf_fmu.py", "target_nav_ekf_fmu").target_nav_ekf_fmu,
    "imu_sensor": _load(SEN, "imu_fmu.py", "imu_fmu").imu_fmu,
    "gnss_sensor": _load(SEN, "gnss_fmu.py", "gnss_fmu").gnss_fmu,
    "baro_sensor": _load(SEN, "baro_fmu.py", "baro_fmu").baro_fmu,
    "ir_seeker": _load(SEN, "ir_seeker_fmu.py", "ir_seeker_fmu").ir_seeker_fmu,
    "radar_seeker": _load(SEN, "semi_active_radar_fmu.py", "semi_active_radar_fmu").semi_active_radar_fmu,
    "threat_missile": _load(SCE, "threat_missile_fmu.py", "threat_missile_fmu").threat_missile_fmu,
}

# Import the composer's parameter merge so the harness uses the exact same
# modular missile/target/scenario parameters as the distributed run.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_sim_config as _composer  # noqa: E402


def default_params(master_path: str | None = None) -> dict:
    """Merged per-FMU parameter dict from the modular config (composer merge)."""
    if master_path is None:
        master_path = os.path.join(_composer.MODULAR_DIR, "master_intercept.json")
    bundle = _composer.load_scenario(master_path)
    return _composer._merged_params(bundle)


def _apply(fmu, vals: dict) -> None:
    for k, v in vals.items():
        setattr(fmu, k, v)


def _apply_overrides(params: dict, overrides: dict) -> dict:
    """Apply dotted 'fmu_id.param' overrides ('target.*' -> threat_missile)."""
    out = {k: dict(v) for k, v in params.items()}
    for dotted, value in (overrides or {}).items():
        fmu_id, key = dotted.split(".", 1)
        if fmu_id == "target":
            fmu_id = "threat_missile"
        out.setdefault(fmu_id, {})[key] = value
    return out


def run_engagement(
    overrides: dict | None = None,
    params: dict | None = None,
    sim_time_s: float = 40.0,
    dt: float = 0.01,
    seed: int = 2024,
    collect_telemetry: bool = False,
) -> dict:
    """Simulate one closed-loop engagement; return miss distance + diagnostics."""
    if params is None:
        params = default_params()
    p = _apply_overrides(params, overrides or {})
    # Per-run deterministic seeding of the stochastic FMUs.
    p.setdefault("corrector", {})["rng_seed"] = int(seed)
    np.random.seed(int(seed) & 0x7FFFFFFF)

    f = {name: cls() for name, cls in _MODS.items()}
    for name, fmu in f.items():
        _apply(fmu, p.get(name, {}))

    order_init = list(f.values())
    for fmu in order_init:
        fmu.enter_initialization_mode()
        if hasattr(fmu, "exit_initialization_mode"):
            fmu.exit_initialization_mode()

    aero, servo, prop, struct = f["aerodynamics_sm"], f["servo_sm"], f["propulsion_sm"], f["structures_sm"]
    corr, guid, auto = f["corrector"], f["guidance"], f["autopilot"]
    egonav, tgtnav = f["ego_nav_ekf"], f["target_nav_ekf"]
    imu, gnss, baro = f["imu_sensor"], f["gnss_sensor"], f["baro_sensor"]
    ir, radar, threat = f["ir_seeker"], f["radar_seeker"], f["threat_missile"]

    def wire():
        servo.elevator_cmd_rad = auto.elevator_cmd_rad
        servo.aileron_cmd_rad = auto.aileron_cmd_rad
        servo.rudder_cmd_rad = auto.rudder_cmd_rad
        servo.throttle_cmd = auto.throttle_cmd
        prop.throttle = servo.throttle
        prop.ambient_pressure_pa = aero.air_pressure_pa
        struct.propellant_fraction = prop.propellant_fraction
        struct.true_fz_n = aero.true_fz_n
        struct.true_fy_n = aero.true_fy_n
        struct.thrust_n = prop.thrust_n
        aero.elevator_rad = servo.elevator_rad
        aero.aileron_rad = servo.aileron_rad
        aero.rudder_rad = servo.rudder_rad
        aero.thrust_n = prop.thrust_n
        aero.mass_kg = struct.mass_kg
        aero.Ixx, aero.Iyy, aero.Izz = struct.Ixx, struct.Iyy, struct.Izz
        imu.vehicle_state = aero.vehicle_state
        gnss.vehicle_state = aero.vehicle_state
        baro.vehicle_state = aero.vehicle_state
        corr.vehicle_state = aero.vehicle_state
        for s in (ir, radar):
            s.ego_pos_n, s.ego_pos_e, s.ego_pos_d = aero.ego_pos_n, aero.ego_pos_e, aero.ego_pos_d
            s.ego_qw, s.ego_qx, s.ego_qy, s.ego_qz = aero.ego_qw, aero.ego_qx, aero.ego_qy, aero.ego_qz
            s.tgt_pos_n, s.tgt_pos_e, s.tgt_pos_d = threat.threat_pos_n, threat.threat_pos_e, threat.threat_pos_d
        radar.ego_vel_n, radar.ego_vel_e, radar.ego_vel_d = aero.ego_vel_n, aero.ego_vel_e, aero.ego_vel_d
        radar.tgt_vel_n, radar.tgt_vel_e, radar.tgt_vel_d = threat.threat_vel_n, threat.threat_vel_e, threat.threat_vel_d
        for a in ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2", "gyro_x_rps", "gyro_y_rps", "gyro_z_rps"):
            setattr(egonav, a, getattr(imu, a))
        egonav.imu_measurement_ready = imu.measurement_ready
        for a in ("pos_n_m", "pos_e_m", "pos_d_m", "vel_n_mps", "vel_e_mps", "vel_d_mps"):
            setattr(egonav, a, getattr(gnss, a))
        egonav.gnss_measurement_ready = gnss.measurement_ready
        egonav.baro_alt_m = baro.baro_alt_m
        egonav.baro_measurement_ready = baro.measurement_ready
        for a in ("sm_fx_n", "sm_fy_n", "sm_fz_n", "sm_mx_nm", "sm_my_nm", "sm_mz_nm",
                  "sm_valid_fx", "sm_valid_fy", "sm_valid_fz", "sm_valid_mx", "sm_valid_my", "sm_valid_mz",
                  "sm_sigma_fx", "sm_sigma_fy", "sm_sigma_fz", "sm_sigma_mx", "sm_sigma_my", "sm_sigma_mz"):
            setattr(corr, a, getattr(aero, a))
        corr.thrust_n = prop.thrust_n
        corr.mass_kg, corr.Ixx, corr.Iyy, corr.Izz = struct.mass_kg, struct.Ixx, struct.Iyy, struct.Izz
        tgtnav.ego_pos_n, tgtnav.ego_pos_e, tgtnav.ego_pos_d = egonav.nav_pos_n, egonav.nav_pos_e, egonav.nav_pos_d
        tgtnav.ego_vel_n, tgtnav.ego_vel_e, tgtnav.ego_vel_d = egonav.nav_vel_n, egonav.nav_vel_e, egonav.nav_vel_d
        tgtnav.radar_range_m, tgtnav.radar_az_rad = radar.radar_range_m, radar.radar_az_rad
        tgtnav.radar_el_rad, tgtnav.radar_range_rate_mps = radar.radar_el_rad, radar.radar_range_rate_mps
        tgtnav.radar_ready = radar.measurement_ready
        tgtnav.ir_az_rad, tgtnav.ir_el_rad, tgtnav.ir_ready = ir.ir_az_rad, ir.ir_el_rad, ir.measurement_ready
        guid.nav_pos_n, guid.nav_pos_e, guid.nav_pos_d = egonav.nav_pos_n, egonav.nav_pos_e, egonav.nav_pos_d
        guid.nav_vel_n, guid.nav_vel_e, guid.nav_vel_d = egonav.nav_vel_n, egonav.nav_vel_e, egonav.nav_vel_d
        guid.nav_valid = egonav.nav_valid
        guid.tgt_pos_n, guid.tgt_pos_e, guid.tgt_pos_d = tgtnav.tgt_pos_n, tgtnav.tgt_pos_e, tgtnav.tgt_pos_d
        guid.tgt_vel_n, guid.tgt_vel_e, guid.tgt_vel_d = tgtnav.tgt_vel_n, tgtnav.tgt_vel_e, tgtnav.tgt_vel_d
        guid.tgt_acc_n, guid.tgt_acc_e, guid.tgt_acc_d = tgtnav.tgt_acc_n, tgtnav.tgt_acc_e, tgtnav.tgt_acc_d
        guid.tgt_valid = tgtnav.tgt_valid
        auto.a_cmd_n, auto.a_cmd_e, auto.a_cmd_d = guid.a_cmd_n, guid.a_cmd_e, guid.a_cmd_d
        auto.guidance_active = guid.guidance_active
        auto.nav_qw, auto.nav_qx, auto.nav_qy, auto.nav_qz = egonav.nav_qw, egonav.nav_qx, egonav.nav_qy, egonav.nav_qz
        auto.nav_p, auto.nav_q, auto.nav_r = egonav.nav_p, egonav.nav_q, egonav.nav_r
        auto.gyro_p, auto.gyro_q, auto.gyro_r = imu.gyro_x_rps, imu.gyro_y_rps, imu.gyro_z_rps
        auto.accel_x, auto.accel_y, auto.accel_z = imu.accel_x_mps2, imu.accel_y_mps2, imu.accel_z_mps2
        auto.nav_vel_n, auto.nav_vel_e, auto.nav_vel_d = egonav.nav_vel_n, egonav.nav_vel_e, egonav.nav_vel_d
        auto.nav_valid = egonav.nav_valid
        auto.qbar_pa, auto.airspeed_mps = aero.dynamic_pressure_pa, aero.airspeed_mps
        auto.mass_kg, auto.Iyy, auto.Izz = struct.mass_kg, struct.Iyy, struct.Izz

    def rel_vec():
        return np.array([threat.threat_pos_n - aero.ego_pos_n,
                         threat.threat_pos_e - aero.ego_pos_e,
                         threat.threat_pos_d - aero.ego_pos_d])

    step_order = [threat, servo, prop, struct, aero, imu, gnss, baro, ir, radar,
                  egonav, tgtnav, guid, auto, corr]

    t = 0.0
    n_steps = int(sim_time_s / dt)
    min_range = 1e18
    min_range_t = 0.0
    prev_range = None
    telem = [] if collect_telemetry else None
    nonfinite = 0

    for i in range(n_steps):
        wire()
        for fmu in step_order:
            fmu.do_step(t, dt)
        t += dt
        r = float(np.linalg.norm(rel_vec()))
        if r < min_range:
            min_range, min_range_t = r, t
        if not np.isfinite(aero.ego_pos_n):
            nonfinite += 1
            break
        if collect_telemetry and i % 10 == 0:
            acmd = float(np.hypot(guid.a_cmd_n, np.hypot(guid.a_cmd_e, guid.a_cmd_d)))
            telem.append((t, r, aero.altitude_msl_m, aero.mach_number,
                          guid.guidance_phase, struct.margin_of_safety,
                          prop.thrust_n, corr.ensemble_spread,
                          aero.alpha_deg, aero.airspeed_mps, egonav.nav_valid,
                          acmd, np.degrees(auto.elevator_cmd_rad)))
        # Endgame: closest approach passed (range now increasing after closing).
        if prev_range is not None and r > prev_range and r < 500.0 and t > 1.0:
            break
        prev_range = r

    return {
        "miss_distance_m": min_range,
        "intercept_time_s": min_range_t,
        "final_range_m": prev_range if prev_range is not None else min_range,
        "nonfinite": nonfinite,
        "model_source": getattr(aero, "model_source", ""),
        "final_mach": aero.mach_number,
        "final_margin_of_safety": struct.margin_of_safety,
        "total_impulse_ns": prop.total_impulse_ns,
        "burn_time_s": prop.burn_time_s,
        "telemetry": telem,
    }


if __name__ == "__main__":
    law = sys.argv[1] if len(sys.argv) > 1 else "propnav"
    ctl = sys.argv[2] if len(sys.argv) > 2 else "lqr"
    res = run_engagement(
        overrides={"guidance.guidance_law": law, "autopilot.controller_type": ctl},
        collect_telemetry=True,
    )
    print(f"[engagement] law={law} controller={ctl} model={res['model_source']}")
    print(f"{'t[s]':>6} {'range[m]':>10} {'alt[m]':>8} {'mach':>5} {'ph':>3} "
          f"{'MoS':>6} {'nav':>3} {'|acmd|':>8} {'alpha':>7} {'elev':>7}")
    for row in (res["telemetry"] or [])[::10]:
        (tt, r, alt, mach, phase, mos, thr, spr, alpha, vas, navok, acmd, elev) = row
        print(f"{tt:6.2f} {r:10.1f} {alt:8.1f} {mach:5.2f} {phase:3.0f} "
              f"{mos:6.2f} {navok:3.0f} {acmd:8.1f} {alpha:7.2f} {elev:7.2f}")
    print(f"\n  miss distance   : {res['miss_distance_m']:.1f} m at t={res['intercept_time_s']:.2f} s")
    print(f"  final mach      : {res['final_mach']:.2f}")
    print(f"  total impulse   : {res['total_impulse_ns']/1000:.1f} kN.s over {res['burn_time_s']:.2f} s")
    print(f"  min struct MoS  : {res['final_margin_of_safety']:.2f}")
    print(f"  RESULT: {'HIT' if res['miss_distance_m'] < 20.0 else 'MISS'}")
