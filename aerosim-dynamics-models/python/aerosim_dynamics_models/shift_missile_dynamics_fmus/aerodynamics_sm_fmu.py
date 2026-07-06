"""
Ego SHIFT-missile aerodynamics plant FMU (6-DOF, quaternion attitude).

Role in the interception scenario
----------------------------------
This FMU is the **perfect ground-truth plant** for the ego interceptor
(``actor1``).  It integrates the full six-degree-of-freedom rigid-body
equations of motion (see ``sixdof.py``) using a physically complete analytic
aerodynamic model (all 6 force/moment components), the rocket thrust from the
propulsion FMU, and the mass/inertia from the structures FMU.  It publishes a
noise-free ``VehicleState`` on ``aerosim.actor1.vehicle_state``.

The learned surrogate (partial)
-------------------------------
Separately, it evaluates the local Luminary ``aero_sm`` surrogate
(``mlp_model.pt``) which — in this first iteration — only predicts three of the
six channels:

    surrogate -> [force_x, force_z, moment_y]     (pitch-plane channels)

The remaining three channels (side force ``force_y``, roll moment ``moment_x``,
yaw moment ``moment_z``) are **not known** to the surrogate.  Because the aux
topics are transported as JSON (which cannot carry NaN/Inf), we publish those
unknown channels as a finite sentinel (0.0) together with an explicit
per-channel *validity flag* (1.0 = predicted by the surrogate, 0.0 = unknown).
The corrector FMU reads those flags and reconstructs the full 6-DOF
force/moment vector from ground-truth kinematics — recovering exactly the
channels the surrogate is blind to.

Atmosphere
----------
Density, pressure and speed of sound are evaluated every step from the ICAO ISA
model (``atmosphere.py``) at the live geometric altitude — nothing is hardcoded.

Inputs  (aux):  elevator_rad, aileron_rad, rudder_rad (servo);
                thrust_n (propulsion); mass_kg, Ixx, Iyy, Izz (structures)
Outputs:        vehicle_state (component, truth);
        (aux)   true 6-DOF forces/moments, surrogate 6-DOF forces/moments +
                validity flags, aero angles, atmosphere, mach, q-bar
"""

from __future__ import annotations

import math
import os

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var
from aerosim_data import dict_to_namespace
from aerosim_data import types as aerosim_types

from atmosphere import isa as _isa
from sixdof import (
    integrate_6dof,
    quat_from_euler,
    quat_to_msg,
    ned_to_body,
    alpha_beta,
)


class aerodynamics_sm_fmu(Fmi3Slave):
    """Ego interceptor 6-DOF plant + partial ML aero surrogate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT missile 6-DOF plant + partial aero_sm surrogate"

        # --- Actuator + subsystem inputs (aux) --------------------------------
        self.elevator_rad = 0.0
        self.aileron_rad = 0.0
        self.rudder_rad = 0.0
        self.thrust_n = 0.0
        self.mass_kg = 500.0
        self.Ixx = 5.0
        self.Iyy = 300.0
        self.Izz = 300.0
        for _n in (
            "elevator_rad", "aileron_rad", "rudder_rad",
            "thrust_n", "mass_kg", "Ixx", "Iyy", "Izz",
        ):
            register_fmu3_var(self, _n, causality="input")

        # --- Truth output -----------------------------------------------------
        self.vehicle_state = dict_to_namespace(aerosim_types.VehicleState().to_dict())
        register_fmu3_var(self, "vehicle_state", causality="output")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # --- Auxiliary outputs ------------------------------------------------
        # True (plant) 6-DOF forces/moments in body FRD.
        self.true_fx_n = 0.0
        self.true_fy_n = 0.0
        self.true_fz_n = 0.0
        self.true_mx_nm = 0.0
        self.true_my_nm = 0.0
        self.true_mz_nm = 0.0
        # Surrogate 6-DOF forces/moments (unknown channels carry 0.0 sentinel).
        self.sm_fx_n = 0.0
        self.sm_fy_n = 0.0
        self.sm_fz_n = 0.0
        self.sm_mx_nm = 0.0
        self.sm_my_nm = 0.0
        self.sm_mz_nm = 0.0
        # Per-channel validity flags (1.0 = surrogate-predicted, 0.0 = unknown).
        self.sm_valid_fx = 1.0
        self.sm_valid_fy = 0.0
        self.sm_valid_fz = 1.0
        self.sm_valid_mx = 0.0
        self.sm_valid_my = 1.0
        self.sm_valid_mz = 0.0
        # Flight condition + atmosphere.
        self.alpha_deg = 0.0
        self.beta_deg = 0.0
        self.mach_number = 0.0
        self.dynamic_pressure_pa = 0.0
        self.air_density_kgm3 = 0.0
        self.speed_of_sound_mps = 0.0
        self.altitude_msl_m = 0.0
        self.model_source = "uninitialized"
        # Ego ground-truth kinematics mirrored as scalar aux (so seekers, which
        # cannot bind two VehicleState component inputs, can read the ego state).
        self.ego_pos_n = 0.0
        self.ego_pos_e = 0.0
        self.ego_pos_d = 0.0
        self.ego_vel_n = 0.0
        self.ego_vel_e = 0.0
        self.ego_vel_d = 0.0
        self.ego_qw = 1.0
        self.ego_qx = 0.0
        self.ego_qy = 0.0
        self.ego_qz = 0.0
        self.airspeed_mps = 0.0
        for _n in (
            "true_fx_n", "true_fy_n", "true_fz_n",
            "true_mx_nm", "true_my_nm", "true_mz_nm",
            "sm_fx_n", "sm_fy_n", "sm_fz_n",
            "sm_mx_nm", "sm_my_nm", "sm_mz_nm",
            "sm_valid_fx", "sm_valid_fy", "sm_valid_fz",
            "sm_valid_mx", "sm_valid_my", "sm_valid_mz",
            "alpha_deg", "beta_deg", "mach_number", "dynamic_pressure_pa",
            "air_density_kgm3", "speed_of_sound_mps", "altitude_msl_m",
            "ego_pos_n", "ego_pos_e", "ego_pos_d",
            "ego_vel_n", "ego_vel_e", "ego_vel_d",
            "ego_qw", "ego_qx", "ego_qy", "ego_qz", "airspeed_mps",
            "model_source",
        ):
            register_fmu3_var(self, _n, causality="output")

        # --- Parameters -------------------------------------------------------
        self.mlp_model_path = "mlp_model.pt"
        register_fmu3_param(self, "mlp_model_path")
        self.world_origin_altitude = 0.0
        register_fmu3_param(self, "world_origin_altitude")

        # Reference geometry.
        self.ref_area_m2 = 0.0314   # pi/4 * d^2, d=0.2 m
        register_fmu3_param(self, "ref_area_m2")
        self.ref_diameter_m = 0.2
        register_fmu3_param(self, "ref_diameter_m")

        # Aerodynamic derivatives (per rad unless noted); representative supersonic
        # missile static + damping coefficients.
        self.CA0 = 0.30
        register_fmu3_param(self, "CA0")
        self.CN_alpha = 15.0
        register_fmu3_param(self, "CN_alpha")
        self.CN_de = 5.0
        register_fmu3_param(self, "CN_de")
        self.CY_beta = -15.0
        register_fmu3_param(self, "CY_beta")
        self.CY_dr = 5.0
        register_fmu3_param(self, "CY_dr")
        self.Cm_alpha = -3.0
        register_fmu3_param(self, "Cm_alpha")
        self.Cm_de = -8.0
        register_fmu3_param(self, "Cm_de")
        self.Cm_q = -50.0
        register_fmu3_param(self, "Cm_q")
        self.Cn_beta = 3.0
        register_fmu3_param(self, "Cn_beta")
        self.Cn_dr = -8.0
        register_fmu3_param(self, "Cn_dr")
        self.Cn_r = -50.0
        register_fmu3_param(self, "Cn_r")
        self.Cl_p = -5.0
        register_fmu3_param(self, "Cl_p")
        self.Cl_da = -2.0
        register_fmu3_param(self, "Cl_da")

        # Feature clamps for the surrogate's valid domain.
        self.min_mach = 0.3
        register_fmu3_param(self, "min_mach")
        self.max_mach = 4.0
        register_fmu3_param(self, "max_mach")
        self.max_abs_alpha_deg = 20.0
        register_fmu3_param(self, "max_abs_alpha_deg")

        # Initial conditions (NED, rad, m/s).
        self.init_pos_north_m = 0.0
        register_fmu3_param(self, "init_pos_north_m")
        self.init_pos_east_m = 0.0
        register_fmu3_param(self, "init_pos_east_m")
        self.init_pos_down_m = -5000.0
        register_fmu3_param(self, "init_pos_down_m")
        self.init_roll_rad = 0.0
        register_fmu3_param(self, "init_roll_rad")
        self.init_pitch_rad = 0.0
        register_fmu3_param(self, "init_pitch_rad")
        self.init_yaw_rad = 0.0
        register_fmu3_param(self, "init_yaw_rad")
        self.init_speed_mps = 600.0
        register_fmu3_param(self, "init_speed_mps")

        self.mass_fallback_kg = 500.0
        register_fmu3_param(self, "mass_fallback_kg")

        # Internal integration sub-step: the high-q airframe is numerically stiff,
        # so the plant integrates several small steps per sim tick for stability
        # regardless of the (coarser) co-simulation step size.
        self.max_substep_s = 0.001
        register_fmu3_param(self, "max_substep_s")

        # --- Internal state ---------------------------------------------------
        self._model = None
        self._torch = None
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._omega = np.zeros(3)
        self._isa_rho = 1.225
        self._isa_a = 340.29

    # ------------------------------------------------------------------ setup
    def enter_initialization_mode(self):
        self._load_model()
        self._pos = np.array(
            [self.init_pos_north_m, self.init_pos_east_m, self.init_pos_down_m],
            dtype=float,
        )
        self._quat = quat_from_euler(
            self.init_roll_rad, self.init_pitch_rad, self.init_yaw_rad
        )
        p, y = self.init_pitch_rad, self.init_yaw_rad
        self._vel = self.init_speed_mps * np.array(
            [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)]
        )
        self._omega = np.zeros(3)
        if self.mass_kg <= 0.0:
            self.mass_kg = self.mass_fallback_kg
        self._refresh_atmosphere()
        self._write_vehicle_state(np.zeros(3))

    def exit_initialization_mode(self):
        pass

    def _load_model(self) -> None:
        try:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import torch  # noqa: F401
            self._torch = torch
        except Exception as exc:
            self._torch = None
            self._model = None
            self.model_source = f"analytic (torch unavailable: {exc})"
            return
        path = self._resolve_model_path(self.mlp_model_path)
        if path is None:
            self._model = None
            self.model_source = "analytic (mlp_model.pt not found)"
            return
        try:
            self._model = self._torch.jit.load(path, map_location="cpu")
            self._model.eval()
            self.model_source = f"mlp:{os.path.basename(path)}"
        except Exception as exc:
            self._model = None
            self.model_source = f"analytic (load failed: {exc})"

    @staticmethod
    def _resolve_model_path(path: str):
        cands = []
        if path:
            cands.append(path)
            if not os.path.isabs(path):
                here = os.path.dirname(os.path.abspath(__file__))
                cands.append(os.path.join(here, path))
        for c in cands:
            if os.path.isfile(c):
                return c
        return None

    # ------------------------------------------------------------------- step
    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size

        inertia = np.array([max(self.Ixx, 1e-6), max(self.Iyy, 1e-6), max(self.Izz, 1e-6)])
        n_sub = max(1, int(math.ceil(step_size / max(self.max_substep_s, 1e-4))))
        h = step_size / n_sub

        fx = fy = fz = mx = my = mz = 0.0
        alpha = beta = qbar = mach = 0.0
        accel_ned = np.zeros(3)
        for _ in range(n_sub):
            self._refresh_atmosphere()
            v_body = ned_to_body(self._quat, self._vel)
            speed = float(np.linalg.norm(self._vel))
            alpha, beta = alpha_beta(v_body)
            qbar = 0.5 * self._isa_rho * speed * speed
            mach = speed / max(self._isa_a, 1e-3)

            fx, fy, fz, mx, my, mz = self._true_aero(qbar, alpha, beta, speed)
            force_body = np.array([fx + self.thrust_n, fy, fz])  # thrust along +x
            moment_body = np.array([mx, my, mz])

            self._pos, self._vel, self._quat, self._omega, accel_ned = integrate_6dof(
                self._pos, self._vel, self._quat, self._omega,
                force_body, moment_body, self.mass_kg, inertia, h,
            )

        # Evaluate the partial ML surrogate on the final flight condition.
        self._evaluate_surrogate(mach, math.degrees(alpha), self.elevator_rad)

        # Publish outputs (aero forces exclude thrust; thrust is a separate input).
        self.true_fx_n, self.true_fy_n, self.true_fz_n = fx, fy, fz
        self.true_mx_nm, self.true_my_nm, self.true_mz_nm = mx, my, mz
        self.alpha_deg = math.degrees(alpha)
        self.beta_deg = math.degrees(beta)
        self.mach_number = mach
        self.dynamic_pressure_pa = qbar
        self._write_vehicle_state(accel_ned)
        return True

    def terminate(self):
        print("Terminating aerodynamics_sm_fmu (ego 6-DOF plant).")
        self._model = None
        self.time = 0.0

    # ------------------------------------------------------- aero + surrogate
    def _true_aero(self, qbar, alpha, beta, speed):
        """Full analytic 6-DOF aerodynamic forces/moments (body FRD)."""
        S = self.ref_area_m2
        d = self.ref_diameter_m
        qS = qbar * S
        qSd = qbar * S * d

        # Non-dimensional body rates for damping terms.
        two_v = 2.0 * max(speed, 1e-3)
        p_hat = self._omega[0] * d / two_v
        q_hat = self._omega[1] * d / two_v
        r_hat = self._omega[2] * d / two_v

        de, da, dr = self.elevator_rad, self.aileron_rad, self.rudder_rad

        CA = self.CA0
        CN = self.CN_alpha * alpha + self.CN_de * de
        CY = self.CY_beta * beta + self.CY_dr * dr
        Cl = self.Cl_p * p_hat + self.Cl_da * da
        Cm = self.Cm_alpha * alpha + self.Cm_de * de + self.Cm_q * q_hat
        Cn = self.Cn_beta * beta + self.Cn_dr * dr + self.Cn_r * r_hat

        fx = -CA * qS          # axial (drag) acts rearward
        fy = CY * qS           # side force
        fz = -CN * qS          # normal force (+alpha -> upward = -z)
        mx = Cl * qSd          # roll
        my = Cm * qSd          # pitch
        mz = Cn * qSd          # yaw
        return fx, fy, fz, mx, my, mz

    def _evaluate_surrogate(self, mach, alpha_deg, elevator_rad):
        """Partial ML surrogate: predicts only fx, fz, my. Others -> unknown."""
        mach_c = float(np.clip(mach, self.min_mach, self.max_mach))
        alpha_c = float(np.clip(alpha_deg, -self.max_abs_alpha_deg, self.max_abs_alpha_deg))
        elev_deg = math.degrees(elevator_rad)

        fx = fz = my = 0.0
        used_mlp = False
        if self._model is not None and self._torch is not None:
            try:
                feats = self._torch.tensor(
                    [[mach_c, alpha_c, elev_deg]], dtype=self._torch.float32
                )
                with self._torch.no_grad():
                    out = self._model(feats).squeeze().tolist()
                fx, fz, my = float(out[0]), float(out[1]), float(out[2])
                used_mlp = True
            except Exception as exc:
                self.model_source = f"analytic (inference failed: {exc})"

        if not used_mlp:
            # Analytic stand-in for the 3 surrogate channels (consistent w/ trainer).
            v = max(mach_c * self._isa_a, 1.0)
            qbar_s = 0.5 * self._isa_rho * v * v * self.ref_area_m2
            cd = 0.30 + 0.015 * alpha_c * alpha_c
            cz = -(0.11 * alpha_c + 0.045 * elev_deg)
            cm = -(0.020 * alpha_c + 0.060 * elev_deg)
            fx, fz, my = -cd * qbar_s, cz * qbar_s, cm * qbar_s

        # Known (surrogate-predicted) channels.
        self.sm_fx_n, self.sm_fz_n, self.sm_my_nm = fx, fz, my
        self.sm_valid_fx = self.sm_valid_fz = self.sm_valid_my = 1.0
        # Unknown channels: finite sentinel + validity flag = 0 (transport-safe null).
        self.sm_fy_n = self.sm_mx_nm = self.sm_mz_nm = 0.0
        self.sm_valid_fy = self.sm_valid_mx = self.sm_valid_mz = 0.0

    # ------------------------------------------------------------- atmosphere
    def _altitude_msl(self) -> float:
        return self.world_origin_altitude - float(self._pos[2])

    def _refresh_atmosphere(self) -> None:
        _, _, rho, a = _isa(self._altitude_msl())
        self._isa_rho = rho
        self._isa_a = a
        self.air_density_kgm3 = rho
        self.speed_of_sound_mps = a
        self.altitude_msl_m = self._altitude_msl()

    # ---------------------------------------------------------------- output
    def _write_vehicle_state(self, accel_ned: np.ndarray) -> None:
        self.vehicle_state.state.pose.position.x = float(self._pos[0])
        self.vehicle_state.state.pose.position.y = float(self._pos[1])
        self.vehicle_state.state.pose.position.z = float(self._pos[2])
        qw, qx, qy, qz = quat_to_msg(self._quat)
        self.vehicle_state.state.pose.orientation.w = qw
        self.vehicle_state.state.pose.orientation.x = qx
        self.vehicle_state.state.pose.orientation.y = qy
        self.vehicle_state.state.pose.orientation.z = qz
        self.vehicle_state.velocity.x = float(self._vel[0])
        self.vehicle_state.velocity.y = float(self._vel[1])
        self.vehicle_state.velocity.z = float(self._vel[2])
        self.vehicle_state.angular_velocity.x = float(self._omega[0])
        self.vehicle_state.angular_velocity.y = float(self._omega[1])
        self.vehicle_state.angular_velocity.z = float(self._omega[2])
        self.vehicle_state.acceleration.x = float(accel_ned[0])
        self.vehicle_state.acceleration.y = float(accel_ned[1])
        self.vehicle_state.acceleration.z = float(accel_ned[2])

        # Mirror truth kinematics to scalar aux for the seekers/consumers.
        self.ego_pos_n, self.ego_pos_e, self.ego_pos_d = (float(v) for v in self._pos)
        self.ego_vel_n, self.ego_vel_e, self.ego_vel_d = (float(v) for v in self._vel)
        self.ego_qw, self.ego_qx, self.ego_qy, self.ego_qz = qw, qx, qy, qz
        self.airspeed_mps = float(np.linalg.norm(self._vel))
