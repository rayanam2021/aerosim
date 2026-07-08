"""
Inner-loop autopilot FMU for the SHIFT interceptor (three-loop acceleration
autopilot, skid-to-turn).

The autopilot converts the guidance maneuver-acceleration command into fin
deflections using the classic tactical-missile **three-loop acceleration
autopilot**, closed on low-latency inertial measurements:

    outer loop   acceleration.  A PI compensator drives the measured body-axis
                 lateral specific force (from the *accelerometer*, not the
                 navigation filter) to the command.  The accelerometer is used
                 because it is a direct, low-lag measurement of the controlled
                 output; feeding back incidence estimated from the (lagged) nav
                 velocity destabilises the loop at high dynamic pressure.

    inner loop   body rate.  Rate feedback from the *rate gyro* (low latency)
                 damps the airframe short-period mode.

The PI output is mapped to a fin deflection through the analytic control
effectiveness d(a)/d(fin) evaluated at the live flight condition, so the loop
gain is automatically gain-scheduled with dynamic pressure, Mach, mass and
inertia (the loop crossover stays roughly constant across the trajectory).

Two interchangeable designs set the rate-damping gain:

    "lqr"  Each step the short-period model {alpha, q} (and {beta, r}) is built
           from the live flight condition and the algebraic Riccati equation is
           solved for the optimal rate-feedback gain (Zarchan; Nesline &
           Zarchan; Bryson & Ho).

    "pid"  Fixed rate-damping gain plus the same PI acceleration compensator.

Roll is held wings-level by a model-based P-D law that commands a roll-moment
coefficient and inverts the aileron derivative (sign-robust).

Inputs  (aux): a_cmd_{n,e,d}, guidance_active (guidance)
               nav_q{w,x,y,z}, nav_p/q/r, nav_vel_{n,e,d}, nav_valid (ego EKF)
               gyro_{p,q,r}, accel_{x,y,z} (strapdown IMU, low latency)
               qbar_pa, airspeed_mps, mass_kg, Iyy, Izz (flight condition)
Outputs (aux): elevator_cmd_rad, aileron_cmd_rad, rudder_cmd_rad, throttle_cmd,
               az_cmd_mps2, ay_cmd_mps2, az_ach_mps2, ay_ach_mps2
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.linalg import solve_continuous_are
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var
from airframe_geometry import (
    derive_control_derivatives,
    geometry_from_params,
    geometry_param_defaults,
)

GRAVITY = 9.80665


class autopilot_fmu(Fmi3Slave):
    """Three-loop PID / LQR skid-to-turn acceleration autopilot."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT interceptor three-loop acceleration autopilot (PID / LQR)"

        # --- Inputs -----------------------------------------------------------
        self.a_cmd_n = 0.0
        self.a_cmd_e = 0.0
        self.a_cmd_d = 0.0
        self.guidance_active = 0.0
        self.nav_qw = 1.0
        self.nav_qx = 0.0
        self.nav_qy = 0.0
        self.nav_qz = 0.0
        self.nav_p = 0.0
        self.nav_q = 0.0
        self.nav_r = 0.0
        # Raw strapdown IMU (low latency): rate gyro + accelerometer specific
        # force in body FRD.  The inner rate loop and outer acceleration loop are
        # closed on these, not on the navigation-filter estimate (which lags).
        self.gyro_p = 0.0
        self.gyro_q = 0.0
        self.gyro_r = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.nav_vel_n = 0.0
        self.nav_vel_e = 0.0
        self.nav_vel_d = 0.0
        self.nav_valid = 0.0
        self.qbar_pa = 0.0
        self.airspeed_mps = 1.0
        self.mass_kg = 200.0
        self.Iyy = 205.0
        self.Izz = 205.0
        for _n in (
            "a_cmd_n", "a_cmd_e", "a_cmd_d", "guidance_active",
            "nav_qw", "nav_qx", "nav_qy", "nav_qz",
            "nav_p", "nav_q", "nav_r",
            "gyro_p", "gyro_q", "gyro_r",
            "accel_x", "accel_y", "accel_z",
            "nav_vel_n", "nav_vel_e", "nav_vel_d", "nav_valid",
            "qbar_pa", "airspeed_mps", "mass_kg", "Iyy", "Izz",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # --- Outputs ----------------------------------------------------------
        self.elevator_cmd_rad = 0.0
        self.aileron_cmd_rad = 0.0
        self.rudder_cmd_rad = 0.0
        self.throttle_cmd = 1.0
        self.az_cmd_mps2 = 0.0
        self.ay_cmd_mps2 = 0.0
        self.az_ach_mps2 = 0.0
        self.ay_ach_mps2 = 0.0
        for _n in (
            "elevator_cmd_rad", "aileron_cmd_rad", "rudder_cmd_rad", "throttle_cmd",
            "az_cmd_mps2", "ay_cmd_mps2", "az_ach_mps2", "ay_ach_mps2",
        ):
            register_fmu3_var(self, _n, causality="output")

        # --- Parameters -------------------------------------------------------
        # "pid" or "lqr"
        self.controller_type = "lqr"
        register_fmu3_param(self, "controller_type")
        # Close inner rate loop / outer accel loop on the raw IMU (1) or the nav
        # estimate (0).  IMU is the low-latency, robust default.
        self.use_gyro_rate = 1.0
        register_fmu3_param(self, "use_gyro_rate")

        # Reference geometry + dimensional aero derivatives (match the plant).
        # Control derivatives are overwritten from modular canard+tail geometry
        # when use_geometry_aero is enabled (same module as the plant).
        self.ref_area_m2 = 0.0314
        register_fmu3_param(self, "ref_area_m2")
        self.ref_diameter_m = 0.2
        register_fmu3_param(self, "ref_diameter_m")
        self.CN_alpha = 15.0
        register_fmu3_param(self, "CN_alpha")
        self.CN_de = 5.0
        register_fmu3_param(self, "CN_de")
        self.Cm_alpha = -3.0
        register_fmu3_param(self, "Cm_alpha")
        self.Cm_de = -8.0
        register_fmu3_param(self, "Cm_de")
        self.Cm_q = -50.0
        register_fmu3_param(self, "Cm_q")
        self.CY_beta = -15.0
        register_fmu3_param(self, "CY_beta")
        self.CY_dr = 5.0
        register_fmu3_param(self, "CY_dr")
        self.Cn_beta = 3.0
        register_fmu3_param(self, "Cn_beta")
        self.Cn_dr = -8.0
        register_fmu3_param(self, "Cn_dr")
        self.Cn_r = -50.0
        register_fmu3_param(self, "Cn_r")
        self.Cl_da = -2.0
        register_fmu3_param(self, "Cl_da")

        for _k, _v in geometry_param_defaults().items():
            setattr(self, _k, _v)
            register_fmu3_param(self, _k)

        # Outer incidence-trim integral (q-bar independent) + inner rate damping.
        self.ki_accel = 3.0         # incidence-trim integral gain [1/s]
        register_fmu3_param(self, "ki_accel")
        self.kq_rate = 0.30         # rad fin per (rad/s) body rate (PID option)
        register_fmu3_param(self, "kq_rate")
        self.max_incidence_rad = 0.30   # incidence-command clamp (~17 deg)
        register_fmu3_param(self, "max_incidence_rad")

        # LQR short-period weights for the optimal rate-damping design.
        self.lqr_q_angle = 5.0
        register_fmu3_param(self, "lqr_q_angle")
        self.lqr_q_rate = 2.0
        register_fmu3_param(self, "lqr_q_rate")
        self.lqr_r_fin = 200.0
        register_fmu3_param(self, "lqr_r_fin")

        # Roll stabilization (roll-moment-coefficient units, inverted through Cl_da).
        self.kp_roll = 0.4
        register_fmu3_param(self, "kp_roll")
        self.kd_roll = 0.15
        register_fmu3_param(self, "kd_roll")

        self.max_fin_rad = 0.436332       # 25 deg
        register_fmu3_param(self, "max_fin_rad")
        self.max_accel_cmd_mps2 = 300.0   # accel-command clamp (~30 g)
        register_fmu3_param(self, "max_accel_cmd_mps2")

        self._int_pitch = 0.0             # accel-error integral (m/s)
        self._int_yaw = 0.0

    def enter_initialization_mode(self):
        self._apply_geometry()
        self._int_pitch = 0.0
        self._int_yaw = 0.0

    def exit_initialization_mode(self):
        pass

    def _apply_geometry(self) -> None:
        """Match plant control derivatives from the modular canard+tail geometry."""
        if float(getattr(self, "use_geometry_aero", 1.0)) < 0.5:
            return
        params = {k: getattr(self, k) for k in geometry_param_defaults()}
        geom = geometry_from_params(params)
        derivs = derive_control_derivatives(geom)
        self.ref_area_m2 = derivs["ref_area_m2"]
        self.ref_diameter_m = derivs["ref_diameter_m"]
        self.CN_de = derivs["CN_de"]
        self.Cm_de = derivs["Cm_de"]
        self.CY_dr = derivs["CY_dr"]
        self.Cn_dr = derivs["Cn_dr"]
        self.Cl_da = derivs["Cl_da"]

    # ------------------------------------------------------------------- step
    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self.throttle_cmd = 1.0

        if self.nav_valid < 0.5 or self.guidance_active < 0.5:
            self.elevator_cmd_rad = 0.0
            self.aileron_cmd_rad = 0.0
            self.rudder_cmd_rad = 0.0
            self._int_pitch = 0.0
            self._int_yaw = 0.0
            return True

        rot = Rotation.from_quat([self.nav_qx, self.nav_qy, self.nav_qz, self.nav_qw])
        a_cmd_ned = np.array([self.a_cmd_n, self.a_cmd_e, self.a_cmd_d])
        g_ned = np.array([0.0, 0.0, GRAVITY])
        # Required *aerodynamic* body specific force = R^-1(a_cmd - g): the fins
        # must produce the guidance maneuver AND hold the airframe against
        # gravity, so gravity is subtracted here.
        f_req_body = rot.inv().apply(a_cmd_ned - g_ned)
        lim = self.max_accel_cmd_mps2
        az_req = float(np.clip(f_req_body[2], -lim, lim))
        ay_req = float(np.clip(f_req_body[1], -lim, lim))

        # Rate feedback source (raw gyro is the low-latency default).
        if self.use_gyro_rate > 0.5:
            p_fb, q_fb, r_fb = self.gyro_p, self.gyro_q, self.gyro_r
        else:
            p_fb, q_fb, r_fb = self.nav_p, self.nav_q, self.nav_r

        v_ned = np.array([self.nav_vel_n, self.nav_vel_e, self.nav_vel_d])
        v_body = rot.inv().apply(v_ned)
        vspeed = max(float(np.linalg.norm(v_body)), 1.0)
        alpha = float(np.arctan2(v_body[2], v_body[0]))
        beta = float(np.arcsin(np.clip(v_body[1] / vspeed, -1.0, 1.0)))

        qbar = max(self.qbar_pa, 1.0)
        S = self.ref_area_m2
        m = max(self.mass_kg, 1e-3)
        speed = max(float(self.airspeed_mps), 1.0)

        # --- Outer loop: required specific force -> incidence command ---------
        # Quasi-static trim a_z = -(qbar S/m) CN_alpha alpha inverts to alpha_ff.
        # The loop is closed on the *incidence* error (q-bar independent); the
        # slow incidence integrator trims out steady model error / gravity.
        cna = self.CN_alpha if abs(self.CN_alpha) > 1e-6 else 15.0
        cyb = self.CY_beta if abs(self.CY_beta) > 1e-6 else -15.0
        mi = self.max_incidence_rad
        alpha_ff = self._clamp(-az_req * m / (qbar * S * cna), mi)
        beta_ff = self._clamp(ay_req * m / (qbar * S * cyb), mi)
        int_lim = mi / max(self.ki_accel, 1e-6)
        self._int_pitch = float(np.clip(self._int_pitch + (alpha_ff - alpha) * step_size,
                                        -int_lim, int_lim))
        self._int_yaw = float(np.clip(self._int_yaw + (beta_ff - beta) * step_size,
                                      -int_lim, int_lim))
        alpha_cmd = self._clamp(alpha_ff + self.ki_accel * self._int_pitch, mi)
        beta_cmd = self._clamp(beta_ff + self.ki_accel * self._int_yaw, mi)

        # --- Inner loop: trim feed-forward + gyro rate damping ----------------
        # A statically-stable airframe self-trims to the commanded incidence for
        # the feed-forward fin, so NO fast incidence/acceleration feedback is
        # used (both are non-minimum-phase or laggy at high q-bar).  Only the
        # low-latency gyro closes the fast loop, which is minimum-phase and
        # robust.  LQR sets the damping gain optimally; PID uses a fixed gain.
        if str(self.controller_type).strip().lower() == "lqr":
            kq_p = self._lqr_rate_gain(self.CN_alpha, self.Cm_alpha, self.Cm_q,
                                       self.CN_de, self.Cm_de, self.Iyy, qbar, speed, m)
            kq_r = self._lqr_rate_gain(self.CY_beta, self.Cn_beta, self.Cn_r,
                                       self.CY_dr, self.Cn_dr, self.Izz, qbar, speed, m)
        else:
            kq_p = kq_r = float(self.kq_rate)

        de_ff = -self.Cm_alpha * alpha_cmd / self.Cm_de
        dr_ff = -self.Cn_beta * beta_cmd / self.Cn_dr
        de = de_ff + kq_p * q_fb
        dr = dr_ff + kq_r * r_fb

        # --- Roll: wings-level P-D on roll angle + roll-rate ------------------
        roll, _, _ = rot.as_euler("xyz")
        cl_cmd = -(self.kp_roll * roll + self.kd_roll * p_fb)
        cl_da = self.Cl_da if abs(self.Cl_da) > 1e-6 else -2.0
        da = cl_cmd / cl_da

        self.elevator_cmd_rad = self._clamp(de, self.max_fin_rad)
        self.rudder_cmd_rad = self._clamp(dr, self.max_fin_rad)
        self.aileron_cmd_rad = self._clamp(da, self.max_fin_rad)

        # Report commanded aero specific force and model-based achieved value.
        self.az_cmd_mps2, self.ay_cmd_mps2 = az_req, ay_req
        self.az_ach_mps2 = -(qbar * S / m) * cna * alpha
        self.ay_ach_mps2 = (qbar * S / m) * cyb * beta
        return True

    def terminate(self):
        print("Terminating autopilot_fmu.")
        self.time = 0.0

    # ------------------------------------------------------------------- laws
    def _lqr_rate_gain(self, C_force, C_mom, C_momrate, C_force_fin, C_mom_fin,
                       inertia, qbar, speed, m):
        """Optimal body-rate feedback gain from the short-period Riccati soln.

        Builds the linear short-period model {incidence, rate} at the live flight
        condition and returns the (positive) rate-feedback gain used for damping.
        The sign is chosen so that de = ... + kq*rate opposes the rate (the fin
        moment derivative is negative)."""
        S, d = self.ref_area_m2, self.ref_diameter_m
        I = max(inertia, 1e-6)
        Z_a = -qbar * S * C_force / (m * speed)
        M_a = qbar * S * d * C_mom / I
        M_q = qbar * S * d * d * C_momrate / (2.0 * speed * I)
        M_de = qbar * S * d * C_mom_fin / I
        Z_de = -qbar * S * C_force_fin / (m * speed)
        A = np.array([[Z_a, 1.0], [M_a, M_q]])
        B = np.array([[Z_de], [M_de]])
        Q = np.diag([self.lqr_q_angle, self.lqr_q_rate])
        R = np.array([[self.lqr_r_fin]])
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = (np.linalg.inv(R) @ B.T @ P).flatten()
            return float(-K[1])
        except Exception:
            return float(self.kq_rate)

    def _clamp(self, x, lim):
        return float(max(-lim, min(lim, x)))
