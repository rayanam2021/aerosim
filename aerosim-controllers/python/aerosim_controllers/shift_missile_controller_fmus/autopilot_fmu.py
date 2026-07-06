"""
Inner-loop autopilot FMU for the SHIFT interceptor.

Tracks the acceleration command from the guidance FMU by deflecting the pitch
(elevator), yaw (rudder) and roll (aileron) fins.  A skid-to-turn (STT) strategy
is used: the pitch and yaw channels generate the commanded lateral accelerations
while the roll channel holds wings level.

Two interchangeable inner-loop laws are selectable via ``controller_type``:

    "pid"  Fixed-gain PID on the angle-of-attack / sideslip error with body-rate
           damping and a moment-trim feed-forward.
    "lqr"  Optimal state feedback: at every step the short-period {alpha,q} and
           {beta,r} linear models are built from the live flight condition
           (q-bar, speed, mass, inertia + aero derivatives) and the algebraic
           Riccati equation is solved to obtain the feedback gains.

All feedback uses the *navigation estimate* (attitude, body rates, velocity),
never ground truth.  Gravity is fed forward so the commanded specific force
matches what the accelerometers will sense.

Inputs  (aux): a_cmd_{n,e,d}, guidance_active (guidance)
               nav_q{w,x,y,z}, nav_p/q/r, nav_vel_{n,e,d}, nav_valid (ego EKF)
               qbar_pa, airspeed_mps, mass_kg, Iyy, Izz (flight condition)
Outputs (aux): elevator_cmd_rad, aileron_cmd_rad, rudder_cmd_rad, throttle_cmd
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.linalg import solve_continuous_are
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var

GRAVITY = 9.80665


class autopilot_fmu(Fmi3Slave):
    """PID / LQR selectable 3-axis skid-to-turn autopilot."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT interceptor inner-loop autopilot (PID / LQR)"

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
        self.nav_vel_n = 0.0
        self.nav_vel_e = 0.0
        self.nav_vel_d = 0.0
        self.nav_valid = 0.0
        self.qbar_pa = 0.0
        self.airspeed_mps = 1.0
        self.mass_kg = 500.0
        self.Iyy = 300.0
        self.Izz = 300.0
        for _n in (
            "a_cmd_n", "a_cmd_e", "a_cmd_d", "guidance_active",
            "nav_qw", "nav_qx", "nav_qy", "nav_qz",
            "nav_p", "nav_q", "nav_r",
            "nav_vel_n", "nav_vel_e", "nav_vel_d", "nav_valid",
            "qbar_pa", "airspeed_mps", "mass_kg", "Iyy", "Izz",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.elevator_cmd_rad = 0.0
        self.aileron_cmd_rad = 0.0
        self.rudder_cmd_rad = 0.0
        self.throttle_cmd = 1.0
        for _n in ("elevator_cmd_rad", "aileron_cmd_rad", "rudder_cmd_rad", "throttle_cmd"):
            register_fmu3_var(self, _n, causality="output")

        # "pid" or "lqr"
        self.controller_type = "lqr"
        register_fmu3_param(self, "controller_type")

        # Reference geometry + aero derivatives (match the plant defaults).
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

        # PID gains (on alpha/beta error + rate damping).
        self.kp_pitch = 6.0
        register_fmu3_param(self, "kp_pitch")
        self.ki_pitch = 2.0
        register_fmu3_param(self, "ki_pitch")
        self.kd_pitch = 0.6
        register_fmu3_param(self, "kd_pitch")
        self.kp_yaw = 6.0
        register_fmu3_param(self, "kp_yaw")
        self.ki_yaw = 2.0
        register_fmu3_param(self, "ki_yaw")
        self.kd_yaw = 0.6
        register_fmu3_param(self, "kd_yaw")

        # LQR weights.
        self.lqr_q_angle = 100.0
        register_fmu3_param(self, "lqr_q_angle")
        self.lqr_q_rate = 1.0
        register_fmu3_param(self, "lqr_q_rate")
        self.lqr_r_fin = 20.0
        register_fmu3_param(self, "lqr_r_fin")

        # Roll stabilisation (gains are in roll-moment-coefficient units; the
        # commanded Cl is inverted through Cl_da to get the aileron angle, so the
        # sign is correct regardless of the aileron derivative's sign).
        self.kp_roll = 0.4
        register_fmu3_param(self, "kp_roll")
        self.kd_roll = 0.15
        register_fmu3_param(self, "kd_roll")

        self.max_fin_rad = 0.436332  # 25 deg
        register_fmu3_param(self, "max_fin_rad")
        self.max_alpha_cmd_rad = 0.349066  # 20 deg
        register_fmu3_param(self, "max_alpha_cmd_rad")

        self._int_pitch = 0.0
        self._int_yaw = 0.0

    def enter_initialization_mode(self):
        self._int_pitch = 0.0
        self._int_yaw = 0.0

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self.throttle_cmd = 1.0

        if self.nav_valid < 0.5 or self.guidance_active < 0.5:
            self.elevator_cmd_rad = 0.0
            self.aileron_cmd_rad = 0.0
            self.rudder_cmd_rad = 0.0
            return True

        rot = Rotation.from_quat([self.nav_qx, self.nav_qy, self.nav_qz, self.nav_qw])
        # Desired specific force in body = R^-1 (a_cmd - g).
        a_cmd_ned = np.array([self.a_cmd_n, self.a_cmd_e, self.a_cmd_d])
        g_ned = np.array([0.0, 0.0, GRAVITY])
        f_des_body = rot.inv().apply(a_cmd_ned - g_ned)
        ay_cmd = float(f_des_body[1])
        az_cmd = float(f_des_body[2])

        # Estimated aero angles from nav velocity.
        v_ned = np.array([self.nav_vel_n, self.nav_vel_e, self.nav_vel_d])
        v_body = rot.inv().apply(v_ned)
        speed = max(float(np.linalg.norm(v_body)), 1.0)
        alpha = float(np.arctan2(v_body[2], v_body[0]))
        beta = float(np.arcsin(np.clip(v_body[1] / speed, -1.0, 1.0)))

        qbar = max(self.qbar_pa, 1.0)
        S = self.ref_area_m2
        m = max(self.mass_kg, 1e-3)

        # Map lateral accel commands to angle commands (quasi-static).
        # a_z = -qbar S CN_alpha alpha / m ; a_y = qbar S CY_beta beta / m
        alpha_cmd = self._clamp(
            -az_cmd * m / (qbar * S * self.CN_alpha), self.max_alpha_cmd_rad
        )
        beta_cmd = self._clamp(
            ay_cmd * m / (qbar * S * self.CY_beta), self.max_alpha_cmd_rad
        )

        if str(self.controller_type).strip().lower() == "lqr":
            de = self._lqr_pitch(alpha, alpha_cmd, self.nav_q, qbar, speed)
            dr = self._lqr_yaw(beta, beta_cmd, self.nav_r, qbar, speed)
        else:
            de, dr = self._pid(alpha, alpha_cmd, beta, beta_cmd, step_size)

        # Roll stabilisation (hold wings level): command a restoring roll-moment
        # coefficient, then invert the aileron derivative to get the deflection.
        roll, _, _ = rot.as_euler("xyz")
        cl_cmd = -(self.kp_roll * roll + self.kd_roll * self.nav_p)
        cl_da = self.Cl_da if abs(self.Cl_da) > 1e-6 else -2.0
        da = cl_cmd / cl_da

        self.elevator_cmd_rad = self._clamp(de, self.max_fin_rad)
        self.rudder_cmd_rad = self._clamp(dr, self.max_fin_rad)
        self.aileron_cmd_rad = self._clamp(da, self.max_fin_rad)
        return True

    def terminate(self):
        print("Terminating autopilot_fmu.")
        self.time = 0.0

    # ------------------------------------------------------------------- laws
    def _pid(self, alpha, alpha_cmd, beta, beta_cmd, dt):
        e_p = alpha_cmd - alpha
        e_y = beta_cmd - beta
        self._int_pitch = np.clip(self._int_pitch + e_p * dt, -0.5, 0.5)
        self._int_yaw = np.clip(self._int_yaw + e_y * dt, -0.5, 0.5)
        de_ff = -self.Cm_alpha * alpha_cmd / self.Cm_de
        dr_ff = -self.Cn_beta * beta_cmd / self.Cn_dr
        de = de_ff + self.kp_pitch * e_p + self.ki_pitch * self._int_pitch - self.kd_pitch * self.nav_q
        dr = dr_ff + self.kp_yaw * e_y + self.ki_yaw * self._int_yaw - self.kd_yaw * self.nav_r
        return de, dr

    def _lqr_pitch(self, alpha, alpha_cmd, q, qbar, speed):
        S, d = self.ref_area_m2, self.ref_diameter_m
        m = max(self.mass_kg, 1e-3)
        Iyy = max(self.Iyy, 1e-6)
        Z_alpha = -qbar * S * self.CN_alpha / (m * speed)
        Z_de = -qbar * S * self.CN_de / (m * speed)
        M_alpha = qbar * S * d * self.Cm_alpha / Iyy
        M_q = qbar * S * d * d * self.Cm_q / (2.0 * speed * Iyy)
        M_de = qbar * S * d * self.Cm_de / Iyy
        A = np.array([[Z_alpha, 1.0], [M_alpha, M_q]])
        B = np.array([[Z_de], [M_de]])
        K = self._lqr_gain(A, B)
        de_ff = -self.Cm_alpha * alpha_cmd / self.Cm_de
        x = np.array([alpha - alpha_cmd, q])
        return float(de_ff - K @ x)

    def _lqr_yaw(self, beta, beta_cmd, r, qbar, speed):
        S, d = self.ref_area_m2, self.ref_diameter_m
        m = max(self.mass_kg, 1e-3)
        Izz = max(self.Izz, 1e-6)
        Y_beta = qbar * S * self.CY_beta / (m * speed)
        Y_dr = qbar * S * self.CY_dr / (m * speed)
        N_beta = qbar * S * d * self.Cn_beta / Izz
        N_r = qbar * S * d * d * self.Cn_r / (2.0 * speed * Izz)
        N_dr = qbar * S * d * self.Cn_dr / Izz
        # State [beta, r]; beta_dot = Y_beta beta - r + Y_dr dr
        A = np.array([[Y_beta, -1.0], [N_beta, N_r]])
        B = np.array([[Y_dr], [N_dr]])
        K = self._lqr_gain(A, B)
        dr_ff = -self.Cn_beta * beta_cmd / self.Cn_dr
        x = np.array([beta - beta_cmd, r])
        return float(dr_ff - K @ x)

    def _lqr_gain(self, A, B):
        Q = np.diag([self.lqr_q_angle, self.lqr_q_rate])
        R = np.array([[self.lqr_r_fin]])
        try:
            P = solve_continuous_are(A, B, Q, R)
            return (np.linalg.inv(R) @ B.T @ P).flatten()
        except Exception:
            return np.array([1.0, 0.1])

    def _clamp(self, x, lim):
        return float(max(-lim, min(lim, x)))
