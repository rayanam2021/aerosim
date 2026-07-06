"""
Guidance FMU for the SHIFT interceptor (outer loop).

Consumes the ego and threat state *estimates* (from the navigation EKFs, not
ground truth) and produces a commanded acceleration vector in NED that the
inner-loop autopilot then tracks.  Two interchangeable guidance laws are
selectable via the ``guidance_law`` parameter:

    "propnav"  True (vector) Proportional Navigation:
                   a_cmd = N * Vc * (omega_los x LOS_hat)
               classic, robust, requires only LOS-rate + closing speed.

    "mpc"      Receding-horizon optimal / predictive guidance implemented as
               augmented zero-effort-miss (ZEM):
                   ZEM = r + v*t_go + 0.5*a_tgt*t_go^2
                   a_cmd = N' * ZEM / t_go^2
               This is the analytic solution of the finite-horizon quadratic
               miss+effort optimal-control problem (i.e. the closed form an MPC
               of the linearised intercept model converges to), and it
               feed-forwards the estimated target acceleration.

Inputs  (aux): nav_pos_{n,e,d}, nav_vel_{n,e,d}, nav_valid  (ego EKF)
               tgt_pos_{n,e,d}, tgt_vel_{n,e,d}, tgt_acc_{n,e,d}, tgt_valid
Outputs (aux): a_cmd_{n,e,d}, range_m, closing_speed_mps, t_go_s,
               los_rate_rps, zem_m, guidance_active
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var


class guidance_fmu(Fmi3Slave):
    """PropNav / optimal-predictive intercept guidance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT interceptor guidance (PropNav / MPC-ZEM)"

        self.nav_pos_n = 0.0
        self.nav_pos_e = 0.0
        self.nav_pos_d = 0.0
        self.nav_vel_n = 0.0
        self.nav_vel_e = 0.0
        self.nav_vel_d = 0.0
        self.nav_valid = 0.0
        self.tgt_pos_n = 0.0
        self.tgt_pos_e = 0.0
        self.tgt_pos_d = 0.0
        self.tgt_vel_n = 0.0
        self.tgt_vel_e = 0.0
        self.tgt_vel_d = 0.0
        self.tgt_acc_n = 0.0
        self.tgt_acc_e = 0.0
        self.tgt_acc_d = 0.0
        self.tgt_valid = 0.0
        for _n in (
            "nav_pos_n", "nav_pos_e", "nav_pos_d",
            "nav_vel_n", "nav_vel_e", "nav_vel_d", "nav_valid",
            "tgt_pos_n", "tgt_pos_e", "tgt_pos_d",
            "tgt_vel_n", "tgt_vel_e", "tgt_vel_d",
            "tgt_acc_n", "tgt_acc_e", "tgt_acc_d", "tgt_valid",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.a_cmd_n = 0.0
        self.a_cmd_e = 0.0
        self.a_cmd_d = 0.0
        self.range_m = 0.0
        self.closing_speed_mps = 0.0
        self.t_go_s = 0.0
        self.los_rate_rps = 0.0
        self.zem_m = 0.0
        self.guidance_active = 0.0
        for _n in (
            "a_cmd_n", "a_cmd_e", "a_cmd_d", "range_m", "closing_speed_mps",
            "t_go_s", "los_rate_rps", "zem_m", "guidance_active",
        ):
            register_fmu3_var(self, _n, causality="output")

        # "propnav" or "mpc"
        self.guidance_law = "propnav"
        register_fmu3_param(self, "guidance_law")
        self.nav_gain = 4.0
        register_fmu3_param(self, "nav_gain")
        self.mpc_gain = 3.0
        register_fmu3_param(self, "mpc_gain")
        self.max_accel_g = 40.0
        register_fmu3_param(self, "max_accel_g")
        self.min_closing_speed_mps = 5.0
        register_fmu3_param(self, "min_closing_speed_mps")

    def enter_initialization_mode(self):
        pass

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size

        if self.nav_valid < 0.5 or self.tgt_valid < 0.5:
            self._publish(np.zeros(3), 0.0, 0.0, 0.0, 0.0, 0.0, active=False)
            return True

        p_ego = np.array([self.nav_pos_n, self.nav_pos_e, self.nav_pos_d])
        p_tgt = np.array([self.tgt_pos_n, self.tgt_pos_e, self.tgt_pos_d])
        v_ego = np.array([self.nav_vel_n, self.nav_vel_e, self.nav_vel_d])
        v_tgt = np.array([self.tgt_vel_n, self.tgt_vel_e, self.tgt_vel_d])
        a_tgt = np.array([self.tgt_acc_n, self.tgt_acc_e, self.tgt_acc_d])

        r = p_tgt - p_ego
        v = v_tgt - v_ego
        rng = float(np.linalg.norm(r))
        if rng < 1e-3:
            self._publish(np.zeros(3), rng, 0.0, 0.0, 0.0, 0.0, active=False)
            return True

        r_hat = r / rng
        closing = -float(np.dot(r, v)) / rng          # >0 when closing
        omega_los = np.cross(r, v) / (rng * rng)       # LOS angular-rate vector
        los_rate = float(np.linalg.norm(omega_los))
        vc = max(closing, self.min_closing_speed_mps)
        t_go = rng / vc

        law = str(self.guidance_law).strip().lower()
        if law == "mpc":
            zem_vec = r + v * t_go + 0.5 * a_tgt * t_go * t_go
            zem_perp = zem_vec - np.dot(zem_vec, r_hat) * r_hat
            a_cmd = self.mpc_gain * zem_perp / max(t_go * t_go, 1e-3)
            zem = float(np.linalg.norm(zem_perp))
        else:  # propnav
            a_cmd = self.nav_gain * vc * np.cross(omega_los, r_hat)
            zem = float(np.linalg.norm(r + v * t_go - np.dot(r + v * t_go, r_hat) * r_hat))

        a_max = self.max_accel_g * 9.80665
        norm = float(np.linalg.norm(a_cmd))
        if norm > a_max and norm > 1e-9:
            a_cmd = a_cmd * (a_max / norm)

        self._publish(a_cmd, rng, closing, t_go, los_rate, zem, active=True)
        return True

    def terminate(self):
        print("Terminating guidance_fmu.")
        self.time = 0.0

    def _publish(self, a_cmd, rng, closing, t_go, los_rate, zem, active):
        self.a_cmd_n, self.a_cmd_e, self.a_cmd_d = (float(v) for v in a_cmd)
        self.range_m = float(rng)
        self.closing_speed_mps = float(closing)
        self.t_go_s = float(t_go)
        self.los_rate_rps = float(los_rate)
        self.zem_m = float(zem)
        self.guidance_active = 1.0 if active else 0.0
