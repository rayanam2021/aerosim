"""
Target navigation EKF FMU — 9-state kinematic tracker.

Estimates the threat missile's NED trajectory from the ego seekers.  Because the
seekers give line-of-sight geometry (and, for radar, range/range-rate) the
tracker uses a nonlinear measurement model referenced to the ego navigation
solution, with a constant-acceleration (nearly-constant-accel) process model:

    state x = [pN pE pD  vN vE vD  aN aE aD]        (threat, NED)
    process: p_dot=v, v_dot=a, a_dot=white  (Singer-like accel random walk)

    relative geometry  r = p_tgt - p_ego :
        radar -> z = [range, azimuth, elevation, range_rate]   (full 3-D fix)
        IR    -> z = [azimuth, elevation]                      (bearing-only)

Measurement Jacobians are formed numerically (robust for the mixed
range/angle observation), and each sensor is fused only when its ``*_ready``
flag is set, naturally handling the different radar and IR update rates.

Inputs  (aux): ego_pos_{n,e,d}, ego_vel_{n,e,d}  (ego nav estimate)
               radar_range_m, radar_az_rad, radar_el_rad,
               radar_range_rate_mps, radar_ready
               ir_az_rad, ir_el_rad, ir_ready
Outputs (aux): tgt_pos_{n,e,d}, tgt_vel_{n,e,d}, tgt_acc_{n,e,d},
               tgt_valid, est_range_m
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class target_nav_ekf_fmu(Fmi3Slave):
    """9-state constant-acceleration threat tracker (radar + IR)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Target 9-state kinematic EKF (radar + IR seeker fusion)"

        self.ego_pos_n = 0.0
        self.ego_pos_e = 0.0
        self.ego_pos_d = 0.0
        self.ego_vel_n = 0.0
        self.ego_vel_e = 0.0
        self.ego_vel_d = 0.0
        self.radar_range_m = 0.0
        self.radar_az_rad = 0.0
        self.radar_el_rad = 0.0
        self.radar_range_rate_mps = 0.0
        self.radar_ready = 0.0
        self.ir_az_rad = 0.0
        self.ir_el_rad = 0.0
        self.ir_ready = 0.0
        for _n in (
            "ego_pos_n", "ego_pos_e", "ego_pos_d",
            "ego_vel_n", "ego_vel_e", "ego_vel_d",
            "radar_range_m", "radar_az_rad", "radar_el_rad",
            "radar_range_rate_mps", "radar_ready",
            "ir_az_rad", "ir_el_rad", "ir_ready",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

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
        self.est_range_m = 0.0
        for _n in (
            "tgt_pos_n", "tgt_pos_e", "tgt_pos_d",
            "tgt_vel_n", "tgt_vel_e", "tgt_vel_d",
            "tgt_acc_n", "tgt_acc_e", "tgt_acc_d",
            "tgt_valid", "est_range_m",
        ):
            register_fmu3_var(self, _n, causality="output")

        self.accel_process_std = 30.0
        register_fmu3_param(self, "accel_process_std")
        self.radar_range_std_m = 15.0
        register_fmu3_param(self, "radar_range_std_m")
        self.radar_angle_std_rad = 0.005
        register_fmu3_param(self, "radar_angle_std_rad")
        self.radar_rate_std_mps = 3.0
        register_fmu3_param(self, "radar_rate_std_mps")
        self.ir_angle_std_rad = 0.001
        register_fmu3_param(self, "ir_angle_std_rad")
        self.init_pos_std_m = 100.0
        register_fmu3_param(self, "init_pos_std_m")
        self.init_vel_std_mps = 150.0
        register_fmu3_param(self, "init_vel_std_mps")
        self.init_acc_std_mps2 = 100.0
        register_fmu3_param(self, "init_acc_std_mps2")

        self._x = np.zeros(9)
        self._P = np.eye(9)
        self._locked = False

    def enter_initialization_mode(self):
        self._x = np.zeros(9)
        self._P = np.eye(9)
        self._locked = False
        self.tgt_valid = 0.0

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        ego_p = np.array([self.ego_pos_n, self.ego_pos_e, self.ego_pos_d])
        ego_v = np.array([self.ego_vel_n, self.ego_vel_e, self.ego_vel_d])

        if not self._locked:
            if self.radar_ready > 0.5 and self.radar_range_m > 1.0:
                self._initialize(ego_p)
            self._write_outputs(ego_p)
            return True

        self._propagate(step_size)
        if self.radar_ready > 0.5:
            self._update_radar(ego_p, ego_v)
        if self.ir_ready > 0.5:
            self._update_ir(ego_p)
        self._write_outputs(ego_p)
        return True

    def terminate(self):
        print("Terminating target_nav_ekf_fmu.")
        self.time = 0.0

    # -------------------------------------------------------------- EKF core
    def _initialize(self, ego_p):
        az, el, rng = self.radar_az_rad, self.radar_el_rad, self.radar_range_m
        los = np.array([
            np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)
        ])
        self._x = np.zeros(9)
        self._x[:3] = ego_p + rng * los
        self._P = np.diag([
            self.init_pos_std_m ** 2, self.init_pos_std_m ** 2, self.init_pos_std_m ** 2,
            self.init_vel_std_mps ** 2, self.init_vel_std_mps ** 2, self.init_vel_std_mps ** 2,
            self.init_acc_std_mps2 ** 2, self.init_acc_std_mps2 ** 2, self.init_acc_std_mps2 ** 2,
        ])
        self._locked = True
        self.tgt_valid = 1.0

    def _propagate(self, dt):
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        F[3:6, 6:9] = np.eye(3) * dt
        self._x = F @ self._x
        q = (self.accel_process_std ** 2) * dt
        Q = np.zeros((9, 9))
        Q[6:9, 6:9] = np.eye(3) * q
        Q[3:6, 3:6] = np.eye(3) * q * dt
        self._P = F @ self._P @ F.T + Q

    def _h_radar(self, x, ego_p, ego_v):
        r = x[:3] - ego_p
        v = x[3:6] - ego_v
        rng = max(float(np.linalg.norm(r)), 1e-3)
        ground = max(float(np.hypot(r[0], r[1])), 1e-6)
        az = np.arctan2(r[1], r[0])
        el = np.arctan2(-r[2], ground)
        rr = float(np.dot(r, v)) / rng
        return np.array([rng, az, el, rr])

    def _h_ir(self, x, ego_p):
        r = x[:3] - ego_p
        ground = max(float(np.hypot(r[0], r[1])), 1e-6)
        az = np.arctan2(r[1], r[0])
        el = np.arctan2(-r[2], ground)
        return np.array([az, el])

    def _numeric_H(self, hfun, dim, *args):
        H = np.zeros((dim, 9))
        h0 = hfun(self._x, *args)
        eps = 1e-3
        for i in range(9):
            xp = self._x.copy()
            xp[i] += eps
            H[:, i] = (hfun(xp, *args) - h0) / eps
        return H, h0

    def _update_radar(self, ego_p, ego_v):
        z = np.array([
            self.radar_range_m, self.radar_az_rad,
            self.radar_el_rad, self.radar_range_rate_mps,
        ])
        H, h0 = self._numeric_H(self._h_radar, 4, ego_p, ego_v)
        innov = z - h0
        innov[1] = _wrap(innov[1])
        innov[2] = _wrap(innov[2])
        R = np.diag([
            self.radar_range_std_m ** 2, self.radar_angle_std_rad ** 2,
            self.radar_angle_std_rad ** 2, self.radar_rate_std_mps ** 2,
        ])
        self._kalman(innov, H, R)

    def _update_ir(self, ego_p):
        z = np.array([self.ir_az_rad, self.ir_el_rad])
        H, h0 = self._numeric_H(self._h_ir, 2, ego_p)
        innov = _wrap(z - h0)
        R = np.diag([self.ir_angle_std_rad ** 2, self.ir_angle_std_rad ** 2])
        self._kalman(innov, H, R)

    def _kalman(self, innov, H, R):
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        I_KH = np.eye(9) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

    def _write_outputs(self, ego_p):
        self.tgt_pos_n, self.tgt_pos_e, self.tgt_pos_d = (float(v) for v in self._x[0:3])
        self.tgt_vel_n, self.tgt_vel_e, self.tgt_vel_d = (float(v) for v in self._x[3:6])
        self.tgt_acc_n, self.tgt_acc_e, self.tgt_acc_d = (float(v) for v in self._x[6:9])
        self.est_range_m = float(np.linalg.norm(self._x[:3] - ego_p))
