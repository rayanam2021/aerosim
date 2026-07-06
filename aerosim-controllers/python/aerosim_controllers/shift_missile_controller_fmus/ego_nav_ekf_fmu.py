"""
Ego navigation EKF FMU — 16-state quaternion INS (error-state / MEKF).

Full 6-DOF strapdown inertial navigation for the ego interceptor.  The nominal
state carries 16 quantities and the covariance is propagated on the 15 error
states (multiplicative attitude error), which is the numerically correct way to
run a quaternion INS:

    nominal (16): p(3) v(3) q(4) accel_bias(3) gyro_bias(3)
    error   (15): dp(3) dv(3) dtheta(3) dba(3) dbg(3)

Prediction (IMU strapdown, every step the IMU flag is set):
    omega = gyro - bg ;  f = accel - ba
    q_dot = 0.5 * q (x) omega            (quaternion kinematics)
    v_dot = R(q) f - g                   (matches the imu_fmu specific-force sign)
    p_dot = v
    error dynamics (local/body error):
        d(dp)/dt = dv
        d(dv)/dt = -R[f x] dtheta - R dba
        d(dtheta)/dt = -[omega x] dtheta - dbg
        biases: random walk

Measurement updates (mismatched rates via each sensor's ``*_ready`` flag):
    GNSS -> position + velocity        (linear in dp, dv)
    Baro -> down-position pD           (linear in dp[2])

After every batch of updates the error state is injected into the nominal state
(quaternion via q <- q (x) Exp(dtheta)) and reset to zero.

Inputs  (aux): IMU accel/gyro + imu_ready; GNSS pos/vel + gnss_ready;
               baro_alt + baro_ready
Outputs (aux): nav_pos_{n,e,d}, nav_vel_{n,e,d}, nav_q{w,x,y,z},
               nav_p/q/r, nav_ax/ay/az (specific force, body), nav_valid;
               also nav_vehicle_state (VehicleState)
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var
from aerosim_data import dict_to_namespace
from aerosim_data import types as aerosim_types

GRAVITY = 9.80665


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class ego_nav_ekf_fmu(Fmi3Slave):
    """16-state quaternion INS error-state EKF."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Ego 16-state quaternion INS (error-state EKF)"

        # IMU inputs.
        self.accel_x_mps2 = 0.0
        self.accel_y_mps2 = 0.0
        self.accel_z_mps2 = 0.0
        self.gyro_x_rps = 0.0
        self.gyro_y_rps = 0.0
        self.gyro_z_rps = 0.0
        self.imu_measurement_ready = 0.0
        # GNSS inputs.
        self.pos_n_m = 0.0
        self.pos_e_m = 0.0
        self.pos_d_m = 0.0
        self.vel_n_mps = 0.0
        self.vel_e_mps = 0.0
        self.vel_d_mps = 0.0
        self.gnss_measurement_ready = 0.0
        # Baro inputs.
        self.baro_alt_m = 0.0
        self.baro_measurement_ready = 0.0
        for _n in (
            "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
            "gyro_x_rps", "gyro_y_rps", "gyro_z_rps", "imu_measurement_ready",
            "pos_n_m", "pos_e_m", "pos_d_m",
            "vel_n_mps", "vel_e_mps", "vel_d_mps", "gnss_measurement_ready",
            "baro_alt_m", "baro_measurement_ready",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # Outputs.
        self.nav_pos_n = 0.0
        self.nav_pos_e = 0.0
        self.nav_pos_d = 0.0
        self.nav_vel_n = 0.0
        self.nav_vel_e = 0.0
        self.nav_vel_d = 0.0
        self.nav_qw = 1.0
        self.nav_qx = 0.0
        self.nav_qy = 0.0
        self.nav_qz = 0.0
        self.nav_p = 0.0
        self.nav_q = 0.0
        self.nav_r = 0.0
        self.nav_ax = 0.0
        self.nav_ay = 0.0
        self.nav_az = 0.0
        self.nav_valid = 0.0
        self.nav_vehicle_state = dict_to_namespace(aerosim_types.VehicleState().to_dict())
        for _n in (
            "nav_pos_n", "nav_pos_e", "nav_pos_d",
            "nav_vel_n", "nav_vel_e", "nav_vel_d",
            "nav_qw", "nav_qx", "nav_qy", "nav_qz",
            "nav_p", "nav_q", "nav_r", "nav_ax", "nav_ay", "nav_az", "nav_valid",
        ):
            register_fmu3_var(self, _n, causality="output")
        register_fmu3_var(self, "nav_vehicle_state", causality="output")

        # Parameters.
        self.world_origin_altitude = 0.0
        register_fmu3_param(self, "world_origin_altitude")
        self.gnss_pos_std_m = 4.0
        register_fmu3_param(self, "gnss_pos_std_m")
        self.gnss_vel_std_mps = 0.15
        register_fmu3_param(self, "gnss_vel_std_mps")
        self.baro_alt_std_m = 3.0
        register_fmu3_param(self, "baro_alt_std_m")
        self.accel_noise_std = 0.05
        register_fmu3_param(self, "accel_noise_std")
        self.gyro_noise_std = 0.002
        register_fmu3_param(self, "gyro_noise_std")
        self.accel_bias_rw = 1e-3
        register_fmu3_param(self, "accel_bias_rw")
        self.gyro_bias_rw = 1e-5
        register_fmu3_param(self, "gyro_bias_rw")
        self.init_pos_std_m = 50.0
        register_fmu3_param(self, "init_pos_std_m")
        self.init_vel_std_mps = 5.0
        register_fmu3_param(self, "init_vel_std_mps")
        self.init_att_std_rad = 0.1
        register_fmu3_param(self, "init_att_std_rad")

        # Nominal state.
        self._p = np.zeros(3)
        self._v = np.zeros(3)
        self._q = np.array([0.0, 0.0, 0.0, 1.0])  # scalar-last
        self._ba = np.zeros(3)
        self._bg = np.zeros(3)
        self._P = np.eye(15)
        self._got_gnss = False
        self._last_omega = np.zeros(3)
        self._last_f = np.zeros(3)

    def enter_initialization_mode(self):
        self._p = np.zeros(3)
        self._v = np.zeros(3)
        self._q = np.array([0.0, 0.0, 0.0, 1.0])
        self._ba = np.zeros(3)
        self._bg = np.zeros(3)
        p2 = self.init_pos_std_m ** 2
        v2 = self.init_vel_std_mps ** 2
        a2 = self.init_att_std_rad ** 2
        self._P = np.diag([
            p2, p2, p2, v2, v2, v2, a2, a2, a2,
            0.25, 0.25, 0.25, 1e-4, 1e-4, 1e-4,
        ])
        self._got_gnss = False
        self.nav_valid = 0.0

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size

        if self.imu_measurement_ready > 0.5:
            f = np.array([self.accel_x_mps2, self.accel_y_mps2, self.accel_z_mps2]) - self._ba
            omega = np.array([self.gyro_x_rps, self.gyro_y_rps, self.gyro_z_rps]) - self._bg
            self._last_f = f
            self._last_omega = omega
            self._propagate(f, omega, step_size)

        if self.gnss_measurement_ready > 0.5:
            self._update_gnss()
        if self.baro_measurement_ready > 0.5:
            self._update_baro()

        self._write_outputs()
        return True

    def terminate(self):
        print("Terminating ego_nav_ekf_fmu.")
        self.time = 0.0

    # -------------------------------------------------------------- EKF core
    def _propagate(self, f, omega, dt):
        rot = Rotation.from_quat(self._q)
        R = rot.as_matrix()
        a_ned = R @ f - np.array([0.0, 0.0, GRAVITY])

        # Nominal integration.
        self._p = self._p + self._v * dt + 0.5 * a_ned * dt * dt
        self._v = self._v + a_ned * dt
        dq = Rotation.from_rotvec(omega * dt)
        self._q = (rot * dq).as_quat()
        self._q /= max(np.linalg.norm(self._q), 1e-12)

        # Error-state transition (continuous -> discrete first order).
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ _skew(f) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = np.eye(3) - _skew(omega) * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        q_an = (self.accel_noise_std ** 2) * dt * dt
        q_gn = (self.gyro_noise_std ** 2) * dt * dt
        q_ab = (self.accel_bias_rw ** 2) * dt
        q_gb = (self.gyro_bias_rw ** 2) * dt
        Q = np.diag([
            0.0, 0.0, 0.0, q_an, q_an, q_an, q_gn, q_gn, q_gn,
            q_ab, q_ab, q_ab, q_gb, q_gb, q_gb,
        ])
        self._P = F @ self._P @ F.T + Q

    def _update_gnss(self):
        z = np.array([
            self.pos_n_m, self.pos_e_m, self.pos_d_m,
            self.vel_n_mps, self.vel_e_mps, self.vel_d_mps,
        ])
        if not self._got_gnss:
            self._p = z[:3].copy()
            self._v = z[3:].copy()
            self._got_gnss = True
            self.nav_valid = 1.0
            return
        h = np.concatenate([self._p, self._v])
        H = np.zeros((6, 15))
        H[0:6, 0:6] = np.eye(6)
        R = np.diag([
            self.gnss_pos_std_m ** 2, self.gnss_pos_std_m ** 2,
            (self.gnss_pos_std_m * 1.5) ** 2,
            self.gnss_vel_std_mps ** 2, self.gnss_vel_std_mps ** 2,
            self.gnss_vel_std_mps ** 2,
        ])
        self._kalman(z - h, H, R)

    def _update_baro(self):
        pD = self.world_origin_altitude - self.baro_alt_m
        H = np.zeros((1, 15))
        H[0, 2] = 1.0
        R = np.array([[self.baro_alt_std_m ** 2]])
        self._kalman(np.array([pD - self._p[2]]), H, R)

    def _kalman(self, innov, H, R):
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        dx = K @ innov
        I_KH = np.eye(15) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
        self._inject(dx)

    def _inject(self, dx):
        self._p = self._p + dx[0:3]
        self._v = self._v + dx[3:6]
        dtheta = dx[6:9]
        rot = Rotation.from_quat(self._q) * Rotation.from_rotvec(dtheta)
        self._q = rot.as_quat()
        self._q /= max(np.linalg.norm(self._q), 1e-12)
        self._ba = self._ba + dx[9:12]
        self._bg = self._bg + dx[12:15]

    # ---------------------------------------------------------------- output
    def _write_outputs(self):
        self.nav_pos_n, self.nav_pos_e, self.nav_pos_d = (float(v) for v in self._p)
        self.nav_vel_n, self.nav_vel_e, self.nav_vel_d = (float(v) for v in self._v)
        qx, qy, qz, qw = self._q
        self.nav_qw, self.nav_qx, self.nav_qy, self.nav_qz = (
            float(qw), float(qx), float(qy), float(qz)
        )
        self.nav_p, self.nav_q, self.nav_r = (float(v) for v in self._last_omega)
        self.nav_ax, self.nav_ay, self.nav_az = (float(v) for v in self._last_f)

        self.nav_vehicle_state.state.pose.position.x = float(self._p[0])
        self.nav_vehicle_state.state.pose.position.y = float(self._p[1])
        self.nav_vehicle_state.state.pose.position.z = float(self._p[2])
        self.nav_vehicle_state.state.pose.orientation.w = float(qw)
        self.nav_vehicle_state.state.pose.orientation.x = float(qx)
        self.nav_vehicle_state.state.pose.orientation.y = float(qy)
        self.nav_vehicle_state.state.pose.orientation.z = float(qz)
        self.nav_vehicle_state.velocity.x = float(self._v[0])
        self.nav_vehicle_state.velocity.y = float(self._v[1])
        self.nav_vehicle_state.velocity.z = float(self._v[2])
        self.nav_vehicle_state.angular_velocity.x = float(self._last_omega[0])
        self.nav_vehicle_state.angular_velocity.y = float(self._last_omega[1])
        self.nav_vehicle_state.angular_velocity.z = float(self._last_omega[2])
