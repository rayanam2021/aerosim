"""
IMU (Inertial Measurement Unit) sensor FMU for the missile stack.

Converts the perfect plant ``VehicleState`` kinematics into a noisy, biased
sensor reading that mimics a tactical-grade MEMS IMU.  The model includes:

Accelerometer
  - Additive white Gaussian noise  (``accel_noise_std_mps2``, m/s²/√Hz × √Hz)
  - Slow-varying bias (random-walk initialised at startup, ``accel_bias_std_mps2``)
  - Scale-factor error (``accel_scale_error_ppm``)

Gyroscope
  - Angle random walk noise (``gyro_noise_std_rps``, rad/s per sample)
  - Slow-varying bias (``gyro_bias_std_rps``)
  - Scale-factor error (``gyro_scale_error_ppm``)

Update rate
  The sensor steps every simulation tick but only produces a *new* measurement
  at ``update_rate_hz`` (default 100 Hz).  Between updates the previous
  measurement is held and ``measurement_ready`` is 0.  The nav filter uses this
  flag to distinguish a fresh sample from a stale one so it can apply the
  correct IMU propagation rate.

Gravity removal
  The accelerometer measures specific force (kinematic accel + gravity reaction).
  We provide both the raw specific-force output (what the sensor actually sees)
  AND the true kinematic acceleration from the plant for reference/testing.

Inputs  (component topic): vehicle_state  (aerosim::types::VehicleState)
Outputs (aux topic):       accel_x/y/z_mps2, gyro_x/y/z_rps,
                           measurement_ready, imu_time_s
"""

from __future__ import annotations

import math

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var
from aerosim_data import dict_to_namespace
from aerosim_data import types as aerosim_types

_GRAVITY = 9.80665  # m/s²


class imu_fmu(Fmi3Slave):
    """Noisy IMU sensor with configurable rate and error model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Missile IMU sensor (noise + bias + configurable rate)"

        self.vehicle_state = dict_to_namespace(aerosim_types.VehicleState().to_dict())
        register_fmu3_var(self, "vehicle_state", causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # Outputs – specific force (= accel + g, what the IMU actually reads).
        self.accel_x_mps2 = 0.0
        self.accel_y_mps2 = 0.0
        self.accel_z_mps2 = 0.0
        self.gyro_x_rps = 0.0
        self.gyro_y_rps = 0.0
        self.gyro_z_rps = 0.0
        self.measurement_ready = 0.0  # 1.0 when a fresh sample is available
        self.imu_time_s = 0.0
        for _n in (
            "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
            "gyro_x_rps", "gyro_y_rps", "gyro_z_rps",
            "measurement_ready", "imu_time_s",
        ):
            register_fmu3_var(self, _n, causality="output")

        # --- Error-model parameters ------------------------------------------
        self.update_rate_hz = 100.0
        register_fmu3_param(self, "update_rate_hz")

        # Accelerometer noise density (m/s²/√Hz) – converted to per-sample sigma
        # inside the FMU: σ = noise_density × √(update_rate_hz).
        self.accel_noise_density_mps2_per_sqrthz = 3e-3
        register_fmu3_param(self, "accel_noise_density_mps2_per_sqrthz")
        self.accel_bias_std_mps2 = 0.05
        register_fmu3_param(self, "accel_bias_std_mps2")
        self.accel_scale_error_ppm = 500.0
        register_fmu3_param(self, "accel_scale_error_ppm")

        # Gyroscope angle random walk (rad/s/√Hz).
        self.gyro_noise_density_rps_per_sqrthz = 1e-4
        register_fmu3_param(self, "gyro_noise_density_rps_per_sqrthz")
        self.gyro_bias_std_rps = 5e-4
        register_fmu3_param(self, "gyro_bias_std_rps")
        self.gyro_scale_error_ppm = 300.0
        register_fmu3_param(self, "gyro_scale_error_ppm")

        self.rng_seed = 42
        register_fmu3_param(self, "rng_seed")

        # --- Internal state ---------------------------------------------------
        self._rng = np.random.default_rng(42)
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._elapsed_since_update = 0.0
        self._update_interval = 0.01
        self._prev_velocity_ned = np.zeros(3)
        self._prev_rot = Rotation.identity()

    def enter_initialization_mode(self):
        self._rng = np.random.default_rng(int(self.rng_seed))
        hz = max(self.update_rate_hz, 1.0)
        self._update_interval = 1.0 / hz
        # Initialise fixed biases sampled at startup (changes slowly in reality).
        self._accel_bias = self._rng.normal(0.0, self.accel_bias_std_mps2, 3)
        self._gyro_bias = self._rng.normal(0.0, self.gyro_bias_std_rps, 3)
        self._elapsed_since_update = self._update_interval  # force first step
        self.measurement_ready = 0.0
        vel = self.vehicle_state.velocity
        self._prev_velocity_ned = np.array([vel.x, vel.y, vel.z])
        q = self.vehicle_state.state.pose.orientation
        if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
            self._prev_rot = Rotation.identity()
        else:
            self._prev_rot = Rotation.from_quat([q.x, q.y, q.z, q.w])

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self._elapsed_since_update += step_size

        if self._elapsed_since_update < self._update_interval - 1e-9:
            self.measurement_ready = 0.0
            return True

        dt = self._elapsed_since_update
        self._elapsed_since_update = 0.0
        self._produce_measurement(dt, current_time + step_size)
        return True

    def terminate(self):
        print("Terminating imu_fmu.")
        self.time = 0.0

    def _produce_measurement(self, dt: float, sim_time: float) -> None:
        """Compute noisy specific-force and angular-rate samples."""
        hz = max(self.update_rate_hz, 1.0)
        accel_sigma = (
            self.accel_noise_density_mps2_per_sqrthz * math.sqrt(hz)
        )
        gyro_sigma = (
            self.gyro_noise_density_rps_per_sqrthz * math.sqrt(hz)
        )
        scale_a = 1.0 + self.accel_scale_error_ppm * 1e-6
        scale_g = 1.0 + self.gyro_scale_error_ppm * 1e-6

        # True kinematic acceleration from differentiated NED velocity.
        vel = self.vehicle_state.velocity
        vel_ned = np.array([vel.x, vel.y, vel.z])
        true_accel_ned = (vel_ned - self._prev_velocity_ned) / max(dt, 1e-6)
        self._prev_velocity_ned = vel_ned

        # Rotate gravity and true accel into body frame to get specific force.
        q = self.vehicle_state.state.pose.orientation
        if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
            rot = Rotation.identity()
        else:
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        gravity_ned = np.array([0.0, 0.0, _GRAVITY])
        specific_force_body = rot.inv().apply(true_accel_ned + gravity_ned)

        # Apply scale error, bias, and white noise.
        noisy_accel = (
            specific_force_body * scale_a
            + self._accel_bias
            + self._rng.normal(0.0, accel_sigma, 3)
        )

        # True angular velocity from attitude differentiation.
        delta_rot = rot * self._prev_rot.inv()
        rotvec = delta_rot.as_rotvec()
        true_omega_body = rotvec / max(dt, 1e-6)
        self._prev_rot = rot

        noisy_gyro = (
            true_omega_body * scale_g
            + self._gyro_bias
            + self._rng.normal(0.0, gyro_sigma, 3)
        )

        self.accel_x_mps2 = float(noisy_accel[0])
        self.accel_y_mps2 = float(noisy_accel[1])
        self.accel_z_mps2 = float(noisy_accel[2])
        self.gyro_x_rps = float(noisy_gyro[0])
        self.gyro_y_rps = float(noisy_gyro[1])
        self.gyro_z_rps = float(noisy_gyro[2])
        self.measurement_ready = 1.0
        self.imu_time_s = sim_time
