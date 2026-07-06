"""
GNSS (GPS) sensor FMU for the missile stack.

Converts the perfect plant ``VehicleState`` NED position into a noisy,
low-update-rate GPS fix.  Models include:

Position accuracy
  Gaussian noise on each NED axis (horizontal σ typically 2–5 m CEP for a
  tactical GPS, vertical σ ≈ 1.5× horizontal).

Velocity accuracy
  Doppler-derived NED velocity with Gaussian noise (~0.1 m/s typical).

Update rate
  Default 10 Hz (real GPS).  Between updates ``measurement_ready`` is 0 so
  the nav filter knows to skip the position/velocity measurement update step.

Dropout
  An optional ``dropout_probability`` per update interval lets you simulate
  GPS signal blockage (e.g., under high-G manoeuvres or in valleys).

Inputs  (component topic): vehicle_state  (aerosim::types::VehicleState)
Outputs (aux topic):       pos_n_m, pos_e_m, pos_d_m,
                           vel_n_mps, vel_e_mps, vel_d_mps,
                           measurement_ready, gnss_time_s
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var
from aerosim_data import dict_to_namespace
from aerosim_data import types as aerosim_types


class gnss_fmu(Fmi3Slave):
    """Noisy GNSS/GPS sensor with configurable rate and position noise."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Missile GNSS/GPS sensor (noise + dropout + configurable rate)"

        self.vehicle_state = dict_to_namespace(aerosim_types.VehicleState().to_dict())
        register_fmu3_var(self, "vehicle_state", causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # Outputs – noisy NED position and velocity.
        self.pos_n_m = 0.0
        self.pos_e_m = 0.0
        self.pos_d_m = 0.0
        self.vel_n_mps = 0.0
        self.vel_e_mps = 0.0
        self.vel_d_mps = 0.0
        self.measurement_ready = 0.0
        self.gnss_time_s = 0.0
        for _n in (
            "pos_n_m", "pos_e_m", "pos_d_m",
            "vel_n_mps", "vel_e_mps", "vel_d_mps",
            "measurement_ready", "gnss_time_s",
        ):
            register_fmu3_var(self, _n, causality="output")

        # --- Error-model parameters ------------------------------------------
        self.update_rate_hz = 10.0
        register_fmu3_param(self, "update_rate_hz")

        self.pos_noise_horizontal_std_m = 3.0
        register_fmu3_param(self, "pos_noise_horizontal_std_m")
        self.pos_noise_vertical_std_m = 5.0  # GPS vertical is ~1.5× worse
        register_fmu3_param(self, "pos_noise_vertical_std_m")

        self.vel_noise_std_mps = 0.1
        register_fmu3_param(self, "vel_noise_std_mps")

        self.dropout_probability = 0.0  # fraction of updates that are dropped
        register_fmu3_param(self, "dropout_probability")

        self.rng_seed = 7
        register_fmu3_param(self, "rng_seed")

        # --- Internal state ---------------------------------------------------
        self._rng = np.random.default_rng(7)
        self._elapsed_since_update = 0.0
        self._update_interval = 0.1

    def enter_initialization_mode(self):
        self._rng = np.random.default_rng(int(self.rng_seed))
        self._update_interval = 1.0 / max(self.update_rate_hz, 0.1)
        self._elapsed_since_update = self._update_interval  # force first step
        self.measurement_ready = 0.0

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self._elapsed_since_update += step_size

        if self._elapsed_since_update < self._update_interval - 1e-9:
            self.measurement_ready = 0.0
            return True

        self._elapsed_since_update = 0.0

        # Optional signal dropout.
        if (
            self.dropout_probability > 0.0
            and self._rng.uniform() < self.dropout_probability
        ):
            self.measurement_ready = 0.0
            return True

        self._produce_measurement(current_time + step_size)
        return True

    def terminate(self):
        print("Terminating gnss_fmu.")
        self.time = 0.0

    def _produce_measurement(self, sim_time: float) -> None:
        pos = self.vehicle_state.state.pose.position
        vel = self.vehicle_state.velocity

        h_std = self.pos_noise_horizontal_std_m
        v_std = self.pos_noise_vertical_std_m
        s_std = self.vel_noise_std_mps

        self.pos_n_m = float(pos.x) + self._rng.normal(0.0, h_std)
        self.pos_e_m = float(pos.y) + self._rng.normal(0.0, h_std)
        self.pos_d_m = float(pos.z) + self._rng.normal(0.0, v_std)

        self.vel_n_mps = float(vel.x) + self._rng.normal(0.0, s_std)
        self.vel_e_mps = float(vel.y) + self._rng.normal(0.0, s_std)
        self.vel_d_mps = float(vel.z) + self._rng.normal(0.0, s_std)

        self.measurement_ready = 1.0
        self.gnss_time_s = sim_time
