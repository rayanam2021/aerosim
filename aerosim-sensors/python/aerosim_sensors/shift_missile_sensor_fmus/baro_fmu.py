"""
Barometric altimeter FMU for the missile stack.

Derives static pressure from the ISA atmosphere model at the vehicle's current
altitude, then converts it back to *pressure altitude* (what a real barometer
reports) and adds sensor noise and a slowly-varying bias.

Real barometric altimeters:
  - Report pressure altitude (altitude at which ISA pressure matches measured P).
  - Have ~1–3 m RMS noise at typical update rates.
  - Carry a fixed bias from calibration error; we model a Gaussian-drawn offset.
  - Update at ~25–50 Hz (much faster than GPS but slower than IMU).
  - Are unaffected by GPS jamming – complementary to GPS for altitude channel.

Inputs  (component topic): vehicle_state  (aerosim::types::VehicleState)
Outputs (aux topic):       pressure_pa, baro_alt_m,
                           measurement_ready, baro_time_s
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var
from aerosim_data import dict_to_namespace
from aerosim_data import types as aerosim_types

from atmosphere import isa as _isa, pressure_altitude_m as _press_alt


class baro_fmu(Fmi3Slave):
    """Barometric altimeter sensor with noise, bias and configurable rate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Missile barometric altimeter (ISA + noise + configurable rate)"

        self.vehicle_state = dict_to_namespace(aerosim_types.VehicleState().to_dict())
        register_fmu3_var(self, "vehicle_state", causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # Outputs.
        self.pressure_pa = 0.0       # measured static pressure [Pa]
        self.baro_alt_m = 0.0        # pressure altitude [m MSL]
        self.measurement_ready = 0.0
        self.baro_time_s = 0.0
        for _n in ("pressure_pa", "baro_alt_m", "measurement_ready", "baro_time_s"):
            register_fmu3_var(self, _n, causality="output")

        # --- Parameters -------------------------------------------------------
        self.update_rate_hz = 25.0
        register_fmu3_param(self, "update_rate_hz")

        # Pressure noise in Pa: converted to altitude error via ΔP/P ≈ ΔH/H_scale
        self.pressure_noise_std_pa = 5.0
        register_fmu3_param(self, "pressure_noise_std_pa")

        self.altitude_bias_std_m = 2.0  # systematic calibration offset [m]
        register_fmu3_param(self, "altitude_bias_std_m")

        self.world_origin_altitude = 0.0
        register_fmu3_param(self, "world_origin_altitude")

        self.rng_seed = 99
        register_fmu3_param(self, "rng_seed")

        # --- Internal state ---------------------------------------------------
        self._rng = np.random.default_rng(99)
        self._alt_bias = 0.0
        self._elapsed_since_update = 0.0
        self._update_interval = 0.04

    def enter_initialization_mode(self):
        self._rng = np.random.default_rng(int(self.rng_seed))
        self._update_interval = 1.0 / max(self.update_rate_hz, 0.1)
        self._elapsed_since_update = self._update_interval
        self._alt_bias = self._rng.normal(0.0, self.altitude_bias_std_m)
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
        self._produce_measurement(current_time + step_size)
        return True

    def terminate(self):
        print("Terminating baro_fmu.")
        self.time = 0.0

    def _altitude_msl(self) -> float:
        ned_down = float(self.vehicle_state.state.pose.position.z)
        return self.world_origin_altitude - ned_down

    def _produce_measurement(self, sim_time: float) -> None:
        alt_true = self._altitude_msl()
        _, P_true, _, _ = _isa(alt_true)

        # Add pressure sensor noise.
        P_noisy = P_true + self._rng.normal(0.0, self.pressure_noise_std_pa)
        P_noisy = max(P_noisy, 1.0)

        # Convert to pressure altitude and add fixed calibration bias.
        baro_alt = _press_alt(P_noisy) + self._alt_bias

        self.pressure_pa = float(P_noisy)
        self.baro_alt_m = float(baro_alt)
        self.measurement_ready = 1.0
        self.baro_time_s = sim_time
