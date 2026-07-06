"""
Infrared (IR) seeker FMU for the SHIFT interceptor.

A passive, body-fixed/gimballed IR seeker that reports the line-of-sight (LOS)
azimuth/elevation to the threat missile.  Being passive it provides **no range**
— only bearing — which is exactly why the target navigation EKF fuses it with
the semi-active radar (range + bearing).  The model captures the operationally
important effects:

  - Maximum lock range (thermal signature falls off with range).
  - Field-of-view / gimbal-limit gating about the ego boresight (body +x).
  - Angular white noise (very low for IR -> precise angles).
  - Configurable update rate with a ``measurement_ready`` strobe.

LOS angles are expressed in the NED frame (azimuth from North, elevation from
the local horizontal), i.e. an INS-stabilised seeker output, which keeps the
target EKF measurement model clean.

The seeker reads the ego and threat kinematics as scalar aux signals (the ego
plant mirrors its truth to aux; the threat plant publishes its own), because a
single FMU cannot bind two VehicleState component inputs in AeroSim.

Inputs  (aux): ego_pos_{n,e,d}, ego_q{w,x,y,z}, tgt_pos_{n,e,d}
Outputs (aux): ir_az_rad, ir_el_rad, ir_locked, measurement_ready, ir_time_s
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var


class ir_seeker_fmu(Fmi3Slave):
    """Passive IR seeker (bearing-only LOS)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Infrared seeker (passive, bearing-only)"

        self.ego_pos_n = 0.0
        self.ego_pos_e = 0.0
        self.ego_pos_d = 0.0
        self.ego_qw = 1.0
        self.ego_qx = 0.0
        self.ego_qy = 0.0
        self.ego_qz = 0.0
        self.tgt_pos_n = 0.0
        self.tgt_pos_e = 0.0
        self.tgt_pos_d = 0.0
        for _n in (
            "ego_pos_n", "ego_pos_e", "ego_pos_d",
            "ego_qw", "ego_qx", "ego_qy", "ego_qz",
            "tgt_pos_n", "tgt_pos_e", "tgt_pos_d",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.ir_az_rad = 0.0
        self.ir_el_rad = 0.0
        self.ir_locked = 0.0
        self.measurement_ready = 0.0
        self.ir_time_s = 0.0
        for _n in ("ir_az_rad", "ir_el_rad", "ir_locked", "measurement_ready", "ir_time_s"):
            register_fmu3_var(self, _n, causality="output")

        self.update_rate_hz = 100.0
        register_fmu3_param(self, "update_rate_hz")
        self.max_lock_range_m = 15000.0
        register_fmu3_param(self, "max_lock_range_m")
        self.fov_halfangle_rad = 0.785398  # 45 deg gimbal limit
        register_fmu3_param(self, "fov_halfangle_rad")
        self.angle_noise_std_rad = 0.001
        register_fmu3_param(self, "angle_noise_std_rad")
        self.rng_seed = 123
        register_fmu3_param(self, "rng_seed")

        self._rng = np.random.default_rng(123)
        self._elapsed = 0.0
        self._interval = 0.01

    def enter_initialization_mode(self):
        self._rng = np.random.default_rng(int(self.rng_seed))
        self._interval = 1.0 / max(self.update_rate_hz, 1.0)
        self._elapsed = self._interval
        self.measurement_ready = 0.0

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self._elapsed += step_size
        if self._elapsed < self._interval - 1e-9:
            self.measurement_ready = 0.0
            return True
        self._elapsed = 0.0
        self._measure(current_time + step_size)
        return True

    def terminate(self):
        print("Terminating ir_seeker_fmu.")
        self.time = 0.0

    def _measure(self, sim_time):
        r = np.array([
            self.tgt_pos_n - self.ego_pos_n,
            self.tgt_pos_e - self.ego_pos_e,
            self.tgt_pos_d - self.ego_pos_d,
        ])
        rng = float(np.linalg.norm(r))
        if rng < 1e-3:
            self.ir_locked = 0.0
            self.measurement_ready = 0.0
            return

        los = r / rng
        if self.ego_qw == 0.0 and self.ego_qx == 0.0 and self.ego_qy == 0.0 and self.ego_qz == 0.0:
            boresight = np.array([1.0, 0.0, 0.0])
        else:
            boresight = Rotation.from_quat(
                [self.ego_qx, self.ego_qy, self.ego_qz, self.ego_qw]
            ).apply([1.0, 0.0, 0.0])
        off_angle = float(np.arccos(np.clip(np.dot(los, boresight), -1.0, 1.0)))

        if rng > self.max_lock_range_m or off_angle > self.fov_halfangle_rad:
            self.ir_locked = 0.0
            self.measurement_ready = 0.0
            return

        ground = max(float(np.hypot(r[0], r[1])), 1e-6)
        az = float(np.arctan2(r[1], r[0]))
        el = float(np.arctan2(-r[2], ground))
        self.ir_az_rad = az + float(self._rng.normal(0.0, self.angle_noise_std_rad))
        self.ir_el_rad = el + float(self._rng.normal(0.0, self.angle_noise_std_rad))
        self.ir_locked = 1.0
        self.measurement_ready = 1.0
        self.ir_time_s = sim_time
