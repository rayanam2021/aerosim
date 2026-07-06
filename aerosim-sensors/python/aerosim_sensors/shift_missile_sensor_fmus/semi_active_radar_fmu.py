"""
Semi-active radar (SAR) seeker FMU for the SHIFT interceptor.

A semi-active radar homing seeker: an external illuminator (e.g. the launch
platform) floods the threat and the missile's receiver measures the reflected
return.  Unlike the passive IR seeker it provides a full 3-D fix — range,
range-rate (Doppler) and LOS bearing — but with coarser angular accuracy and a
lower update rate.  Modelled effects:

  - Maximum detection range (radar-range-equation roll-off, hard-gated here).
  - Field-of-view gating about the ego boresight (body +x).
  - Independent range, Doppler range-rate and angular white noise.
  - Configurable update rate with a ``measurement_ready`` strobe.

LOS angles are reported in the NED frame (INS-stabilised), matching the IR
seeker and the target EKF measurement model.

The seeker reads ego and threat kinematics as scalar aux signals (the ego plant
mirrors its truth to aux; the threat plant publishes its own), because a single
FMU cannot bind two VehicleState component inputs in AeroSim.

Inputs  (aux): ego_pos_{n,e,d}, ego_vel_{n,e,d}, ego_q{w,x,y,z},
               tgt_pos_{n,e,d}, tgt_vel_{n,e,d}
Outputs (aux): radar_range_m, radar_az_rad, radar_el_rad,
               radar_range_rate_mps, radar_locked, measurement_ready,
               radar_time_s
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var


class semi_active_radar_fmu(Fmi3Slave):
    """Semi-active radar homing seeker (range + Doppler + bearing)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Semi-active radar seeker (range/Doppler/bearing)"

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
        self.tgt_pos_n = 0.0
        self.tgt_pos_e = 0.0
        self.tgt_pos_d = 0.0
        self.tgt_vel_n = 0.0
        self.tgt_vel_e = 0.0
        self.tgt_vel_d = 0.0
        for _n in (
            "ego_pos_n", "ego_pos_e", "ego_pos_d",
            "ego_vel_n", "ego_vel_e", "ego_vel_d",
            "ego_qw", "ego_qx", "ego_qy", "ego_qz",
            "tgt_pos_n", "tgt_pos_e", "tgt_pos_d",
            "tgt_vel_n", "tgt_vel_e", "tgt_vel_d",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.radar_range_m = 0.0
        self.radar_az_rad = 0.0
        self.radar_el_rad = 0.0
        self.radar_range_rate_mps = 0.0
        self.radar_locked = 0.0
        self.measurement_ready = 0.0
        self.radar_time_s = 0.0
        for _n in (
            "radar_range_m", "radar_az_rad", "radar_el_rad",
            "radar_range_rate_mps", "radar_locked", "measurement_ready",
            "radar_time_s",
        ):
            register_fmu3_var(self, _n, causality="output")

        self.update_rate_hz = 20.0
        register_fmu3_param(self, "update_rate_hz")
        self.max_detection_range_m = 40000.0
        register_fmu3_param(self, "max_detection_range_m")
        self.fov_halfangle_rad = 0.523599  # 30 deg
        register_fmu3_param(self, "fov_halfangle_rad")
        self.range_noise_std_m = 15.0
        register_fmu3_param(self, "range_noise_std_m")
        self.range_rate_noise_std_mps = 3.0
        register_fmu3_param(self, "range_rate_noise_std_mps")
        self.angle_noise_std_rad = 0.005
        register_fmu3_param(self, "angle_noise_std_rad")
        self.rng_seed = 321
        register_fmu3_param(self, "rng_seed")

        self._rng = np.random.default_rng(321)
        self._elapsed = 0.0
        self._interval = 0.05

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
        print("Terminating semi_active_radar_fmu.")
        self.time = 0.0

    def _measure(self, sim_time):
        r = np.array([
            self.tgt_pos_n - self.ego_pos_n,
            self.tgt_pos_e - self.ego_pos_e,
            self.tgt_pos_d - self.ego_pos_d,
        ])
        v = np.array([
            self.tgt_vel_n - self.ego_vel_n,
            self.tgt_vel_e - self.ego_vel_e,
            self.tgt_vel_d - self.ego_vel_d,
        ])
        rng = float(np.linalg.norm(r))
        if rng < 1e-3:
            self.radar_locked = 0.0
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

        if rng > self.max_detection_range_m or off_angle > self.fov_halfangle_rad:
            self.radar_locked = 0.0
            self.measurement_ready = 0.0
            return

        ground = max(float(np.hypot(r[0], r[1])), 1e-6)
        az = float(np.arctan2(r[1], r[0]))
        el = float(np.arctan2(-r[2], ground))
        range_rate = float(np.dot(r, v)) / rng

        self.radar_range_m = rng + float(self._rng.normal(0.0, self.range_noise_std_m))
        self.radar_az_rad = az + float(self._rng.normal(0.0, self.angle_noise_std_rad))
        self.radar_el_rad = el + float(self._rng.normal(0.0, self.angle_noise_std_rad))
        self.radar_range_rate_mps = range_rate + float(
            self._rng.normal(0.0, self.range_rate_noise_std_mps)
        )
        self.radar_locked = 1.0
        self.measurement_ready = 1.0
        self.radar_time_s = sim_time
