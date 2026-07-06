"""
Threat missile scenario FMU (the target the ego SHIFT missile intercepts).

This is the **perfect ground-truth plant** for the threat (``actor2``).  It is a
guided point-mass with a 6-DOF-consistent attitude (a quaternion is synthesised
from the flight-path angles plus the bank required for the commanded turn), so
downstream consumers get a full ``VehicleState`` with orientation and body
rates.

Everything an analyst would want to "rapidly adjust" about the threat is a
parameter and is reflected immediately in its trajectory:

    Engagement geometry : launch_range_m, launch_bearing_deg, cruise_altitude_m
    Speed               : cruise_speed_mps, max_axial_accel_mps2
    Agility (surfaces)  : max_lateral_accel_g  (control-surface authority)
    Ingress             : target_north_m, target_east_m (defended asset it runs at)
    Evasion             : weave_amplitude_g, weave_frequency_hz
    Sensor grade        : onboard_nav_error_std_m (documents seeker/nav quality)

The guidance law is a simple altitude-hold + speed-hold + turn-to-waypoint with
a superimposed sinusoidal weave, all saturated by the configured accel limits so
a less-agile threat (small ``max_lateral_accel_g``) visibly under-turns.

Inputs  : none (autonomous scenario actor)
Outputs : vehicle_state (component, threat truth) + aux pos/vel/speed/altitude
"""

from __future__ import annotations

import math

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var
from aerosim_data import dict_to_namespace
from aerosim_data import types as aerosim_types

from sixdof import quat_from_euler, quat_to_msg

GRAVITY = 9.80665


class threat_missile_fmu(Fmi3Slave):
    """Configurable guided threat missile (point-mass, 6-DOF-consistent attitude)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Configurable threat missile (interception target)"

        self.vehicle_state = dict_to_namespace(aerosim_types.VehicleState().to_dict())
        register_fmu3_var(self, "vehicle_state", causality="output")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.threat_pos_n = 0.0
        self.threat_pos_e = 0.0
        self.threat_pos_d = 0.0
        self.threat_vel_n = 0.0
        self.threat_vel_e = 0.0
        self.threat_vel_d = 0.0
        self.threat_speed_mps = 0.0
        self.threat_altitude_m = 0.0
        for _n in (
            "threat_pos_n", "threat_pos_e", "threat_pos_d",
            "threat_vel_n", "threat_vel_e", "threat_vel_d",
            "threat_speed_mps", "threat_altitude_m",
        ):
            register_fmu3_var(self, _n, causality="output")

        # Engagement geometry.
        self.launch_range_m = 20000.0
        register_fmu3_param(self, "launch_range_m")
        self.launch_bearing_deg = 0.0
        register_fmu3_param(self, "launch_bearing_deg")
        self.cruise_altitude_m = 5000.0
        register_fmu3_param(self, "cruise_altitude_m")
        # Speed / agility.
        self.cruise_speed_mps = 300.0
        register_fmu3_param(self, "cruise_speed_mps")
        self.max_axial_accel_mps2 = 30.0
        register_fmu3_param(self, "max_axial_accel_mps2")
        self.max_lateral_accel_g = 8.0
        register_fmu3_param(self, "max_lateral_accel_g")
        # Ingress target (defended asset the threat flies toward).
        self.target_north_m = 0.0
        register_fmu3_param(self, "target_north_m")
        self.target_east_m = 0.0
        register_fmu3_param(self, "target_east_m")
        # Evasion.
        self.weave_amplitude_g = 0.0
        register_fmu3_param(self, "weave_amplitude_g")
        self.weave_frequency_hz = 0.2
        register_fmu3_param(self, "weave_frequency_hz")
        # Documented sensor/nav grade (affects how well it can be tracked/kept on
        # course in higher-fidelity iterations; not used by the point-mass plant).
        self.onboard_nav_error_std_m = 10.0
        register_fmu3_param(self, "onboard_nav_error_std_m")
        # World origin altitude for MSL/NED conversion.
        self.world_origin_altitude = 0.0
        register_fmu3_param(self, "world_origin_altitude")

        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._prev_rot = Rotation.identity()

    def enter_initialization_mode(self):
        bearing = math.radians(self.launch_bearing_deg)
        pD = self.world_origin_altitude - self.cruise_altitude_m
        self._pos = np.array([
            self.launch_range_m * math.cos(bearing),
            self.launch_range_m * math.sin(bearing),
            pD,
        ])
        # Initial heading: toward the ingress waypoint.
        to_wp = np.array([self.target_north_m, self.target_east_m]) - self._pos[:2]
        head = math.atan2(to_wp[1], to_wp[0]) if np.linalg.norm(to_wp) > 1e-6 else bearing + math.pi
        self._vel = self.cruise_speed_mps * np.array([math.cos(head), math.sin(head), 0.0])
        self._prev_rot = self._attitude(self._vel, 0.0)
        self._write_state(np.zeros(3), self._prev_rot, 0.0)

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        t = current_time + step_size
        self.time = t

        speed = max(float(np.linalg.norm(self._vel)), 1.0)
        vhat = self._vel / speed

        # Axial (speed-hold) acceleration along velocity.
        axial = np.clip(
            self.cruise_speed_mps - speed, -self.max_axial_accel_mps2, self.max_axial_accel_mps2
        ) * vhat

        # Turn-to-waypoint lateral command in the horizontal plane.
        to_wp = np.array([self.target_north_m, self.target_east_m]) - self._pos[:2]
        if np.linalg.norm(to_wp) > 1e-6:
            desired_head = math.atan2(to_wp[1], to_wp[0])
        else:
            desired_head = math.atan2(self._vel[1], self._vel[0])
        cur_head = math.atan2(self._vel[1], self._vel[0])
        head_err = math.atan2(math.sin(desired_head - cur_head), math.cos(desired_head - cur_head))

        max_lat = self.max_lateral_accel_g * GRAVITY
        # Horizontal turn accel (perpendicular to velocity, left-positive).
        left = np.array([-vhat[1], vhat[0], 0.0])
        turn_cmd = np.clip(4.0 * head_err * speed, -max_lat, max_lat)
        weave = self.weave_amplitude_g * GRAVITY * math.sin(2.0 * math.pi * self.weave_frequency_hz * t)
        lateral = left * (turn_cmd + weave)

        # Altitude hold (vertical accel, NED down positive).
        alt = self.world_origin_altitude - self._pos[2]
        alt_err = self.cruise_altitude_m - alt
        vert_accel = np.clip(2.0 * alt_err - 1.5 * (-self._vel[2]), -max_lat, max_lat)
        vertical = np.array([0.0, 0.0, -vert_accel])

        accel = axial + lateral + vertical
        self._vel = self._vel + accel * step_size
        self._pos = self._pos + self._vel * step_size

        rot = self._attitude(self._vel, turn_cmd + weave)
        omega = self._body_rates(self._prev_rot, rot, step_size)
        self._prev_rot = rot
        self._write_state(accel, rot, 0.0, omega)
        return True

    def terminate(self):
        print("Terminating threat_missile_fmu.")
        self.time = 0.0

    def _attitude(self, vel, lateral_accel):
        speed = max(float(np.linalg.norm(vel)), 1e-3)
        yaw = math.atan2(vel[1], vel[0])
        pitch = math.atan2(-vel[2], max(math.hypot(vel[0], vel[1]), 1e-6))
        # Coordinated-turn bank from commanded horizontal lateral accel.
        roll = math.atan2(lateral_accel, GRAVITY)
        return Rotation.from_quat(quat_from_euler(roll, pitch, yaw))

    @staticmethod
    def _body_rates(prev_rot, rot, dt):
        if dt <= 0.0:
            return np.zeros(3)
        delta = rot * prev_rot.inv()
        return delta.as_rotvec() / dt

    def _write_state(self, accel, rot, _unused, omega=None):
        if omega is None:
            omega = np.zeros(3)
        self._pos = np.asarray(self._pos, dtype=float)
        self.vehicle_state.state.pose.position.x = float(self._pos[0])
        self.vehicle_state.state.pose.position.y = float(self._pos[1])
        self.vehicle_state.state.pose.position.z = float(self._pos[2])
        qw, qx, qy, qz = quat_to_msg(rot.as_quat())
        self.vehicle_state.state.pose.orientation.w = qw
        self.vehicle_state.state.pose.orientation.x = qx
        self.vehicle_state.state.pose.orientation.y = qy
        self.vehicle_state.state.pose.orientation.z = qz
        self.vehicle_state.velocity.x = float(self._vel[0])
        self.vehicle_state.velocity.y = float(self._vel[1])
        self.vehicle_state.velocity.z = float(self._vel[2])
        self.vehicle_state.angular_velocity.x = float(omega[0])
        self.vehicle_state.angular_velocity.y = float(omega[1])
        self.vehicle_state.angular_velocity.z = float(omega[2])
        self.vehicle_state.acceleration.x = float(accel[0])
        self.vehicle_state.acceleration.y = float(accel[1])
        self.vehicle_state.acceleration.z = float(accel[2])

        self.threat_pos_n, self.threat_pos_e, self.threat_pos_d = (float(v) for v in self._pos)
        self.threat_vel_n, self.threat_vel_e, self.threat_vel_d = (float(v) for v in self._vel)
        self.threat_speed_mps = float(np.linalg.norm(self._vel))
        self.threat_altitude_m = float(self.world_origin_altitude - self._pos[2])
