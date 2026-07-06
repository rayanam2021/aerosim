"""
Fin servo / actuator FMU (3-axis) for the SHIFT missile.

Models the finite bandwidth and authority of the elevator (pitch), aileron
(roll) and rudder (yaw) fin actuators.  Each channel is an independent
first-order lag with a slew-rate limit and a hard deflection limit; throttle is
passed through so downstream FMUs can source it from one actuator topic.

Inputs  (aux): elevator_cmd_rad, aileron_cmd_rad, rudder_cmd_rad, throttle_cmd
Outputs (aux): elevator_rad, aileron_rad, rudder_rad, throttle
"""

from __future__ import annotations

from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var


class servo_sm_fmu(Fmi3Slave):
    """Three independent first-order fin actuators with rate/position limits."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT missile 3-axis fin servo dynamics"

        self.elevator_cmd_rad = 0.0
        self.aileron_cmd_rad = 0.0
        self.rudder_cmd_rad = 0.0
        self.throttle_cmd = 0.0
        for _n in ("elevator_cmd_rad", "aileron_cmd_rad", "rudder_cmd_rad", "throttle_cmd"):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.elevator_rad = 0.0
        self.aileron_rad = 0.0
        self.rudder_rad = 0.0
        self.throttle = 0.0
        for _n in ("elevator_rad", "aileron_rad", "rudder_rad", "throttle"):
            register_fmu3_var(self, _n, causality="output")

        self.time_constant_s = 0.03
        register_fmu3_param(self, "time_constant_s")
        self.max_rate_rps = 10.0
        register_fmu3_param(self, "max_rate_rps")
        self.max_deflection_rad = 0.436332  # 25 deg
        register_fmu3_param(self, "max_deflection_rad")

    def enter_initialization_mode(self):
        self.elevator_rad = 0.0
        self.aileron_rad = 0.0
        self.rudder_rad = 0.0

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self.elevator_rad = self._advance(self.elevator_rad, self.elevator_cmd_rad, step_size)
        self.aileron_rad = self._advance(self.aileron_rad, self.aileron_cmd_rad, step_size)
        self.rudder_rad = self._advance(self.rudder_rad, self.rudder_cmd_rad, step_size)
        self.throttle = max(0.0, min(1.0, self.throttle_cmd))
        return True

    def terminate(self):
        print("Terminating servo_sm_fmu (3-axis fins).")
        self.time = 0.0

    def _advance(self, current: float, command: float, dt: float) -> float:
        lim = self.max_deflection_rad
        cmd = max(-lim, min(lim, command))
        tau = max(self.time_constant_s, 1e-4)
        desired_rate = (cmd - current) / tau
        rate = max(-self.max_rate_rps, min(self.max_rate_rps, desired_rate))
        new = current + rate * dt
        return max(-lim, min(lim, new))
