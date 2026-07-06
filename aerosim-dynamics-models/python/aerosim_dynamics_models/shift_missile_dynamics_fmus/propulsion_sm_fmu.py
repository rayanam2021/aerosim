"""
Propulsion (rocket motor) surrogate FMU for the missile stack.

Placeholder solid-motor thrust model with a boost/sustain profile and finite
propellant. Thrust is throttle-scaled and cuts off when propellant is depleted.
It reports the remaining propellant fraction and mass-flow so the structures FMU
can update the vehicle mass/inertia consistently.

Inputs  (auxiliary topic): throttle
Outputs (auxiliary topic): thrust_n, propellant_fraction, mass_flow_kg_s
"""

from __future__ import annotations

from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var


class propulsion_sm_fmu(Fmi3Slave):
    """Boost/sustain solid rocket motor surrogate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Missile propulsion (solid motor) surrogate"

        self.throttle = 1.0
        register_fmu3_var(self, "throttle", causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.thrust_n = 0.0
        self.propellant_fraction = 1.0
        self.mass_flow_kg_s = 0.0
        for _name in ("thrust_n", "propellant_fraction", "mass_flow_kg_s"):
            register_fmu3_var(self, _name, causality="output")

        # Motor parameters.
        self.boost_thrust_n = 50000.0
        register_fmu3_param(self, "boost_thrust_n")
        self.sustain_thrust_n = 8000.0
        register_fmu3_param(self, "sustain_thrust_n")
        self.boost_time_s = 2.0
        register_fmu3_param(self, "boost_time_s")
        self.burn_time_s = 6.0
        register_fmu3_param(self, "burn_time_s")
        self.propellant_mass_kg = 350.0
        register_fmu3_param(self, "propellant_mass_kg")

        self._propellant_remaining_kg = 350.0

    def enter_initialization_mode(self):
        self._propellant_remaining_kg = self.propellant_mass_kg
        self.propellant_fraction = 1.0
        self.thrust_n = 0.0
        self.mass_flow_kg_s = 0.0

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size

        throttle = max(0.0, min(1.0, self.throttle))

        if self._propellant_remaining_kg <= 0.0 or current_time >= self.burn_time_s:
            commanded_thrust = 0.0
        elif current_time < self.boost_time_s:
            commanded_thrust = self.boost_thrust_n
        else:
            commanded_thrust = self.sustain_thrust_n
        commanded_thrust *= throttle

        # Deplete propellant proportionally to thrust over the burn.
        total_impulse = (
            self.boost_thrust_n * self.boost_time_s
            + self.sustain_thrust_n * max(0.0, self.burn_time_s - self.boost_time_s)
        )
        if total_impulse > 0.0 and commanded_thrust > 0.0:
            burn_rate = commanded_thrust / total_impulse * self.propellant_mass_kg
            consumed = min(self._propellant_remaining_kg, burn_rate * step_size)
            self._propellant_remaining_kg -= consumed
            self.mass_flow_kg_s = consumed / step_size if step_size > 0.0 else 0.0
            if self._propellant_remaining_kg <= 0.0:
                commanded_thrust = 0.0
        else:
            self.mass_flow_kg_s = 0.0

        self.thrust_n = commanded_thrust
        self.propellant_fraction = (
            self._propellant_remaining_kg / self.propellant_mass_kg
            if self.propellant_mass_kg > 0.0
            else 0.0
        )
        return True

    def terminate(self):
        print("Terminating propulsion_sm_fmu (missile motor).")
        self.time = 0.0
