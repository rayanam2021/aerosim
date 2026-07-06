"""
Structures / mass-properties FMU for the SHIFT missile (full inertia tensor).

As the propulsion FMU burns propellant the vehicle mass, principal inertias and
CG shift.  A slender axisymmetric missile has a small roll inertia (Ixx) and
large, nearly equal pitch/yaw inertias (Iyy ~ Izz).  Each is modelled as a
linear function of mass so the values stay physically consistent as propellant
depletes.  A structural load factor is derived from the aerodynamic normal
force fed back from the aerodynamics FMU.

Inputs  (aux): propellant_fraction (propulsion), true_fz_n (aero)
Outputs (aux): mass_kg, Ixx, Iyy, Izz, cg_x_m, load_factor_g
"""

from __future__ import annotations

from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var


class structures_sm_fmu(Fmi3Slave):
    """Mass/inertia depletion + structural load-factor model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT missile structures / mass-properties (full inertia)"

        self.propellant_fraction = 1.0
        register_fmu3_var(self, "propellant_fraction", causality="input")
        self.true_fz_n = 0.0
        register_fmu3_var(self, "true_fz_n", causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.mass_kg = 500.0
        self.Ixx = 5.0
        self.Iyy = 300.0
        self.Izz = 300.0
        self.cg_x_m = 1.5
        self.load_factor_g = 0.0
        for _n in ("mass_kg", "Ixx", "Iyy", "Izz", "cg_x_m", "load_factor_g"):
            register_fmu3_var(self, _n, causality="output")

        self.dry_mass_kg = 150.0
        register_fmu3_param(self, "dry_mass_kg")
        self.propellant_mass_kg = 350.0
        register_fmu3_param(self, "propellant_mass_kg")
        # Roll inertia per kg (slender body -> small); pitch/yaw per kg (large).
        self.roll_inertia_per_kg = 0.01
        register_fmu3_param(self, "roll_inertia_per_kg")
        self.pitch_inertia_per_kg = 0.60
        register_fmu3_param(self, "pitch_inertia_per_kg")
        self.cg_full_x_m = 1.35
        register_fmu3_param(self, "cg_full_x_m")
        self.cg_empty_x_m = 1.70
        register_fmu3_param(self, "cg_empty_x_m")

    def enter_initialization_mode(self):
        self._update(1.0, 0.0)

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self._update(self.propellant_fraction, self.true_fz_n)
        return True

    def terminate(self):
        print("Terminating structures_sm_fmu (mass properties).")
        self.time = 0.0

    def _update(self, propellant_fraction: float, fz_n: float) -> None:
        frac = max(0.0, min(1.0, propellant_fraction))
        self.mass_kg = self.dry_mass_kg + frac * self.propellant_mass_kg
        self.Ixx = self.roll_inertia_per_kg * self.mass_kg
        self.Iyy = self.pitch_inertia_per_kg * self.mass_kg
        self.Izz = self.pitch_inertia_per_kg * self.mass_kg
        self.cg_x_m = self.cg_full_x_m + (1.0 - frac) * (
            self.cg_empty_x_m - self.cg_full_x_m
        )
        weight = max(self.mass_kg, 1e-3) * 9.80665
        self.load_factor_g = abs(fz_n) / weight
