"""
Structures FMU — mass properties + Euler-Bernoulli airframe load solver.

Beyond bookkeeping the mass/inertia/CG as propellant burns, this FMU runs a
station-discretized **free-free beam** load analysis of the airframe every step
using the aerodynamic and control forces fed back from the aerodynamics FMU.  It
computes the internal shear and bending-moment distribution, the peak bending
stress, the axial (thrust) stress, a von-Mises-style combined stress, the
structural margin of safety against the material allowable, the airframe first
bending natural frequency (an aeroelastic quantity of interest), and a
probabilistic structural-failure estimate (UQ) from the load/material scatter.

Load model (Euler-Bernoulli, free-free)
---------------------------------------
The missile is a slender beam.  In a maneuver the aerodynamic normal force N acts
near the center of pressure while the body accelerates at a = N/m; d'Alembert
inertial relief distributes ``-(m_i/m) N`` along the body, so the transverse load
per station is

    q_i = N_aero,i - (m_i / m) * N_total          (Σ q_i = 0, self-equilibrated)

Integrating gives the shear ``V(x)=∫q`` and bending moment ``M(x)=∫V`` (Euler-
Bernoulli ``M = -EI w''``, ``V = dM/dx``).  The peak bending stress on the
thin-wall tube is ``σ_b = M_max r_o / I_area`` and the axial stress is
``σ_a = T / A_x``; they superpose on the compression fiber.  The margin of
safety is ``MoS = σ_allow / (FoS · σ_max) − 1`` with a 1.5 aerospace factor of
safety.  The free-free first bending mode uses ``β₁L = 4.730``:

    f₁ = (β₁L)² / (2π L²) · sqrt(EI / μ),   μ = m / L.

Inputs  (aux): propellant_fraction (propulsion); true_fz_n, true_fy_n (aero);
               thrust_n (propulsion)
Outputs (aux): mass_kg, Ixx, Iyy, Izz, cg_x_m, load_factor_g,
               max_bending_moment_nm, max_bending_stress_pa, axial_stress_pa,
               combined_stress_pa, margin_of_safety, first_bending_freq_hz,
               prob_structural_failure

References
----------
* S. Timoshenko, *Strength of Materials, Part I*, Van Nostrand 1955.
* R. D. Blevins, *Formulas for Natural Frequency and Mode Shape*, Krieger 1979
  (free-free beam β₁L = 4.730).
* Bruhn, *Analysis and Design of Flight Vehicle Structures*, 1973 (margins).
* K. Nakka / R. Nakka, solid-rocket airframe loads notes; ERAU *Introduction to
  Aerospace Flight Vehicles* (beam bending of slender vehicles).
"""

from __future__ import annotations

import math

import numpy as np
from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var

G0 = 9.80665


class structures_sm_fmu(Fmi3Slave):
    """Mass-property depletion + Euler-Bernoulli airframe load/stress solver."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT missile structures (mass props + beam loads/stress)"

        self.propellant_fraction = 1.0
        self.true_fz_n = 0.0
        self.true_fy_n = 0.0
        self.thrust_n = 0.0
        for _n in ("propellant_fraction", "true_fz_n", "true_fy_n", "thrust_n"):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.mass_kg = 230.0
        self.Ixx = 2.3
        self.Iyy = 235.0
        self.Izz = 235.0
        self.cg_x_m = 1.8
        self.load_factor_g = 0.0
        self.max_bending_moment_nm = 0.0
        self.max_bending_stress_pa = 0.0
        self.axial_stress_pa = 0.0
        self.combined_stress_pa = 0.0
        self.margin_of_safety = 0.0
        self.first_bending_freq_hz = 0.0
        self.prob_structural_failure = 0.0
        for _n in (
            "mass_kg", "Ixx", "Iyy", "Izz", "cg_x_m", "load_factor_g",
            "max_bending_moment_nm", "max_bending_stress_pa", "axial_stress_pa",
            "combined_stress_pa", "margin_of_safety", "first_bending_freq_hz",
            "prob_structural_failure",
        ):
            register_fmu3_var(self, _n, causality="output")

        # --- Mass properties --------------------------------------------------
        self.dry_mass_kg = 150.0
        register_fmu3_param(self, "dry_mass_kg")
        self.propellant_mass_kg = 80.0
        register_fmu3_param(self, "propellant_mass_kg")
        self.roll_inertia_per_kg = 0.01
        register_fmu3_param(self, "roll_inertia_per_kg")
        self.pitch_inertia_per_kg = 1.02
        register_fmu3_param(self, "pitch_inertia_per_kg")
        self.cg_full_x_m = 1.75
        register_fmu3_param(self, "cg_full_x_m")
        self.cg_empty_x_m = 1.55
        register_fmu3_param(self, "cg_empty_x_m")

        # --- Airframe geometry + material ------------------------------------
        self.body_length_m = 3.5
        register_fmu3_param(self, "body_length_m")
        self.body_outer_radius_m = 0.10
        register_fmu3_param(self, "body_outer_radius_m")
        self.wall_thickness_m = 0.004
        register_fmu3_param(self, "wall_thickness_m")
        # Legacy single-CP fallback (used only if canard/tail stations disabled).
        self.cp_fraction = 0.55
        register_fmu3_param(self, "cp_fraction")
        # Modular canard + tail load stations (nose = 0).  When both are > 0 the
        # beam solver splits the transverse aero load between the two banks
        # (canards typically carry more of the control force).
        self.canard_x_m = 0.55
        register_fmu3_param(self, "canard_x_m")
        self.tail_x_m = 3.15
        register_fmu3_param(self, "tail_x_m")
        self.canard_load_fraction = 0.65   # share of |N| applied at canards
        register_fmu3_param(self, "canard_load_fraction")
        self.youngs_modulus_pa = 71.0e9    # aluminum 7075-T6
        register_fmu3_param(self, "youngs_modulus_pa")
        self.yield_strength_pa = 480.0e6   # 7075-T6 tensile yield
        register_fmu3_param(self, "yield_strength_pa")
        self.factor_of_safety = 1.5
        register_fmu3_param(self, "factor_of_safety")
        self.n_stations = 40
        register_fmu3_param(self, "n_stations")

        # --- UQ ---------------------------------------------------------------
        self.sigma_yield_rel = 0.06        # material allowable scatter
        register_fmu3_param(self, "sigma_yield_rel")
        self.sigma_load_rel = 0.10         # aero/structural load scatter
        register_fmu3_param(self, "sigma_load_rel")

        # Precomputed section properties.
        self._I_area = 1.0
        self._area_x = 1.0

    def enter_initialization_mode(self):
        r_o = self.body_outer_radius_m
        r_i = max(r_o - self.wall_thickness_m, 1e-4)
        self._I_area = 0.25 * math.pi * (r_o ** 4 - r_i ** 4)   # tube 2nd moment
        self._area_x = math.pi * (r_o ** 2 - r_i ** 2)          # cross-section area
        self._update(1.0, 0.0, 0.0, 0.0)

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        self._update(self.propellant_fraction, self.true_fz_n,
                     self.true_fy_n, self.thrust_n)
        return True

    def terminate(self):
        print("Terminating structures_sm_fmu (mass props + beam loads).")
        self.time = 0.0

    # ------------------------------------------------------------------ core
    def _update(self, propellant_fraction, fz_n, fy_n, thrust_n) -> None:
        frac = max(0.0, min(1.0, propellant_fraction))
        self.mass_kg = self.dry_mass_kg + frac * self.propellant_mass_kg
        self.Ixx = self.roll_inertia_per_kg * self.mass_kg
        self.Iyy = self.pitch_inertia_per_kg * self.mass_kg
        self.Izz = self.pitch_inertia_per_kg * self.mass_kg
        self.cg_x_m = self.cg_full_x_m + (1.0 - frac) * (
            self.cg_empty_x_m - self.cg_full_x_m
        )

        # Transverse aerodynamic force resultant (pitch + yaw planes).
        n_total = math.hypot(fz_n, fy_n)
        weight = max(self.mass_kg, 1e-3) * G0
        self.load_factor_g = n_total / weight

        M_max = self._beam_bending_moment(n_total)
        self.max_bending_moment_nm = M_max
        self.max_bending_stress_pa = M_max * self.body_outer_radius_m / max(self._I_area, 1e-12)
        self.axial_stress_pa = abs(thrust_n) / max(self._area_x, 1e-12)
        # Bending + axial compression superpose on the compression fiber.
        self.combined_stress_pa = self.max_bending_stress_pa + self.axial_stress_pa

        allow = self.yield_strength_pa
        applied = self.factor_of_safety * max(self.combined_stress_pa, 1.0)
        self.margin_of_safety = allow / applied - 1.0
        self.first_bending_freq_hz = self._first_bending_freq()
        self.prob_structural_failure = self._failure_probability()

    def _beam_bending_moment(self, n_total: float) -> float:
        """Peak bending moment of the free-free beam under aero + inertial load."""
        if n_total <= 0.0:
            return 0.0
        n = max(6, int(self.n_stations))
        L = self.body_length_m
        dx = L / n
        m_i = np.full(n, self.mass_kg / n)               # uniform mass distribution
        aero = np.zeros(n)

        # Prefer canard + tail stations when configured; otherwise legacy CP.
        use_banks = (self.canard_x_m > 0.0 and self.tail_x_m > 0.0
                     and self.tail_x_m > self.canard_x_m)
        if use_banks:
            f_can = float(np.clip(self.canard_load_fraction, 0.0, 1.0))
            i_can = int(np.clip(self.canard_x_m / L * n, 0, n - 1))
            i_tail = int(np.clip(self.tail_x_m / L * n, 0, n - 1))
            aero[i_can] += f_can * n_total
            aero[i_tail] += (1.0 - f_can) * n_total
        else:
            cp_idx = int(np.clip(self.cp_fraction * n, 0, n - 1))
            aero[cp_idx] = n_total

        # d'Alembert inertial relief -> self-equilibrated transverse load.
        q = aero - (m_i / max(self.mass_kg, 1e-9)) * n_total
        V = np.cumsum(q)                                  # shear (station sums)
        M = np.cumsum(V) * dx                             # bending moment
        return float(np.max(np.abs(M)))

    def _first_bending_freq(self) -> float:
        beta_L = 4.730040745                              # free-free first mode
        L = self.body_length_m
        EI = self.youngs_modulus_pa * self._I_area
        mu = max(self.mass_kg, 1e-3) / L                  # mass per unit length
        return (beta_L ** 2) / (2.0 * math.pi * L * L) * math.sqrt(EI / mu)

    def _failure_probability(self) -> float:
        """P(FoS·load > allowable) with Gaussian load & material scatter."""
        applied = self.factor_of_safety * self.combined_stress_pa
        allow = self.yield_strength_pa
        s_load = self.sigma_load_rel * max(applied, 1.0)
        s_allow = self.sigma_yield_rel * allow
        margin = allow - applied
        sigma = math.sqrt(s_load ** 2 + s_allow ** 2)
        if sigma <= 0.0:
            return 0.0 if margin > 0 else 1.0
        # P(failure) = P(margin < 0) = Phi(-margin/sigma).
        return 0.5 * math.erfc((margin / sigma) / math.sqrt(2.0))
