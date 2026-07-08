"""
Propulsion FMU — 0-D internal-ballistics rocket motor (solid or liquid).

This replaces the earlier boost/sustain lookup with a physics-based lumped
(0-D) internal-ballistics model that computes chamber pressure, thrust and
mass-flow from first principles, and rolls up the quantities of interest (QoI)
that drive interceptor performance and the probability of success.

Solid motor (``motor_type = "solid"``)
--------------------------------------
Cylindrical (BATES-style, ends-inhibited, case-bonded) grain.  The grain length
is derived from the loaded propellant mass so the motor is self-consistent with
the ``structures`` FMU.  Each step:

    burn-surface     A_b   = 2 pi (r_port0 + w) L           (internal tube)
    klemmung         Kn    = A_b / A_t
    chamber pressure p_c   = (rho_p * a * c* * Kn)^(1/(1-n))   (mass balance)
    burn rate        r_dot = a * p_c^n                        (St. Robert)
    mass generation  mdot  = rho_p * A_b * r_dot
    web burned       w    += r_dot * dt                       (until burnout)
    thrust           F     = C_f * p_c * A_t                  (+ ambient corr.)

with the thrust coefficient C_f evaluated from the nozzle expansion ratio and
the flight ambient pressure (so thrust rises with altitude).

Liquid engine (``motor_type = "liquid"``)
-----------------------------------------
Throttleable pressure-fed/pump-fed abstraction: ``mdot = throttle * mdot_ref``,
``p_c = throttle * p_c_ref``, thrust from the same nozzle C_f relation, cut off
when the propellant load is exhausted.

Uncertainty quantification (UQ)
-------------------------------
Motor performance scatters run-to-run (propellant temperature, c* efficiency,
burn-rate coefficient, throat erosion).  A first-order (delta-method) thrust
1-sigma is reported from the relative uncertainties of ``a`` and ``c*`` — the
dominant contributors — so downstream p_kill Monte-Carlo can propagate it.

Inputs  (aux): throttle; ambient_pressure_pa (from aero atmosphere)
Outputs (aux): thrust_n, thrust_sigma_n, propellant_fraction, mass_flow_kg_s,
               chamber_pressure_pa, isp_s, total_impulse_ns, burn_time_s,
               thrust_coeff, motor_burning

References
----------
* G. P. Sutton and O. Biblarz, *Rocket Propulsion Elements*, 9th ed., Wiley
  2017 (St. Robert's law, chamber-pressure equilibrium, C_f, Isp, c*).
* A. Davenas, *Solid Rocket Propulsion Technology*, Pergamon 1993.
* N. Kubota, *Propellants and Explosives*, Wiley-VCH 2015.
"""

from __future__ import annotations

import math

from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var

G0 = 9.80665


class propulsion_sm_fmu(Fmi3Slave):
    """0-D internal-ballistics solid/liquid rocket motor with UQ + QoI."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Rocket motor (0-D internal ballistics, solid/liquid)"

        self.throttle = 1.0
        self.ambient_pressure_pa = 101325.0
        register_fmu3_var(self, "throttle", causality="input")
        register_fmu3_var(self, "ambient_pressure_pa", causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        self.thrust_n = 0.0
        self.thrust_sigma_n = 0.0
        self.propellant_fraction = 1.0
        self.mass_flow_kg_s = 0.0
        self.chamber_pressure_pa = 0.0
        self.isp_s = 0.0
        self.total_impulse_ns = 0.0
        self.burn_time_s = 0.0
        self.thrust_coeff = 0.0
        self.motor_burning = 0.0
        for _n in (
            "thrust_n", "thrust_sigma_n", "propellant_fraction", "mass_flow_kg_s",
            "chamber_pressure_pa", "isp_s", "total_impulse_ns", "burn_time_s",
            "thrust_coeff", "motor_burning",
        ):
            register_fmu3_var(self, _n, causality="output")

        # --- Configuration ----------------------------------------------------
        self.motor_type = "solid"                 # "solid" | "liquid"
        register_fmu3_param(self, "motor_type")
        self.propellant_mass_kg = 80.0
        register_fmu3_param(self, "propellant_mass_kg")

        # Grain / propellant (solid).  grain_type sets the A_b(web) trend:
        #   "neutral"     A_b constant (well-designed star/finocyl -> flat p_c)
        #   "progressive" internal tube (A_b grows with web)
        #   "regressive"  outer/rod burner (A_b shrinks with web)
        self.grain_type = "neutral"
        register_fmu3_param(self, "grain_type")
        self.grain_outer_radius_m = 0.09
        register_fmu3_param(self, "grain_outer_radius_m")
        self.grain_port_radius_m = 0.035
        register_fmu3_param(self, "grain_port_radius_m")
        self.propellant_density_kgm3 = 1800.0
        register_fmu3_param(self, "propellant_density_kgm3")
        self.burn_rate_coeff_a = 5.0e-5           # r = a * p_c^n  [m/s, p_c in Pa]
        register_fmu3_param(self, "burn_rate_coeff_a")
        self.burn_rate_exponent_n = 0.35
        register_fmu3_param(self, "burn_rate_exponent_n")
        self.c_star_mps = 1550.0                  # characteristic velocity
        register_fmu3_param(self, "c_star_mps")

        # Nozzle.  If auto_size_throat, the throat area is sized at init so the
        # initial burn area yields design_chamber_pressure_pa (standard motor
        # design practice); otherwise throat_area_m2 is used as given.
        self.auto_size_throat = 1.0
        register_fmu3_param(self, "auto_size_throat")
        self.design_chamber_pressure_pa = 9.0e6
        register_fmu3_param(self, "design_chamber_pressure_pa")
        self.throat_area_m2 = 0.0025
        register_fmu3_param(self, "throat_area_m2")
        self.expansion_ratio = 8.0                # Ae/At
        register_fmu3_param(self, "expansion_ratio")
        self.gamma_exhaust = 1.22
        register_fmu3_param(self, "gamma_exhaust")

        # Liquid-engine reference operating point.
        self.liquid_pc_ref_pa = 7.0e6
        register_fmu3_param(self, "liquid_pc_ref_pa")
        self.liquid_mdot_ref_kg_s = 12.0
        register_fmu3_param(self, "liquid_mdot_ref_kg_s")

        # UQ: relative 1-sigma of the dominant performance parameters.
        self.sigma_a_rel = 0.05                   # burn-rate coefficient
        register_fmu3_param(self, "sigma_a_rel")
        self.sigma_cstar_rel = 0.02               # c* efficiency
        register_fmu3_param(self, "sigma_cstar_rel")

        # --- Internal state ---------------------------------------------------
        self._prop_remaining_kg = 80.0
        self._web_burned_m = 0.0
        self._grain_length_m = 1.0
        self._burn_area0_m2 = 0.4
        self._cf_vacuum = 1.6
        self._exit_area_m2 = 0.02
        self._pe_over_pc = 0.02

    # ------------------------------------------------------------------ setup
    def enter_initialization_mode(self):
        self._prop_remaining_kg = self.propellant_mass_kg
        self._web_burned_m = 0.0
        self.total_impulse_ns = 0.0
        self.burn_time_s = 0.0
        # Derive grain length from loaded mass so geometry ~ configured load.
        r_o, r_p = self.grain_outer_radius_m, self.grain_port_radius_m
        area = math.pi * (r_o * r_o - r_p * r_p)
        vol = self.propellant_mass_kg / max(self.propellant_density_kgm3, 1e-6)
        self._grain_length_m = vol / max(area, 1e-9)
        self._burn_area0_m2 = 2.0 * math.pi * r_p * self._grain_length_m
        self._precompute_nozzle()
        # Size the throat so the initial burn area gives the design chamber
        # pressure: A_t = A_b0 * rho * a * c* / p_design^(1-n).
        if self.auto_size_throat > 0.5:
            n = self.burn_rate_exponent_n
            pc_d = max(self.design_chamber_pressure_pa, 1e3)
            self.throat_area_m2 = (
                self._burn_area0_m2 * self.propellant_density_kgm3
                * self.burn_rate_coeff_a * self.c_star_mps / pc_d ** (1.0 - n)
            )
        self._exit_area_m2 = self.expansion_ratio * self.throat_area_m2
        self.propellant_fraction = 1.0

    def exit_initialization_mode(self):
        pass

    def _precompute_nozzle(self) -> None:
        """Exit Mach / pressure ratio and vacuum C_f from the expansion ratio."""
        g = self.gamma_exhaust
        eps = max(self.expansion_ratio, 1.001)
        Me = self._area_ratio_to_mach(eps, g)
        self._pe_over_pc = (1.0 + 0.5 * (g - 1.0) * Me * Me) ** (-g / (g - 1.0))
        term = (2.0 * g * g / (g - 1.0)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
        self._cf_vacuum = math.sqrt(
            term * (1.0 - self._pe_over_pc ** ((g - 1.0) / g))
        ) + self._pe_over_pc * eps

    @staticmethod
    def _area_ratio_to_mach(eps: float, g: float) -> float:
        """Solve the isentropic area-Mach relation for the supersonic branch."""
        exp = (g + 1.0) / (2.0 * (g - 1.0))

        def area_ratio(M):
            return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** exp

        M = 2.5
        for _ in range(60):
            f = area_ratio(M) - eps
            dM = 1e-6
            df = (area_ratio(M + dM) - area_ratio(M - dM)) / (2 * dM)
            if abs(df) < 1e-12:
                break
            step = f / df
            M -= step
            M = min(max(M, 1.001), 20.0)
            if abs(step) < 1e-8:
                break
        return M

    # ------------------------------------------------------------------- step
    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        throttle = max(0.0, min(1.0, self.throttle))
        pa = max(self.ambient_pressure_pa, 0.0)

        if str(self.motor_type).strip().lower() == "liquid":
            pc, mdot = self._liquid(throttle)
        else:
            pc, mdot = self._solid(step_size)

        # Cap mass flow by remaining propellant.
        if mdot * step_size > self._prop_remaining_kg:
            mdot = self._prop_remaining_kg / step_size if step_size > 0 else 0.0
        if self._prop_remaining_kg <= 0.0:
            pc = mdot = 0.0

        # Thrust from C_f (ambient-corrected) * p_c * A_t.
        cf = self._cf(pc, pa)
        thrust = max(cf * pc * self.throat_area_m2, 0.0)

        self._prop_remaining_kg = max(0.0, self._prop_remaining_kg - mdot * step_size)
        self.mass_flow_kg_s = mdot
        self.chamber_pressure_pa = pc
        self.thrust_coeff = cf
        self.thrust_n = thrust
        self.isp_s = thrust / (mdot * G0) if mdot > 1e-9 else 0.0
        self.thrust_sigma_n = self._thrust_sigma(thrust)
        self.propellant_fraction = (
            self._prop_remaining_kg / self.propellant_mass_kg
            if self.propellant_mass_kg > 0.0 else 0.0
        )
        self.motor_burning = 1.0 if thrust > 1.0 else 0.0
        if thrust > 1.0:
            self.total_impulse_ns += thrust * step_size
            self.burn_time_s += step_size
        return True

    def terminate(self):
        print("Terminating propulsion_sm_fmu (0-D internal ballistics).")
        self.time = 0.0

    # ---------------------------------------------------------------- solid
    def _solid(self, dt: float):
        grain = str(self.grain_type).strip().lower()
        web = self.grain_outer_radius_m - self.grain_port_radius_m
        r_port = self.grain_port_radius_m + self._web_burned_m
        if grain == "progressive":
            if self._web_burned_m >= web:          # tube burned out to case
                return 0.0, 0.0
            A_b = 2.0 * math.pi * r_port * self._grain_length_m
        elif grain == "regressive":
            if self._web_burned_m >= web:
                return 0.0, 0.0
            A_b = 2.0 * math.pi * max(self.grain_outer_radius_m - self._web_burned_m,
                                      1e-4) * self._grain_length_m
        else:  # neutral: design-flat burn area; burnout is mass-driven (do_step)
            A_b = self._burn_area0_m2
        Kn = A_b / max(self.throat_area_m2, 1e-9)
        n = self.burn_rate_exponent_n
        base = (self.propellant_density_kgm3 * self.burn_rate_coeff_a
                * self.c_star_mps * Kn)
        pc = base ** (1.0 / (1.0 - n)) if base > 0 else 0.0
        r_dot = self.burn_rate_coeff_a * pc ** n
        mdot = self.propellant_density_kgm3 * A_b * r_dot
        self._web_burned_m += r_dot * dt
        return pc, mdot

    # --------------------------------------------------------------- liquid
    def _liquid(self, throttle: float):
        pc = throttle * self.liquid_pc_ref_pa
        mdot = throttle * self.liquid_mdot_ref_kg_s
        return pc, mdot

    # --------------------------------------------------------------- nozzle
    def _cf(self, pc: float, pa: float) -> float:
        if pc <= 1.0:
            return 0.0
        # Ambient correction: subtract (pa/pc)*eps from the vacuum coefficient.
        return max(self._cf_vacuum - (pa / pc) * self.expansion_ratio, 0.0)

    # ------------------------------------------------------------------- UQ
    def _thrust_sigma(self, thrust: float) -> float:
        """First-order thrust 1-sigma from a and c* relative uncertainties.

        For a solid, p_c ~ (a c*)^(1/(1-n)) and F ~ p_c, so
        dF/F ~ 1/(1-n) * (da/a + dc*/c*).  For a liquid, F ~ c* mdot with mdot
        set by throttle, so dF/F ~ dc*/c*."""
        if thrust <= 0.0:
            return 0.0
        if str(self.motor_type).strip().lower() == "liquid":
            rel = self.sigma_cstar_rel
        else:
            k = 1.0 / max(1.0 - self.burn_rate_exponent_n, 1e-3)
            rel = k * math.sqrt(self.sigma_a_rel ** 2 + self.sigma_cstar_rel ** 2)
        return abs(thrust) * rel
