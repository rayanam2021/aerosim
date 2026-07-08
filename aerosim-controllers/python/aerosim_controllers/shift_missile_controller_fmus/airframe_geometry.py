"""
Modular airframe geometry for SHIFT-class interceptors.

Captures a configurable **canard + tail** layout (default: 4 longer forward
canards + 4 taller aft fins, diamond cross-section, trapezoidal planform) and
derives the equivalent body-axis aerodynamic control derivatives used by the
analytic plant and the autopilot.

Why this exists
---------------
The GNC stack still commands logical elevator / aileron / rudder (skid-to-turn),
but the plant must know that those commands are mixed onto two physical fin
banks with different areas and lever arms.  Keeping the geometry in one module
lets a future missile swap canard/tail sizes, counts, or stations without
rewriting the aero or autopilot FMUs.

Planform model (trapezoidal)
----------------------------
Each fin bank is described by root chord ``c_root``, tip chord ``c_tip``,
semi-span ``b_semi`` (from body surface to tip), and axial station ``x_m``
(nose = 0).  Planform area of one fin::

    S_fin = 0.5 * (c_root + c_tip) * b_semi

Diamond cross-section thickness enters only as a parasitic-drag increment
(``t_over_c``), not as a full panel CFD model.

Control mixing (X-config, 4 fins per bank)
------------------------------------------
Logical commands map to bank-average deflections:

    elevator  -> pitch pair on the *primary* bank (default: canards)
    rudder    -> yaw pair on the primary bank
    aileron   -> differential roll on the *roll* bank (default: canards;
                 set ``roll_bank="tail"`` to use the taller aft fins)

Equivalent derivatives (per radian of logical command) are then::

    CN_de  ~  n_eff * (S_bank / S_ref) * CLalpha_fin * k_pitch
    Cm_de  ~  CN_de * (x_cg - x_bank) / d_ref     (sign: nose-up +)
    ... similarly for CY_dr, Cn_dr, Cl_da

Body static derivatives (CN_alpha, Cm_alpha, ...) remain separate parameters so
the body + fin-fixed contribution can be tuned independently of control
geometry.

References
----------
* Nielsen, *Missile Aerodynamics*, AIAA 1988 (fin effectiveness, interference).
* Blake, *Missile DATCOM User's Manual*, AFRL (trapezoidal fin geometry).
* Zipfel, *Modeling and Simulation of Aerospace Vehicle Dynamics*, AIAA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Mapping


@dataclass(frozen=True)
class FinBank:
    """One bank of identical fins (canards or tails)."""

    name: str
    n_fins: int = 4
    # Trapezoidal planform (metres).
    c_root_m: float = 0.20
    c_tip_m: float = 0.08
    b_semi_m: float = 0.18          # span from body surface to tip
    x_m: float = 0.55               # axial station of MAC, nose = 0
    # Diamond cross-section thickness / chord (parasitic drag only).
    t_over_c: float = 0.08
    # Fin lift-curve slope per rad (isolated; interference applied later).
    cl_alpha_per_rad: float = 3.5
    # Interference / body-carryover factor (0..1+).
    interference: float = 0.85

    @property
    def area_one_m2(self) -> float:
        return 0.5 * (self.c_root_m + self.c_tip_m) * self.b_semi_m

    @property
    def area_bank_m2(self) -> float:
        return self.n_fins * self.area_one_m2

    @property
    def mac_m(self) -> float:
        """Mean aerodynamic chord of the trapezoidal planform."""
        cr, ct = self.c_root_m, self.c_tip_m
        if cr + ct <= 0.0:
            return 0.0
        return (2.0 / 3.0) * (cr + ct - cr * ct / (cr + ct))


@dataclass(frozen=True)
class AirframeGeometry:
    """Full interceptor outer-mold-line summary used by aero / structures."""

    body_length_m: float = 3.5
    body_diameter_m: float = 0.20
    # CG used for moment arms (full-propellant default; structures updates live).
    cg_x_m: float = 1.75
    # Which bank is the primary pitch/yaw effector and which does roll.
    pitch_yaw_bank: str = "canard"   # "canard" | "tail"
    roll_bank: str = "canard"        # "canard" | "tail"
    # Effective fins participating in a pitch (or yaw) command for an X-config
    # 4-fin bank: two fins carry most of the load.
    n_eff_pitch: float = 2.0
    n_eff_roll: float = 2.0
    canard: FinBank = FinBank(
        name="canard",
        n_fins=4,
        c_root_m=0.28,      # longer chord ("longer canards")
        c_tip_m=0.10,
        b_semi_m=0.16,
        x_m=0.55,
        t_over_c=0.08,
        cl_alpha_per_rad=3.5,
        interference=0.90,
    )
    tail: FinBank = FinBank(
        name="tail",
        n_fins=4,
        c_root_m=0.18,
        c_tip_m=0.07,
        b_semi_m=0.22,      # taller span ("taller fins")
        x_m=3.15,
        t_over_c=0.08,
        cl_alpha_per_rad=3.5,
        interference=0.85,
    )

    @property
    def ref_diameter_m(self) -> float:
        return self.body_diameter_m

    @property
    def ref_area_m2(self) -> float:
        return 0.25 * math.pi * self.body_diameter_m ** 2

    def bank(self, name: str) -> FinBank:
        key = str(name).strip().lower()
        if key.startswith("can"):
            return self.canard
        if key.startswith("tai") or key.startswith("fin"):
            return self.tail
        raise KeyError(f"unknown fin bank {name!r}; expected 'canard' or 'tail'")


def geometry_from_params(p: Mapping[str, float | int | str]) -> AirframeGeometry:
    """Build an ``AirframeGeometry`` from a flat FMU-parameter dict.

    Missing keys fall back to the SHIFT defaults.  Nested bank fields use the
    prefixes ``canard_`` and ``tail_`` (e.g. ``canard_c_root_m``).
    """
    def _f(key, default):
        return float(p[key]) if key in p else float(default)

    def _i(key, default):
        return int(p[key]) if key in p else int(default)

    def _s(key, default):
        return str(p[key]) if key in p else str(default)

    def _bank(prefix: str, defaults: FinBank) -> FinBank:
        return FinBank(
            name=defaults.name,
            n_fins=_i(f"{prefix}n_fins", defaults.n_fins),
            c_root_m=_f(f"{prefix}c_root_m", defaults.c_root_m),
            c_tip_m=_f(f"{prefix}c_tip_m", defaults.c_tip_m),
            b_semi_m=_f(f"{prefix}b_semi_m", defaults.b_semi_m),
            x_m=_f(f"{prefix}x_m", defaults.x_m),
            t_over_c=_f(f"{prefix}t_over_c", defaults.t_over_c),
            cl_alpha_per_rad=_f(f"{prefix}cl_alpha_per_rad", defaults.cl_alpha_per_rad),
            interference=_f(f"{prefix}interference", defaults.interference),
        )

    base = AirframeGeometry()
    return AirframeGeometry(
        body_length_m=_f("body_length_m", base.body_length_m),
        body_diameter_m=_f("body_diameter_m", base.body_diameter_m),
        cg_x_m=_f("cg_x_m", base.cg_x_m),
        pitch_yaw_bank=_s("pitch_yaw_bank", base.pitch_yaw_bank),
        roll_bank=_s("roll_bank", base.roll_bank),
        n_eff_pitch=_f("n_eff_pitch", base.n_eff_pitch),
        n_eff_roll=_f("n_eff_roll", base.n_eff_roll),
        canard=_bank("canard_", base.canard),
        tail=_bank("tail_", base.tail),
    )


def derive_control_derivatives(geom: AirframeGeometry) -> dict:
    """Equivalent body-axis control derivatives from the fin geometry.

    Returns a dict suitable for writing into ``CN_de``, ``Cm_de``, ``CY_dr``,
    ``Cn_dr``, ``Cl_da``, plus diagnostic areas / arms.  Body static derivatives
    (``CN_alpha``, ``Cm_alpha``, ...) are *not* overwritten here.
    """
    S = max(geom.ref_area_m2, 1e-9)
    d = max(geom.ref_diameter_m, 1e-9)
    cg = geom.cg_x_m

    py = geom.bank(geom.pitch_yaw_bank)
    rl = geom.bank(geom.roll_bank)

    # Pitch/yaw: force coefficient per rad of logical elevator/rudder.
    # Two of four X-fins dominate a pure pitch (or yaw) command.
    cn_de = (geom.n_eff_pitch * py.area_one_m2 / S
             * py.cl_alpha_per_rad * py.interference)
    # Moment arm: positive (x_cg - x_fin) means fin ahead of CG (canard) produces
    # nose-up moment for positive normal force.  Autopilot / plant convention:
    # Cm_de < 0 (positive elevator -> nose-down moment for a *tail*).  Canards
    # reverse the physical arm, so we apply ``-sign(arm)`` to keep Cm_de
    # negative and the existing trim law
    # ``de_ff = -Cm_alpha/Cm_de * alpha_cmd`` valid for either bank.
    arm_py = (cg - py.x_m) / d
    cm_de = -abs(arm_py) * cn_de if abs(arm_py) > 1e-9 else -cn_de

    # Yaw mirrors pitch for an axisymmetric airframe.
    cy_dr = cn_de
    cn_dr = cm_de

    # Roll: differential deflection on the roll bank.  Roll moment arm ~ mean
    # span of the fin (body radius + half semi-span).
    body_r = 0.5 * geom.body_diameter_m
    y_arm = body_r + 0.5 * rl.b_semi_m
    cl_da = -(geom.n_eff_roll * rl.area_one_m2 / S
              * rl.cl_alpha_per_rad * rl.interference * (y_arm / d))

    # Parasitic axial-force increment from diamond-section fin thickness
    # (order-of-magnitude; body CA0 remains the dominant term).
    ca_fins = 0.0
    for bank in (geom.canard, geom.tail):
        ca_fins += bank.area_bank_m2 / S * (bank.t_over_c ** 2)

    return {
        "CN_de": float(cn_de),
        "Cm_de": float(cm_de),
        "CY_dr": float(cy_dr),
        "Cn_dr": float(cn_dr),
        "Cl_da": float(cl_da),
        "CA_fins": float(ca_fins),
        "ref_area_m2": float(S),
        "ref_diameter_m": float(d),
        "canard_area_bank_m2": float(geom.canard.area_bank_m2),
        "tail_area_bank_m2": float(geom.tail.area_bank_m2),
        "canard_arm_d": float((cg - geom.canard.x_m) / d),
        "tail_arm_d": float((cg - geom.tail.x_m) / d),
        "pitch_yaw_bank": geom.pitch_yaw_bank,
        "roll_bank": geom.roll_bank,
    }


def geometry_param_defaults() -> dict:
    """Flat parameter dict (SHIFT defaults) for FMU ``register_fmu3_param``."""
    g = AirframeGeometry()
    out = {
        "body_length_m": g.body_length_m,
        "body_diameter_m": g.body_diameter_m,
        "cg_x_m": g.cg_x_m,
        "pitch_yaw_bank": g.pitch_yaw_bank,
        "roll_bank": g.roll_bank,
        "n_eff_pitch": g.n_eff_pitch,
        "n_eff_roll": g.n_eff_roll,
        "use_geometry_aero": 1.0,
    }
    for prefix, bank in (("canard_", g.canard), ("tail_", g.tail)):
        d = asdict(bank)
        d.pop("name", None)
        for k, v in d.items():
            out[f"{prefix}{k}"] = v
    return out
