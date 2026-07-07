"""
ICAO International Standard Atmosphere (ISA) model.

Covers four standard layers from sea level to 47 km MSL — the full range
relevant for supersonic missile flight profiles.  Below sea level the sea-level
values are returned unchanged; above 47 km the stratospheric trend is linearly
extrapolated (adequate for off-nominal edge cases without crashing).

Reference: ICAO Doc 7488 / Manual of the ICAO Standard Atmosphere (3rd ed.)

All inputs/outputs use SI units unless otherwise noted.
"""

from __future__ import annotations

import math

# ── Physical constants ──────────────────────────────────────────────────────
GAMMA_AIR: float = 1.4          # ratio of specific heats for dry air
R_AIR: float = 287.05287        # specific gas constant for dry air [J/(kg·K)]
G0: float = 9.80665             # standard gravity [m/s²]

# ── ISA layer definitions ───────────────────────────────────────────────────
# Each layer: (base altitude [m], base temperature [K], lapse rate [K/m])
# Negative lapse rate = temperature decreasing with altitude (troposphere).
# Zero lapse rate     = isothermal layer (lower stratosphere).
# Positive lapse rate = temperature increasing with altitude (inversion).
_LAYERS: list[tuple[float, float, float]] = [
    (0.0,     288.15, -0.0065),   # troposphere
    (11_000.0, 216.65,  0.0),     # lower stratosphere (isothermal)
    (20_000.0, 216.65, +0.0010),  # middle stratosphere
    (32_000.0, 228.65, +0.0028),  # upper stratosphere
    (47_000.0, 270.65,  0.0),     # sentinel / extrapolation cap
]

# Pre-compute base pressures for each layer boundary using the barometric
# formula, so that per-call evaluation only needs one layer multiplication.
def _compute_base_pressures() -> list[float]:
    p0 = 101_325.0  # sea-level pressure [Pa]
    pressures = [p0]
    for i in range(len(_LAYERS) - 1):
        h_base, T_base, L = _LAYERS[i]
        h_top, T_top, _  = _LAYERS[i + 1]
        dh = h_top - h_base
        if abs(L) < 1e-12:  # isothermal layer
            p_top = pressures[-1] * math.exp(-G0 * dh / (R_AIR * T_base))
        else:
            p_top = pressures[-1] * (T_top / T_base) ** (G0 / (R_AIR * (-L)))
        pressures.append(p_top)
    return pressures

_BASE_PRESSURES: list[float] = _compute_base_pressures()


def isa(altitude_m: float) -> tuple[float, float, float, float]:
    """Return ISA values at *altitude_m* meters above mean sea level.

    Parameters
    ----------
    altitude_m : geometric altitude [m MSL]

    Returns
    -------
    temperature_K : static temperature [K]
    pressure_Pa   : static pressure [Pa]
    density_kgm3  : air density [kg/m³]
    speed_of_sound_mps : speed of sound [m/s]
    """
    h = max(0.0, altitude_m)  # clamp: treat below-sea-level as sea level

    # Find the layer that contains h.
    layer_idx = 0
    for i in range(len(_LAYERS) - 1):
        if h >= _LAYERS[i][0]:
            layer_idx = i
        else:
            break

    h_base, T_base, L = _LAYERS[layer_idx]
    p_base = _BASE_PRESSURES[layer_idx]
    dh = h - h_base

    if abs(L) < 1e-12:  # isothermal
        T = T_base
        P = p_base * math.exp(-G0 * dh / (R_AIR * T_base))
    else:
        T = T_base + L * dh
        T = max(T, 1.0)  # guard against unphysical T at extreme altitudes
        P = p_base * (T / T_base) ** (G0 / (R_AIR * (-L)))

    rho = P / (R_AIR * T)
    a = math.sqrt(GAMMA_AIR * R_AIR * T)
    return T, P, rho, a


def temperature_K(altitude_m: float) -> float:
    """Static temperature [K] at the given altitude."""
    return isa(altitude_m)[0]


def pressure_Pa(altitude_m: float) -> float:
    """Static pressure [Pa] at the given altitude."""
    return isa(altitude_m)[1]


def density_kgm3(altitude_m: float) -> float:
    """Air density [kg/m³] at the given altitude."""
    return isa(altitude_m)[2]


def speed_of_sound_mps(altitude_m: float) -> float:
    """Speed of sound [m/s] at the given altitude."""
    return isa(altitude_m)[3]


def mach(speed_mps: float, altitude_m: float) -> float:
    """Mach number for *speed_mps* at *altitude_m*."""
    a = speed_of_sound_mps(altitude_m)
    return speed_mps / max(a, 1e-3)


def dynamic_pressure_Pa(speed_mps: float, altitude_m: float) -> float:
    """Dynamic pressure q = ½ ρ V² [Pa] at the given speed and altitude."""
    rho = density_kgm3(altitude_m)
    return 0.5 * rho * speed_mps * speed_mps


def pressure_altitude_m(pressure_Pa_value: float) -> float:
    """Inverse of pressure_Pa: altitude [m] that corresponds to a given
    static pressure.  Useful for converting a barometric pressure reading
    into pressure altitude without temperature correction (as a real
    altimeter does in ISA-standard conditions).

    Pressure decreases monotonically with altitude, so we find the highest
    layer index whose base pressure is still >= the given pressure.
    """
    # _BASE_PRESSURES are strictly decreasing with index (high P at sea level,
    # low P at altitude). We want the largest i such that _BASE_PRESSURES[i]
    # >= pressure_Pa_value, which is the layer the pressure falls within.
    layer_idx = 0
    for i in range(len(_LAYERS) - 1):
        if _BASE_PRESSURES[i] >= pressure_Pa_value:
            layer_idx = i
        else:
            break

    h_base, T_base, L = _LAYERS[layer_idx]
    p_base = _BASE_PRESSURES[layer_idx]
    if abs(L) < 1e-12:  # isothermal
        dh = -R_AIR * T_base / G0 * math.log(pressure_Pa_value / p_base)
    else:
        dh = (T_base / L) * ((pressure_Pa_value / p_base) ** (R_AIR * (-L) / G0) - 1.0)
    return h_base + dh  # at or above sea level


if __name__ == "__main__":
    # Quick self-test against tabulated ICAO values.
    _checks = [
        (0,      288.15, 101_325.0, 1.2250,  340.29),
        (5_000,  255.68,  54_020.0, 0.7364,  320.53),
        (11_000, 216.65,  22_632.0, 0.3639,  295.07),
        (20_000, 216.65,   5_474.9, 0.0880,  295.07),
    ]
    print(f"{'Alt':>8}  {'T':>8}  {'P':>10}  {'rho':>7}  {'a':>7}")
    for h, T_ref, P_ref, rho_ref, a_ref in _checks:
        T, P, rho, a = isa(h)
        # Allow 0.1 K: ICAO tables use geopotential altitude; we use geometric.
        # The difference is < 0.05 K at 5000 m and < 0.01% in speed of sound.
        ok = (
            abs(T - T_ref) < 0.1
            and abs(P - P_ref) / P_ref < 1e-3
            and abs(rho - rho_ref) / rho_ref < 2e-3
            and abs(a - a_ref) < 0.1
        )
        print(f"{h:>8.0f}  {T:8.2f}  {P:10.1f}  {rho:7.4f}  {a:7.2f}  {'OK' if ok else 'FAIL'}")
