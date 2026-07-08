"""
Uncertainty-quantification and probability-of-kill helpers.

This module collects the small, dependency-light statistical primitives shared by
the SHIFT interceptor stack:

* Lethality / damage functions mapping a miss distance to a conditional kill
  probability (cookie-cutter and Carleton diffuse-Gaussian models).
* Single-shot probability of kill (SSPK / P_kill) estimators, both closed-form
  (Rayleigh miss + Carleton damage) and Monte-Carlo.
* Binomial confidence bounds (Wilson score interval) for reporting P_kill with
  uncertainty when it is estimated from a finite Monte-Carlo sample.
* Simple first-order (linearized) error propagation and sampling utilities used
  to turn surrogate-model uncertainty into engagement-outcome uncertainty.

References
----------
* R. E. Ball, *The Fundamentals of Aircraft Combat Survivability Analysis and
  Design*, 2nd ed., AIAA 2003 (Carleton damage function, P_kill, lethal radius).
* D. A. Wilkening, "A simple model for calculating ballistic missile defense
  effectiveness," *Science & Global Security* 8(2):183-215, 2000 (SSPK).
* E. B. Wilson, "Probable inference, the law of succession, and statistical
  inference," *JASA* 22:209-212, 1927 (Wilson score interval).
* A. Gelman et al., *Bayesian Data Analysis*, 3rd ed., CRC 2013 (MC estimation).
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

LN2 = math.log(2.0)


# ── Damage / lethality functions ─────────────────────────────────────────────
def carleton_b_from_lethal_radius(lethal_radius_m: float) -> float:
    """Carleton lethality parameter ``b`` such that P_kill(lethal_radius)=0.5."""
    return float(lethal_radius_m) / math.sqrt(2.0 * LN2)


def pk_given_miss(miss_m, lethal_radius_m: float, model: str = "carleton") -> np.ndarray:
    """Conditional kill probability given a scalar/array miss distance.

    ``carleton``: diffuse-Gaussian damage, ``P = exp(-r^2 / (2 b^2))`` with ``b``
    chosen so ``P(lethal_radius)=0.5``.  ``cookie`` (a.k.a. cookie-cutter):
    ``P = 1`` inside the lethal radius, else ``0``.
    """
    r = np.abs(np.asarray(miss_m, dtype=float))
    if str(model).lower().startswith("cookie"):
        return (r <= lethal_radius_m).astype(float)
    b = carleton_b_from_lethal_radius(lethal_radius_m)
    return np.exp(-(r ** 2) / (2.0 * b ** 2))


# ── Closed-form single-shot P_kill ───────────────────────────────────────────
def sspk_rayleigh_carleton(sigma_miss_m: float, lethal_radius_m: float) -> float:
    """Closed-form SSPK for a circular-Gaussian miss and Carleton damage.

    With a 2-D miss whose per-axis standard deviation is ``sigma_miss_m`` and the
    Carleton damage function, the SSPK integrates to ``b^2 / (b^2 + sigma^2)``.
    """
    b2 = carleton_b_from_lethal_radius(lethal_radius_m) ** 2
    s2 = float(sigma_miss_m) ** 2
    return float(b2 / (b2 + s2)) if (b2 + s2) > 0 else 0.0


# ── Monte-Carlo single-shot P_kill + confidence bounds ───────────────────────
def sspk_monte_carlo(
    miss_distances_m: Sequence[float],
    lethal_radius_m: float,
    model: str = "carleton",
    confidence: float = 0.95,
) -> dict:
    """Estimate SSPK and a confidence interval from Monte-Carlo miss distances.

    For the cookie-cutter model the outcome per run is Bernoulli (hit inside the
    lethal radius) and the Wilson score interval is exact-ish and reported.  For
    the Carleton model each run contributes a continuous kill probability; the
    point estimate is their mean and the interval is a normal approximation of
    the mean plus the Wilson interval on the implied binomial for reference.
    """
    miss = np.asarray(miss_distances_m, dtype=float)
    miss = miss[np.isfinite(miss)]
    n = miss.size
    if n == 0:
        return {"pkill": 0.0, "low": 0.0, "high": 0.0, "n": 0, "model": model}

    pk = pk_given_miss(miss, lethal_radius_m, model)
    p_hat = float(np.mean(pk))

    if str(model).lower().startswith("cookie"):
        k = int(np.sum(pk))
        low, high = wilson_interval(k, n, confidence)
    else:
        # Normal (CLT) interval on the mean of the per-run kill probabilities.
        z = _z_score(confidence)
        se = float(np.std(pk, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        low = max(0.0, p_hat - z * se)
        high = min(1.0, p_hat + z * se)

    return {
        "pkill": p_hat,
        "low": low,
        "high": high,
        "n": n,
        "model": model,
        "cep_m": float(np.percentile(miss, 50.0)),
        "mean_miss_m": float(np.mean(miss)),
        "sigma_miss_m": float(np.std(miss)),
    }


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = _z_score(confidence)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _z_score(confidence: float) -> float:
    """Two-sided normal z for a confidence level (rational approx of probit)."""
    p = 0.5 + confidence / 2.0
    # Acklam's inverse-normal approximation.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── Sampling / error propagation ─────────────────────────────────────────────
def sample_params(
    spec: Mapping[str, object],
    rng: np.random.Generator,
) -> dict:
    """Draw a parameter dictionary from an uncertainty spec.

    Each value in ``spec`` is either a scalar (deterministic) or a mapping with a
    ``distribution`` key: ``normal`` (``mean``,``std``), ``uniform``
    (``low``,``high``) or ``lognormal`` (``mean``,``std`` of the underlying
    normal).  Returns a flat dict of realized values.
    """
    out: dict = {}
    for key, val in spec.items():
        if isinstance(val, Mapping) and "distribution" in val:
            dist = str(val["distribution"]).lower()
            if dist == "normal":
                out[key] = float(rng.normal(val["mean"], val.get("std", 0.0)))
            elif dist == "uniform":
                out[key] = float(rng.uniform(val["low"], val["high"]))
            elif dist == "lognormal":
                out[key] = float(rng.lognormal(val["mean"], val.get("std", 0.0)))
            else:
                out[key] = float(val.get("mean", 0.0))
        else:
            out[key] = val
    return out


def propagate_linear(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    cov_x: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order (delta-method) mean/covariance propagation of ``y = f(x)``.

    Returns ``(y0, cov_y)`` with ``cov_y = J cov_x J'`` and ``J`` from central
    finite differences.  Cheap surrogate-uncertainty propagation when a full
    Monte-Carlo pass is too expensive.
    """
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(f(x0), dtype=float)
    n, m = x0.size, y0.size
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (np.asarray(f(x0 + dx)) - np.asarray(f(x0 - dx))) / (2 * eps)
    return y0, J @ np.asarray(cov_x, dtype=float) @ J.T
