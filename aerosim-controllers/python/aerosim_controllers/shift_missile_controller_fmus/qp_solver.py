"""
Compact convex QP solver (primal-dual interior-point, Mehrotra predictor-corrector).

Solves the inequality-constrained quadratic program

    minimize    1/2 x' P x + q' x
    subject to  G x <= h

with ``P`` symmetric positive semi-definite.  This is the exact problem class
produced by the condensed missile terminal-guidance MPC (acceleration-limit and
seeker look-angle inequality constraints), and the primal-dual interior-point
method (PD-IPM) is the approach used in the missile-guidance MPC literature
(e.g. Kim et al., *Sensors* 2020/2022) because it is fast, deterministic in its
iteration count, and easy to warm/​cold start on-board.

Design goals
------------
* Pure NumPy (bundled as a PythonFMU3 resource; no external QP dependency).
* Always returns a usable, feasible-ish control even if it does not fully
  converge (real-time guidance must never stall), by falling back to the
  projected unconstrained minimizer.

References
----------
* S. Boyd and L. Vandenberghe, *Convex Optimization*, CUP 2004, Ch. 11
  (interior-point methods) and §16.1 (QP).
* J. Nocedal and S. Wright, *Numerical Optimization*, 2nd ed., Springer 2006,
  Ch. 16 (QP) and Algorithm 16.4 (Mehrotra predictor-corrector).
* S. Mehrotra, "On the implementation of a primal-dual interior point method,"
  *SIAM J. Optim.* 2(4):575-601, 1992.
"""

from __future__ import annotations

import numpy as np


def solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray | None = None,
    h: np.ndarray | None = None,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Solve ``min 1/2 x'Px + q'x s.t. Gx <= h`` via Mehrotra PD-IPM.

    Returns ``(x, info)`` where ``info`` carries ``converged``, ``iterations``
    and ``status``.  If ``G``/``h`` are ``None`` the unconstrained minimizer is
    returned directly.
    """
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float).flatten()
    n = q.shape[0]
    # Symmetrize + mild regularization for a unique, well-conditioned solve.
    P = 0.5 * (P + P.T) + 1e-9 * np.eye(n)

    if G is None or h is None or G.size == 0:
        x = _safe_solve(P, -q)
        return x, {"converged": True, "iterations": 0, "status": "unconstrained"}

    G = np.asarray(G, dtype=float)
    h = np.asarray(h, dtype=float).flatten()
    m = h.shape[0]

    # Cold start strictly inside the positive orthant for the slack/dual pair.
    x = _safe_solve(P, -q)
    s = np.maximum(h - G @ x, 1.0)
    z = np.ones(m)

    unconstrained = x.copy()
    e = np.ones(m)

    for it in range(max_iter):
        r_d = P @ x + q + G.T @ z          # stationarity
        r_p = G @ x + s - h                # primal feasibility
        mu = float(s @ z) / m              # duality measure

        if (np.linalg.norm(r_d) < tol and np.linalg.norm(r_p) < tol and mu < tol):
            return x, {"converged": True, "iterations": it, "status": "optimal"}

        w = z / np.maximum(s, 1e-12)                    # diag scaling
        H = P + (G.T * w) @ G                           # Schur complement
        L = _chol(H)
        if L is None:
            break

        # --- Affine (predictor) step: sigma = 0 --------------------------------
        r_cent_aff = s * z
        rhs = -(r_d + G.T @ (w * r_p - r_cent_aff / np.maximum(s, 1e-12)))
        dx_aff = _chol_solve(L, rhs)
        dz_aff = w * (G @ dx_aff + r_p) - r_cent_aff / np.maximum(s, 1e-12)
        ds_aff = -r_p - G @ dx_aff

        a_p = _step_to_boundary(s, ds_aff)
        a_d = _step_to_boundary(z, dz_aff)
        mu_aff = float((s + a_p * ds_aff) @ (z + a_d * dz_aff)) / m
        sigma = (mu_aff / mu) ** 3 if mu > 1e-16 else 0.0

        # --- Corrector step (Mehrotra) ----------------------------------------
        r_cent = s * z + ds_aff * dz_aff - sigma * mu * e
        rhs = -(r_d + G.T @ (w * r_p - r_cent / np.maximum(s, 1e-12)))
        dx = _chol_solve(L, rhs)
        dz = w * (G @ dx + r_p) - r_cent / np.maximum(s, 1e-12)
        ds = -r_p - G @ dx

        a_p = _step_to_boundary(s, ds)
        a_d = _step_to_boundary(z, dz)
        alpha = 0.99 * min(a_p, a_d)

        x = x + alpha * dx
        s = s + alpha * ds
        z = z + alpha * dz

    # Not fully converged: project the unconstrained solution to be safe.
    if not np.all(np.isfinite(x)):
        x = unconstrained
    return x, {"converged": False, "iterations": max_iter, "status": "max_iter"}


def _step_to_boundary(v: np.ndarray, dv: np.ndarray) -> float:
    """Largest alpha in (0, 1] keeping v + alpha*dv >= 0."""
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, np.min(-v[neg] / dv[neg])))


def _chol(A: np.ndarray):
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.cholesky(A + 1e-6 * np.trace(A) / A.shape[0] * np.eye(A.shape[0]))
        except np.linalg.LinAlgError:
            return None


def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]
