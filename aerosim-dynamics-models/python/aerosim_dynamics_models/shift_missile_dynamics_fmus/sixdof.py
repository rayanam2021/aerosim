"""
Shared 6-DOF rigid-body dynamics core (quaternion attitude) for the SHIFT
missile FMUs.

Frames
------
NED   : North-East-Down world frame (right-handed, z down).
Body  : FRD  (x forward, y right, z down), origin at the CG.

Quaternion convention
---------------------
Quaternions are stored scalar-LAST ``[x, y, z, w]`` to match ``scipy`` and the
AeroSim ``Orientation`` message (which stores w,x,y,z but we convert on I/O).
``q`` represents the body->NED rotation, i.e. ``v_ned = R(q) @ v_body``.

State vector layout (used by plants and EKFs) — 13 elements:
    [ pN pE pD  vN vE vD  qx qy qz qw  wx wy wz ]
      position   velocity   attitude quat   body rates

The functions here are deliberately dependency-light (numpy + scipy Rotation) so
each FMU can bundle this single file as a PythonFMU3 project resource.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

GRAVITY = 9.80665  # m/s^2


# ── Quaternion / rotation helpers ────────────────────────────────────────────
def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalise a scalar-last quaternion; fall back to identity if degenerate."""
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def rot_from_quat(q_xyzw: np.ndarray) -> Rotation:
    return Rotation.from_quat(quat_normalize(np.asarray(q_xyzw, dtype=float)))


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3-2-1 (yaw, pitch, roll) Euler -> scalar-last quaternion (body->NED)."""
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()


def euler_from_quat(q_xyzw: np.ndarray) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) in radians from a scalar-last quaternion."""
    yaw, pitch, roll = rot_from_quat(q_xyzw).as_euler("ZYX")
    return float(roll), float(pitch), float(yaw)


def body_to_ned(q_xyzw: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    return rot_from_quat(q_xyzw).apply(np.asarray(v_body, dtype=float))


def ned_to_body(q_xyzw: np.ndarray, v_ned: np.ndarray) -> np.ndarray:
    return rot_from_quat(q_xyzw).inv().apply(np.asarray(v_ned, dtype=float))


def quat_to_msg(q_xyzw: np.ndarray) -> tuple[float, float, float, float]:
    """Return (w, x, y, z) for populating an AeroSim Orientation message."""
    q = quat_normalize(q_xyzw)
    return float(q[3]), float(q[0]), float(q[1]), float(q[2])


def quat_from_msg(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Build a scalar-last quaternion from an AeroSim (w,x,y,z) orientation."""
    if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quat_normalize(np.array([x, y, z, w], dtype=float))


# ── Aerodynamic angles ────────────────────────────────────────────────────────
def alpha_beta(v_body: np.ndarray) -> tuple[float, float]:
    """Angle of attack and sideslip [rad] from body-frame velocity (FRD)."""
    u, v, w = float(v_body[0]), float(v_body[1]), float(v_body[2])
    speed = np.linalg.norm(v_body)
    if speed < 1e-3:
        return 0.0, 0.0
    alpha = np.arctan2(w, u)          # nose-up positive
    beta = np.arcsin(np.clip(v / speed, -1.0, 1.0))  # nose-right positive
    return float(alpha), float(beta)


# ── Rigid-body integration ────────────────────────────────────────────────────
def integrate_6dof(
    pos_ned: np.ndarray,
    vel_ned: np.ndarray,
    q_xyzw: np.ndarray,
    omega_body: np.ndarray,
    force_body: np.ndarray,
    moment_body: np.ndarray,
    mass: float,
    inertia_diag: np.ndarray,
    dt: float,
    gravity: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Semi-implicit Euler step of the full 6-DOF rigid-body equations.

    Returns (pos_ned, vel_ned, q_xyzw, omega_body, accel_ned) updated by ``dt``.
    ``inertia_diag`` is the body-frame principal inertia [Ixx, Iyy, Izz].
    """
    mass = max(float(mass), 1e-6)
    I = np.maximum(np.asarray(inertia_diag, dtype=float), 1e-6)

    rot = rot_from_quat(q_xyzw)

    # Translational dynamics in NED.
    f_ned = rot.apply(np.asarray(force_body, dtype=float))
    g_ned = np.array([0.0, 0.0, GRAVITY * mass]) if gravity else np.zeros(3)
    accel_ned = (f_ned + g_ned) / mass

    new_vel = np.asarray(vel_ned, dtype=float) + accel_ned * dt
    new_pos = np.asarray(pos_ned, dtype=float) + new_vel * dt

    # Rotational dynamics (Euler's equations) in body frame.
    w = np.asarray(omega_body, dtype=float)
    Iw = I * w
    ang_accel = (np.asarray(moment_body, dtype=float) - np.cross(w, Iw)) / I
    new_omega = w + ang_accel * dt

    # Attitude update: right-multiply by the body-rate rotation increment.
    delta = Rotation.from_rotvec(new_omega * dt)
    new_rot = rot * delta
    new_q = quat_normalize(new_rot.as_quat())

    return new_pos, new_vel, new_q, new_omega, accel_ned
