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
    """Normalize a scalar-last quaternion; fall back to identity if degenerate."""
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


# ── Quaternion kinematics ─────────────────────────────────────────────────────
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-last quaternions (q1 ⊗ q2)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_deriv(q_xyzw: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Time-derivative of a body->NED scalar-last quaternion for body rates.

    ``q_dot = 0.5 * q ⊗ [omega, 0]`` (omega expressed in the body frame)."""
    w = np.asarray(omega_body, dtype=float)
    return 0.5 * quat_mul(q_xyzw, np.array([w[0], w[1], w[2], 0.0]))


# ── Rigid-body integration ────────────────────────────────────────────────────
def integrate_6dof_rk4(
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
    """Classical fourth-order Runge-Kutta step of the 6-DOF rigid-body EOM.

    Body-frame forces/moments are held constant across the step (co-simulation
    zero-order hold); the state derivative still captures the attitude-dependent
    gravity/force rotation and the gyroscopic coupling, so RK4 is markedly more
    accurate and stable than explicit Euler for the stiff high-q airframe.

    State packed as ``[pos(3), vel(3), quat_xyzw(4), omega(3)]``.  Returns
    ``(pos, vel, q, omega, accel_ned)`` advanced by ``dt``.
    """
    mass = max(float(mass), 1e-6)
    I = np.maximum(np.asarray(inertia_diag, dtype=float), 1e-6)
    fb = np.asarray(force_body, dtype=float)
    mb = np.asarray(moment_body, dtype=float)
    g_ned = np.array([0.0, 0.0, GRAVITY]) if gravity else np.zeros(3)

    def deriv(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = state[6:10]
        w = state[10:13]
        rot = rot_from_quat(q)
        accel = rot.apply(fb) / mass + g_ned
        qd = quat_deriv(q, w)
        ang_acc = (mb - np.cross(w, I * w)) / I
        d = np.zeros(13)
        d[0:3] = state[3:6]      # pos_dot = vel
        d[3:6] = accel           # vel_dot
        d[6:10] = qd             # quat_dot
        d[10:13] = ang_acc       # omega_dot
        return d, accel

    s0 = np.concatenate([pos_ned, vel_ned, quat_normalize(q_xyzw), omega_body])
    k1, a1 = deriv(s0)
    k2, _ = deriv(s0 + 0.5 * dt * k1)
    k3, _ = deriv(s0 + 0.5 * dt * k2)
    k4, _ = deriv(s0 + dt * k3)
    s1 = s0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    pos = s1[0:3]
    vel = s1[3:6]
    q = quat_normalize(s1[6:10])
    omega = s1[10:13]
    return pos, vel, q, omega, a1


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
