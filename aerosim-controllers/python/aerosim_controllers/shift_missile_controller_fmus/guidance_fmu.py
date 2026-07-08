"""
Guidance FMU for the SHIFT interceptor (outer loop + fire-control logic).

This FMU is the interceptor's **fire-control and guidance computer**.  It
consumes the ego and threat state *estimates* (from the navigation EKFs, never
ground truth) plus an optional launcher/track handoff cue, sequences the
engagement through its fire-control phases, and produces a commanded
maneuver-acceleration vector in NED that the inner-loop autopilot tracks.

Fire-control phases
-------------------
IDLE (0)        No usable track.  Hold the launch attitude (zero maneuver cmd).
MIDCOURSE (1)   A cue or track exists but the terminal seeker has not taken over.
                Fly a lead-collision course to the Predicted Intercept Point
                (PIP) computed from the target estimate and the interceptor's
                achievable speed.  This is the "initial guidance from launcher
                information and detected direction."
TERMINAL (2)    Seeker/target-EKF locked and range < ``terminal_range_m``.  Run
                the selected terminal homing law.

Terminal homing laws (``guidance_law``)
--------------------------------------
"propnav"   True vector Proportional Navigation, optionally Augmented PN (APN)
            with an estimated-target-acceleration feed-forward:
                a_cmd = N * Vc * (omega_los x LOS_hat) + (N/2) * a_tgt_perp
"mpc"       Real receding-horizon Model Predictive Control.  The relative
            engagement is a decoupled double integrator per NED axis
            (missile accel is the control, target accel a known disturbance).
            Each step condenses an N-step horizon into a small convex QP

                min  w_miss*rN^2 + w_eff*Σuk^2 + w_rate*Σ(uk-uk-1)^2
                s.t. |uk| <= a_max      (actuator limit)

            solved by a primal-dual interior-point QP (``qp_solver``).  Only the
            first control move is applied (receding horizon).  No ZEM shortcut.

Inputs  (aux): nav_pos_{n,e,d}, nav_vel_{n,e,d}, nav_valid  (ego EKF)
               tgt_pos_{n,e,d}, tgt_vel_{n,e,d}, tgt_acc_{n,e,d}, tgt_valid
               cue_pos_{n,e,d}, cue_valid  (launcher/track handoff)
Outputs (aux): a_cmd_{n,e,d}, range_m, closing_speed_mps, t_go_s, los_rate_rps,
               zem_m, pip_n/e/d, guidance_phase, guidance_active
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave

from aerosim_core import register_fmu3_param, register_fmu3_var
from qp_solver import solve_qp

GRAVITY = 9.80665
PHASE_IDLE, PHASE_MIDCOURSE, PHASE_TERMINAL = 0.0, 1.0, 2.0


class guidance_fmu(Fmi3Slave):
    """Fire-control sequencing + PropNav/APN and receding-horizon MPC guidance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "SHIFT interceptor fire-control + guidance (PropNav/APN, MPC)"

        # --- Ego / threat estimates + handoff cue (aux inputs) ----------------
        self.nav_pos_n = 0.0
        self.nav_pos_e = 0.0
        self.nav_pos_d = 0.0
        self.nav_vel_n = 0.0
        self.nav_vel_e = 0.0
        self.nav_vel_d = 0.0
        self.nav_valid = 0.0
        self.tgt_pos_n = 0.0
        self.tgt_pos_e = 0.0
        self.tgt_pos_d = 0.0
        self.tgt_vel_n = 0.0
        self.tgt_vel_e = 0.0
        self.tgt_vel_d = 0.0
        self.tgt_acc_n = 0.0
        self.tgt_acc_e = 0.0
        self.tgt_acc_d = 0.0
        self.tgt_valid = 0.0
        self.cue_pos_n = 0.0
        self.cue_pos_e = 0.0
        self.cue_pos_d = 0.0
        self.cue_valid = 0.0
        for _n in (
            "nav_pos_n", "nav_pos_e", "nav_pos_d",
            "nav_vel_n", "nav_vel_e", "nav_vel_d", "nav_valid",
            "tgt_pos_n", "tgt_pos_e", "tgt_pos_d",
            "tgt_vel_n", "tgt_vel_e", "tgt_vel_d",
            "tgt_acc_n", "tgt_acc_e", "tgt_acc_d", "tgt_valid",
            "cue_pos_n", "cue_pos_e", "cue_pos_d", "cue_valid",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # --- Outputs ----------------------------------------------------------
        self.a_cmd_n = 0.0
        self.a_cmd_e = 0.0
        self.a_cmd_d = 0.0
        self.range_m = 0.0
        self.closing_speed_mps = 0.0
        self.t_go_s = 0.0
        self.los_rate_rps = 0.0
        self.zem_m = 0.0
        self.pip_n = 0.0
        self.pip_e = 0.0
        self.pip_d = 0.0
        self.guidance_phase = PHASE_IDLE
        self.guidance_active = 0.0
        for _n in (
            "a_cmd_n", "a_cmd_e", "a_cmd_d", "range_m", "closing_speed_mps",
            "t_go_s", "los_rate_rps", "zem_m", "pip_n", "pip_e", "pip_d",
            "guidance_phase", "guidance_active",
        ):
            register_fmu3_var(self, _n, causality="output")

        # --- Parameters -------------------------------------------------------
        self.guidance_law = "propnav"            # terminal law: "propnav" | "mpc"
        register_fmu3_param(self, "guidance_law")
        self.nav_gain = 4.0                       # PropNav effective navigation ratio N'
        register_fmu3_param(self, "nav_gain")
        self.augmented_propnav = 1.0              # 1 -> APN (use target-accel term)
        register_fmu3_param(self, "augmented_propnav")
        self.max_accel_g = 40.0
        register_fmu3_param(self, "max_accel_g")
        self.min_closing_speed_mps = 5.0
        register_fmu3_param(self, "min_closing_speed_mps")
        self.terminal_range_m = 3000.0            # range to hand over to terminal homing
        register_fmu3_param(self, "terminal_range_m")
        self.midcourse_gain = 1.0                 # lead-collision proportional gain
        register_fmu3_param(self, "midcourse_gain")
        self.midcourse_max_g = 8.0                # energy-managed midcourse accel cap
        register_fmu3_param(self, "midcourse_max_g")
        # Soft-start: linearly ramp the commanded accel over the first
        # ``command_ramp_s`` seconds after guidance first goes active.  This
        # suppresses the transient command spike produced while the navigation
        # EKF attitude/velocity estimates are still settling at hand-off, which
        # would otherwise pitch the airframe up and loft it off the collision
        # course.  A ramp can only *attenuate* early commands, so it cannot
        # destabilize the loop.
        self.command_ramp_s = 0.0
        register_fmu3_param(self, "command_ramp_s")
        self._t_active0 = -1.0

        # MPC weights + horizon.
        self.mpc_horizon = 20
        register_fmu3_param(self, "mpc_horizon")
        self.mpc_w_miss = 1.0
        register_fmu3_param(self, "mpc_w_miss")
        self.mpc_w_effort = 1.0e-4
        register_fmu3_param(self, "mpc_w_effort")
        self.mpc_w_rate = 1.0e-4
        register_fmu3_param(self, "mpc_w_rate")
        self.mpc_min_dt = 0.02                    # floor on per-step horizon dt
        register_fmu3_param(self, "mpc_min_dt")

        self._u_prev = np.zeros(3)                # last applied accel (rate weight)
        self._t_active0 = -1.0                     # time guidance first went active

    def enter_initialization_mode(self):
        self._u_prev = np.zeros(3)
        self._t_active0 = -1.0

    def exit_initialization_mode(self):
        pass

    # ------------------------------------------------------------------- step
    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size

        if self.nav_valid < 0.5:
            self._u_prev = np.zeros(3)
            self._publish(np.zeros(3), 0.0, 0.0, 0.0, 0.0, 0.0,
                          np.zeros(3), PHASE_IDLE, active=False)
            return True

        p_ego = np.array([self.nav_pos_n, self.nav_pos_e, self.nav_pos_d])
        v_ego = np.array([self.nav_vel_n, self.nav_vel_e, self.nav_vel_d])

        have_track = self.tgt_valid > 0.5
        have_cue = self.cue_valid > 0.5
        if have_track:
            p_tgt = np.array([self.tgt_pos_n, self.tgt_pos_e, self.tgt_pos_d])
            v_tgt = np.array([self.tgt_vel_n, self.tgt_vel_e, self.tgt_vel_d])
            a_tgt = np.array([self.tgt_acc_n, self.tgt_acc_e, self.tgt_acc_d])
        elif have_cue:
            p_tgt = np.array([self.cue_pos_n, self.cue_pos_e, self.cue_pos_d])
            v_tgt = np.zeros(3)   # cue is position-only handoff
            a_tgt = np.zeros(3)
        else:
            self._u_prev = np.zeros(3)
            self._publish(np.zeros(3), 0.0, 0.0, 0.0, 0.0, 0.0,
                          np.zeros(3), PHASE_IDLE, active=False)
            return True

        r = p_tgt - p_ego
        v = v_tgt - v_ego
        rng = float(np.linalg.norm(r))
        if rng < 1e-3:
            self._publish(np.zeros(3), rng, 0.0, 0.0, 0.0, 0.0,
                          p_tgt, PHASE_TERMINAL, active=False)
            return True

        r_hat = r / rng
        closing = -float(np.dot(r, v)) / rng
        vc = max(closing, self.min_closing_speed_mps)
        t_go = rng / vc
        omega_los = np.cross(r, v) / (rng * rng)
        los_rate = float(np.linalg.norm(omega_los))

        # Predicted intercept point (constant-velocity target, current ego speed).
        pip = self._compute_pip(p_ego, v_ego, p_tgt, v_tgt)

        terminal = have_track and rng <= self.terminal_range_m
        a_max = self.max_accel_g * GRAVITY

        if terminal:
            phase = PHASE_TERMINAL
            law = str(self.guidance_law).strip().lower()
            if law == "mpc":
                a_cmd = self._mpc(r, v, a_tgt, t_go, a_max)
            else:
                a_cmd = self._propnav(r_hat, vc, omega_los, a_tgt)
        else:
            phase = PHASE_MIDCOURSE
            mid_max = min(a_max, self.midcourse_max_g * GRAVITY)
            a_cmd = self._midcourse(p_ego, v_ego, pip, mid_max)

        a_cmd = self._limit(a_cmd, a_max)

        # Soft-start ramp on the first seconds of active guidance.
        if self._t_active0 < 0.0:
            self._t_active0 = self.time
        ramp = self.command_ramp_s
        if ramp > 1e-3:
            frac = (self.time - self._t_active0) / ramp
            if frac < 1.0:
                a_cmd = a_cmd * max(0.0, frac)

        self._u_prev = a_cmd

        # Perpendicular ZEM diagnostic (not used as a control law).
        zem_vec = r + v * t_go + 0.5 * a_tgt * t_go * t_go
        zem = float(np.linalg.norm(zem_vec - np.dot(zem_vec, r_hat) * r_hat))

        self._publish(a_cmd, rng, closing, t_go, los_rate, zem, pip, phase, active=True)
        return True

    def terminate(self):
        print("Terminating guidance_fmu.")
        self.time = 0.0

    # ------------------------------------------------------------------- laws
    def _propnav(self, r_hat, vc, omega_los, a_tgt):
        """True PN + optional APN target-acceleration term."""
        a_cmd = self.nav_gain * vc * np.cross(omega_los, r_hat)
        if self.augmented_propnav > 0.5:
            a_perp = a_tgt - np.dot(a_tgt, r_hat) * r_hat
            a_cmd = a_cmd + 0.5 * self.nav_gain * a_perp
        return a_cmd

    def _midcourse(self, p_ego, v_ego, pip, a_max):
        """Lead-collision steering: turn the velocity vector toward the PIP.

        Commands a lateral acceleration proportional to the angle between the
        current velocity and the desired line to the PIP (a proportional
        heading-hold / pursuit of the predicted intercept point)."""
        speed = float(np.linalg.norm(v_ego))
        to_pip = pip - p_ego
        dist = float(np.linalg.norm(to_pip))
        if speed < 1.0 or dist < 1e-3:
            return np.zeros(3)
        v_hat = v_ego / speed
        los_hat = to_pip / dist
        # Perpendicular component of the desired direction = heading error.
        perp = los_hat - np.dot(los_hat, v_hat) * v_hat
        a_cmd = self.midcourse_gain * speed * speed / max(dist, 1.0) * perp
        # Energy-managed cap: never bleed more than the midcourse accel limit far
        # from the target (prevents chasing PIP jitter / target weave up-range).
        return self._limit(a_cmd, a_max)

    def _mpc(self, r, v, a_tgt, t_go, a_max):
        """Receding-horizon constrained MPC; decoupled double integrator per axis."""
        N = max(2, int(self.mpc_horizon))
        T = max(self.mpc_min_dt, t_go / N)
        A = np.array([[1.0, T], [0.0, 1.0]])
        B = np.array([0.5 * T * T, T])
        e1 = np.array([1.0, 0.0])

        # Terminal-position sensitivity g_k to control u_k, and free/disturbance
        # response f, per axis (dynamics identical across axes).
        g = np.zeros(N)
        An = [np.linalg.matrix_power(A, k) for k in range(N + 1)]
        for k in range(N):
            g[k] = e1 @ An[N - 1 - k] @ B          # d rN / d(a_tgt - u_k) chain
        # r_N(axis) = e1 A^N x0 + Σ_k e1 A^{N-1-k} B (a_tgt - u_k)
        #           = f0 + g·a_tgt - g·U
        Dm = np.eye(N) - np.eye(N, k=-1)           # first-difference (rate) matrix
        w_miss, w_eff, w_rate = self.mpc_w_miss, self.mpc_w_effort, self.mpc_w_rate
        P = 2.0 * (w_miss * np.outer(g, g) + w_eff * np.eye(N) + w_rate * (Dm.T @ Dm))
        Gc = np.vstack([np.eye(N), -np.eye(N)])
        hc = np.full(2 * N, a_max)

        a_cmd = np.zeros(3)
        for ax in range(3):
            x0 = np.array([r[ax], v[ax]])
            f0 = e1 @ An[N] @ x0 + float(g @ (np.ones(N) * a_tgt[ax]))
            # rate term couples u_0 to previous applied command.
            rate_lin = np.zeros(N)
            rate_lin[0] = -self._u_prev[ax]
            q = 2.0 * (-w_miss * f0 * g + w_rate * (Dm.T @ rate_lin))
            U, _ = solve_qp(P, q, Gc, hc)
            a_cmd[ax] = float(U[0])
        return a_cmd

    def _compute_pip(self, p_ego, v_ego, p_tgt, v_tgt):
        """Predicted intercept point for a constant-velocity target.

        Solve |p_tgt + v_tgt t - p_ego| = Vm t for the smallest positive t
        (missile flies at its current speed Vm toward the PIP), then
        PIP = p_tgt + v_tgt t."""
        Vm = float(np.linalg.norm(v_ego))
        rel = p_tgt - p_ego
        if Vm < 1.0:
            return p_tgt
        a = float(v_tgt @ v_tgt) - Vm * Vm
        b = 2.0 * float(rel @ v_tgt)
        c = float(rel @ rel)
        t = None
        if abs(a) < 1e-6:
            if abs(b) > 1e-9:
                t = -c / b
        else:
            disc = b * b - 4 * a * c
            if disc >= 0.0:
                sq = np.sqrt(disc)
                roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
                pos = [x for x in roots if x > 1e-3]
                if pos:
                    t = min(pos)
        if t is None or not np.isfinite(t):
            t = float(np.linalg.norm(rel)) / max(Vm, 1.0)
        return p_tgt + v_tgt * t

    def _limit(self, a_cmd, a_max):
        norm = float(np.linalg.norm(a_cmd))
        if norm > a_max and norm > 1e-9:
            return a_cmd * (a_max / norm)
        return a_cmd

    def _publish(self, a_cmd, rng, closing, t_go, los_rate, zem, pip, phase, active):
        self.a_cmd_n, self.a_cmd_e, self.a_cmd_d = (float(v) for v in a_cmd)
        self.range_m = float(rng)
        self.closing_speed_mps = float(closing)
        self.t_go_s = float(t_go)
        self.los_rate_rps = float(los_rate)
        self.zem_m = float(zem)
        self.pip_n, self.pip_e, self.pip_d = (float(v) for v in pip)
        self.guidance_phase = float(phase)
        self.guidance_active = 1.0 if active else 0.0
