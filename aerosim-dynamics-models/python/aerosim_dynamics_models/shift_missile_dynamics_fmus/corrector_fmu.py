"""
Corrector FMU — full 6-DOF Ensemble Kalman Filter (EnKF).

The aerodynamics FMU's ML surrogate predicts only three of the six force/moment
channels (force_x, force_z, moment_y) and flags the other three (force_y,
moment_x, moment_z) as *unknown*.  This corrector assimilates the ego's
ground-truth 6-DOF kinematics to estimate the **complete** body-frame
force/moment vector, taking every channel into account:

    state  x = [Fx, Fy, Fz, Mx, My, Mz]            (body FRD)
    forecast/prior:
        - surrogate-valid channels  -> centred on the surrogate value (tight Q)
        - surrogate-unknown channels -> persistence prior (diffuse Q) so the
          observations alone determine them
    observation y = [ax, ay, az, d(p)/dt, d(q)/dt, d(r)/dt]   (body frame)
        derived from the ground-truth VehicleState (specific force + angular
        acceleration by finite difference)
    observation operator:
        a_body       = ([Fx,Fy,Fz] + [thrust,0,0]) / m
        angular_acc  = I^-1 ( [Mx,My,Mz] - omega x (I omega) )

A stochastic perturbed-observation EnKF analysis pulls the forecast toward the
forces implied by the observed accelerations.  The result is a physically
complete estimate of the aerodynamic + control loads, including the channels
the surrogate never modelled, plus the per-channel bias of the surrogate.

Inputs  (VehicleState topic): vehicle_state (ego ground truth)
Inputs  (aux): sm_fx_n..sm_mz_nm + sm_valid_* (aero surrogate),
               thrust_n (propulsion), mass_kg, Ixx, Iyy, Izz (structures)
Outputs (aux): fx_corrected_n..mz_corrected_nm, bias_* , innovation_norm,
               ensemble_spread, ground_truth_valid
"""

from __future__ import annotations

import numpy as np
from pythonfmu3 import Fmi3Slave
from scipy.spatial.transform import Rotation

from aerosim_core import register_fmu3_param, register_fmu3_var
from aerosim_data import dict_to_namespace
from aerosim_data import types as aerosim_types

GRAVITY = 9.80665
_CH = ("fx", "fy", "fz", "mx", "my", "mz")


class corrector_fmu(Fmi3Slave):
    """Full 6-DOF EnKF force/moment corrector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.author = "AeroSim"
        self.description = "Full 6-DOF EnKF corrector (reconstructs unknown channels)"

        self.vehicle_state = dict_to_namespace(aerosim_types.VehicleState().to_dict())
        register_fmu3_var(self, "vehicle_state", causality="input")

        # Surrogate predictions + validity flags.
        self.sm_fx_n = 0.0
        self.sm_fy_n = 0.0
        self.sm_fz_n = 0.0
        self.sm_mx_nm = 0.0
        self.sm_my_nm = 0.0
        self.sm_mz_nm = 0.0
        self.sm_valid_fx = 1.0
        self.sm_valid_fy = 0.0
        self.sm_valid_fz = 1.0
        self.sm_valid_mx = 0.0
        self.sm_valid_my = 1.0
        self.sm_valid_mz = 0.0
        for _n in (
            "sm_fx_n", "sm_fy_n", "sm_fz_n", "sm_mx_nm", "sm_my_nm", "sm_mz_nm",
            "sm_valid_fx", "sm_valid_fy", "sm_valid_fz",
            "sm_valid_mx", "sm_valid_my", "sm_valid_mz",
        ):
            register_fmu3_var(self, _n, causality="input")

        self.thrust_n = 0.0
        self.mass_kg = 500.0
        self.Ixx = 5.0
        self.Iyy = 300.0
        self.Izz = 300.0
        for _n in ("thrust_n", "mass_kg", "Ixx", "Iyy", "Izz"):
            register_fmu3_var(self, _n, causality="input")

        self.time = 0.0
        register_fmu3_var(self, "time", causality="independent")

        # Outputs.
        self.fx_corrected_n = 0.0
        self.fy_corrected_n = 0.0
        self.fz_corrected_n = 0.0
        self.mx_corrected_nm = 0.0
        self.my_corrected_nm = 0.0
        self.mz_corrected_nm = 0.0
        self.bias_fx_n = 0.0
        self.bias_fy_n = 0.0
        self.bias_fz_n = 0.0
        self.bias_mx_nm = 0.0
        self.bias_my_nm = 0.0
        self.bias_mz_nm = 0.0
        self.innovation_norm = 0.0
        self.ensemble_spread = 0.0
        self.ground_truth_valid = 0.0
        for _n in (
            "fx_corrected_n", "fy_corrected_n", "fz_corrected_n",
            "mx_corrected_nm", "my_corrected_nm", "mz_corrected_nm",
            "bias_fx_n", "bias_fy_n", "bias_fz_n",
            "bias_mx_nm", "bias_my_nm", "bias_mz_nm",
            "innovation_norm", "ensemble_spread", "ground_truth_valid",
        ):
            register_fmu3_var(self, _n, causality="output")

        # Tuning.
        self.ensemble_size = 40
        register_fmu3_param(self, "ensemble_size")
        self.proc_std_force_valid_n = 200.0
        register_fmu3_param(self, "proc_std_force_valid_n")
        self.proc_std_moment_valid_nm = 50.0
        register_fmu3_param(self, "proc_std_moment_valid_nm")
        # Diffuse process noise for surrogate-unknown channels.
        self.proc_std_force_unknown_n = 4000.0
        register_fmu3_param(self, "proc_std_force_unknown_n")
        self.proc_std_moment_unknown_nm = 1500.0
        register_fmu3_param(self, "proc_std_moment_unknown_nm")
        self.obs_std_accel_mps2 = 0.4
        register_fmu3_param(self, "obs_std_accel_mps2")
        self.obs_std_angaccel_rps2 = 0.2
        register_fmu3_param(self, "obs_std_angaccel_rps2")
        self.rng_seed = 2024
        register_fmu3_param(self, "rng_seed")

        self._rng = np.random.default_rng(2024)
        self._ensemble = None          # (6, N)
        self._prev_vel_ned = None
        self._prev_omega = None

    # ------------------------------------------------------------------ setup
    def enter_initialization_mode(self):
        self._rng = np.random.default_rng(int(self.rng_seed))
        n = max(4, int(self.ensemble_size))
        prior = self._surrogate_vector()
        self._ensemble = prior.reshape(6, 1) + self._rng.multivariate_normal(
            np.zeros(6), self._process_cov(), size=n
        ).T
        self._prev_vel_ned = None
        self._prev_omega = None
        self._publish(prior, np.zeros(6), 0.0, valid=False)

    def exit_initialization_mode(self):
        pass

    # ------------------------------------------------------------------- step
    def do_step(self, current_time: float, step_size: float) -> bool:
        self.time = current_time + step_size
        surrogate = self._surrogate_vector()
        valid_mask = self._valid_mask()

        obs, ok = self._ground_truth_observation(step_size)
        if not ok or self._ensemble is None:
            self._recenter(surrogate, valid_mask)
            self._publish(surrogate, np.zeros(6), 0.0, valid=False)
            return True

        analysis, innov = self._enkf_update(surrogate, valid_mask, obs)
        # Bias only meaningful for surrogate-valid channels.
        bias = np.where(valid_mask > 0.5, analysis - surrogate, 0.0)
        self._publish(analysis, bias, innov, valid=True)
        return True

    def terminate(self):
        print("Terminating corrector_fmu (6-DOF EnKF).")
        self.time = 0.0

    # ------------------------------------------------------------- EnKF core
    def _surrogate_vector(self) -> np.ndarray:
        return np.array([
            self.sm_fx_n, self.sm_fy_n, self.sm_fz_n,
            self.sm_mx_nm, self.sm_my_nm, self.sm_mz_nm,
        ], dtype=float)

    def _valid_mask(self) -> np.ndarray:
        return np.array([
            self.sm_valid_fx, self.sm_valid_fy, self.sm_valid_fz,
            self.sm_valid_mx, self.sm_valid_my, self.sm_valid_mz,
        ], dtype=float)

    def _process_std(self, valid_mask=None) -> np.ndarray:
        if valid_mask is None:
            valid_mask = self._valid_mask()
        fstd = np.where(
            valid_mask[:3] > 0.5, self.proc_std_force_valid_n, self.proc_std_force_unknown_n
        )
        mstd = np.where(
            valid_mask[3:] > 0.5, self.proc_std_moment_valid_nm, self.proc_std_moment_unknown_nm
        )
        return np.concatenate([fstd, mstd])

    def _process_cov(self, valid_mask=None) -> np.ndarray:
        return np.diag(self._process_std(valid_mask) ** 2)

    def _obs_cov(self) -> np.ndarray:
        return np.diag([
            self.obs_std_accel_mps2 ** 2, self.obs_std_accel_mps2 ** 2,
            self.obs_std_accel_mps2 ** 2, self.obs_std_angaccel_rps2 ** 2,
            self.obs_std_angaccel_rps2 ** 2, self.obs_std_angaccel_rps2 ** 2,
        ])

    def _recenter(self, surrogate: np.ndarray, valid_mask: np.ndarray) -> None:
        """Forecast step: valid channels re-centre on the surrogate; unknown
        channels persist their current estimate. Then inflate by process noise."""
        if self._ensemble is None:
            return
        n = self._ensemble.shape[1]
        cur_mean = self._ensemble.mean(axis=1)
        center = np.where(valid_mask > 0.5, surrogate, cur_mean)
        self._ensemble = center.reshape(6, 1) + self._rng.multivariate_normal(
            np.zeros(6), self._process_cov(valid_mask), size=n
        ).T

    def _obs_operator(self, ensemble: np.ndarray, omega: np.ndarray) -> np.ndarray:
        mass = max(self.mass_kg, 1e-3)
        I = np.array([max(self.Ixx, 1e-6), max(self.Iyy, 1e-6), max(self.Izz, 1e-6)])
        forces = ensemble[:3, :]
        moments = ensemble[3:, :]
        a = forces.copy()
        a[0, :] += self.thrust_n
        a /= mass
        # Angular acceleration: I^-1 (M - omega x (I omega)); gyroscopic term is
        # constant across the ensemble (omega from ground truth).
        gyro = np.cross(omega, I * omega)
        ang = (moments - gyro.reshape(3, 1)) / I.reshape(3, 1)
        return np.vstack([a, ang])

    def _enkf_update(self, surrogate, valid_mask, obs):
        self._recenter(surrogate, valid_mask)
        ens = self._ensemble
        n = ens.shape[1]
        omega = self._prev_omega if self._prev_omega is not None else np.zeros(3)

        pred = self._obs_operator(ens, omega)
        y_mean = pred.mean(axis=1, keepdims=True)
        x_anom = ens - ens.mean(axis=1, keepdims=True)
        y_anom = pred - y_mean
        denom = max(n - 1, 1)
        R = self._obs_cov()
        pxy = (x_anom @ y_anom.T) / denom
        pyy = (y_anom @ y_anom.T) / denom + R
        # Regularise + robust solve so the analysis can never blow up numerically.
        pyy += np.eye(6) * (1e-6 * float(np.trace(pyy)) + 1e-9)
        innov = float(np.linalg.norm(obs - y_mean.flatten()))
        try:
            gain = np.linalg.solve(pyy.T, pxy.T).T
        except np.linalg.LinAlgError:
            gain = pxy @ np.linalg.pinv(pyy)
        noise = self._rng.multivariate_normal(np.zeros(6), R, size=n).T
        new_ens = ens + gain @ (obs.reshape(6, 1) + noise - pred)
        if not np.all(np.isfinite(new_ens)):
            # Reject the pathological update; reinitialise around the surrogate.
            self._ensemble = surrogate.reshape(6, 1) + self._rng.multivariate_normal(
                np.zeros(6), self._process_cov(valid_mask), size=n
            ).T
            return self._ensemble.mean(axis=1), innov
        self._ensemble = new_ens
        return self._ensemble.mean(axis=1), innov

    # ------------------------------------------------------ ground-truth obs
    def _ground_truth_observation(self, dt: float):
        pose = self.vehicle_state.state.pose
        vel = self.vehicle_state.velocity
        av = self.vehicle_state.angular_velocity
        vel_ned = np.array([vel.x, vel.y, vel.z], dtype=float)
        omega = np.array([av.x, av.y, av.z], dtype=float)
        q = pose.orientation
        has_orient = not (q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0)
        moving = (np.linalg.norm(vel_ned) + abs(pose.position.x) + abs(pose.position.z)) > 1e-6
        if not (has_orient or moving):
            self._prev_vel_ned = None
            self._prev_omega = None
            return np.zeros(6), False
        if self._prev_vel_ned is None or dt <= 0.0:
            self._prev_vel_ned = vel_ned
            self._prev_omega = omega
            return np.zeros(6), False

        accel_ned = (vel_ned - self._prev_vel_ned) / dt
        specific_ned = accel_ned - np.array([0.0, 0.0, GRAVITY])
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]) if has_orient else Rotation.identity()
        accel_body = rot.inv().apply(specific_ned)
        ang_acc = (omega - self._prev_omega) / dt

        self._prev_vel_ned = vel_ned
        self._prev_omega = omega
        obs = np.concatenate([accel_body, ang_acc])
        if not np.all(np.isfinite(obs)):
            return np.zeros(6), False
        return obs, True

    # ---------------------------------------------------------------- output
    def _publish(self, corrected, bias, innov, valid):
        (self.fx_corrected_n, self.fy_corrected_n, self.fz_corrected_n,
         self.mx_corrected_nm, self.my_corrected_nm, self.mz_corrected_nm) = (
            float(v) for v in corrected
        )
        (self.bias_fx_n, self.bias_fy_n, self.bias_fz_n,
         self.bias_mx_nm, self.bias_my_nm, self.bias_mz_nm) = (float(v) for v in bias)
        self.innovation_norm = float(innov)
        if self._ensemble is not None:
            self.ensemble_spread = float(np.mean(np.std(self._ensemble, axis=1)))
        self.ground_truth_valid = 1.0 if valid else 0.0
