# SHIFT Interceptor — Guidance, Navigation & Control (GNC) and Fire-Control FMUs

Guidance, navigation, control, and fire-control co-simulation units for the
SHIFT surface-to-air interceptor. Every module in this folder is a
[`pythonfmu3`](https://github.com/NTNU-IHB/PythonFMU3) `Fmi3Slave` (or a shared
helper bundled as an FMU resource) and participates in the AeroSim distributed
co-simulation as an FMI 3.0 model exchanged over the AeroSim message bus.

All estimator, guidance, and autopilot FMUs run on *estimated* state only — they
consume the outputs of the on-board navigation filters and raw sensor streams,
never simulation ground truth — so the closed loop reproduces the observability
and latency limits of a real fire-control system.

---

## 1. Executive Summary

This package implements the complete GNC / fire-control chain of a modern
homing interceptor, decomposed exactly the way a real weapon system is:

```
   Seekers / IMU / GNSS / Baro
              │
     ┌────────┴─────────┐
     ▼                  ▼
 ego_nav_ekf_fmu   target_nav_ekf_fmu      ← Navigation (state estimation)
  (16-state INS)    (9-state tracker)
     │                  │
     └────────┬─────────┘
              ▼
        guidance_fmu                        ← Fire-control + guidance (outer loop)
   IDLE → MIDCOURSE → TERMINAL
   PIP lead-collision │ PropNav/APN │ MPC
              │  a_cmd (NED acceleration)
              ▼
       autopilot_fmu                         ← Inner-loop autopilot (3-loop, LQR/PID)
   accel → incidence trim → rate damping
              │  fin deflections
              ▼
        Airframe plant (dynamics FMUs)
```

**Initial (boost / midcourse) guidance.** At launch the interceptor has only a
launcher cue or a coarse track — a *detected direction* to the threat. The
fire-control computer (`guidance_fmu`) enters the **MIDCOURSE** phase and flies a
lead-collision course toward a **Predicted Intercept Point (PIP)** computed from
the target estimate and the interceptor's achievable speed. This is energy
managed (a lower acceleration cap `midcourse_max_g`) so the missile does not
bleed speed chasing PIP jitter far from the target.

**Terminal guidance.** When a valid target track exists and range falls below
`terminal_range_m`, the computer switches to the **TERMINAL** phase and runs the
selected homing law:

- **PropNav / APN** — true vector Proportional Navigation with an optional
  Augmented-PN target-acceleration feed-forward.
- **MPC** — a receding-horizon Model Predictive Controller. The
  relative engagement is modeled as a decoupled double integrator per NED axis;
  each step an \(N\)-step horizon is *condensed* into a small convex QP
  (terminal-miss + control-effort + control-rate cost, subject to actuator
  acceleration limits) and solved by the bundled interior-point `qp_solver`.

**Inner loop.** `autopilot_fmu` is a classical tactical-missile **three-loop
skid-to-turn acceleration autopilot**. It converts the guidance NED
acceleration command into incidence commands (\(\alpha, \beta\)) via quasi-static
trim, closes an incidence-trim integral outer loop, and drives the fins with an
elevator/rudder trim feed-forward plus rate-gyro damping. The damping gain is
set either by an online **LQR** short-period Riccati solution or by a fixed
**PID** gain.

**Navigation.** Two extended Kalman filters supply the estimated state:
`ego_nav_ekf_fmu`, a 16-state quaternion strapdown INS (error-state / MEKF)
fusing IMU, GNSS, and baro at mismatched rates; and `target_nav_ekf_fmu`, a
9-state constant-acceleration kinematic tracker fusing IR angles and
semi-active-radar range/angle/range-rate.

**Uncertainty support.** `uncertainty.py` turns miss-distance distributions into
a probability of kill (\(P_{\text{kill}}\)) with confidence bounds, using the
Carleton diffuse-Gaussian damage function, closed-form and Monte-Carlo SSPK
estimators, and the Wilson score interval, plus parameter sampling and
first-order error propagation for surrogate-model UQ.

Shared, dependency-light math lives in `sixdof.py` (quaternion 6-DOF RK4 and
kinematics) and `atmosphere.py` (ICAO ISA), each bundled as an FMU resource.

---

## 2. Component Descriptions

Naming conventions used throughout: positions/velocities are in the **NED**
(North-East-Down) world frame; the body frame is **FRD** (x forward, y right, z
down); quaternions are stored scalar-last `[x, y, z, w]` and represent the
body→NED rotation. Gravity is `GRAVITY = 9.80665 m/s²`.

### 2.1 `guidance_fmu.py` — Fire-control computer + outer-loop guidance

Sequences the engagement through fire-control phases and produces the commanded
maneuver-acceleration vector the autopilot tracks.

| Phase | Value | Condition | Behavior |
|-------|-------|-----------|----------|
| `IDLE` | `0.0` | `nav_valid` false, or no track and no cue | Zero maneuver command; hold launch attitude |
| `MIDCOURSE` | `1.0` | Cue or track exists, range `> terminal_range_m` | Lead-collision steering to the PIP, capped at `midcourse_max_g` |
| `TERMINAL` | `2.0` | Valid track **and** range `≤ terminal_range_m` | Selected terminal homing law (`propnav` or `mpc`) |

**Inputs** (aux, all `float`): ego estimate `nav_pos_{n,e,d}`, `nav_vel_{n,e,d}`,
`nav_valid`; threat estimate `tgt_pos_{n,e,d}`, `tgt_vel_{n,e,d}`,
`tgt_acc_{n,e,d}`, `tgt_valid`; launcher/track handoff `cue_pos_{n,e,d}`,
`cue_valid` (a position-only cue; velocity and acceleration are taken as zero
when only the cue is available).

**Outputs** (aux): `a_cmd_{n,e,d}` (commanded NED acceleration, m/s²), `range_m`,
`closing_speed_mps`, `t_go_s`, `los_rate_rps`, `zem_m` (diagnostic),
`pip_{n,e,d}`, `guidance_phase`, `guidance_active`.

**Parameters** (defaults in parentheses):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `guidance_law` | `"propnav"` | Terminal law: `"propnav"` or `"mpc"` |
| `nav_gain` | `4.0` | Effective navigation ratio \(N'\) for PropNav |
| `augmented_propnav` | `1.0` | `>0.5` enables the APN target-acceleration term |
| `max_accel_g` | `40.0` | Hard acceleration-command cap (g) |
| `min_closing_speed_mps` | `5.0` | Floor on closing speed used for \(V_c\) / \(t_{go}\) |
| `terminal_range_m` | `3000.0` | Range at which terminal homing takes over |
| `midcourse_gain` | `1.0` | Lead-collision proportional gain |
| `midcourse_max_g` | `8.0` | Energy-managed midcourse acceleration cap (g) |
| `command_ramp_s` | `0.0` | Soft-start ramp duration (see below) |
| `mpc_horizon` | `20` | MPC horizon length \(N\) (steps) |
| `mpc_w_miss` | `1.0` | MPC terminal-miss weight |
| `mpc_w_effort` | `1.0e-4` | MPC control-effort weight |
| `mpc_w_rate` | `1.0e-4` | MPC control-rate (slew) weight |
| `mpc_min_dt` | `0.02` | Floor on the per-step horizon interval \(T\) (s) |

**Command soft-start ramp (`command_ramp_s`).** When guidance first goes active
the commanded acceleration is linearly scaled by
\(\text{frac} = (t - t_{\text{active},0}) / \texttt{command\_ramp\_s}\), clamped to
\([0, 1]\). This attenuates the transient command spike produced while the ego
EKF attitude/velocity estimates are still settling at hand-off, which would
otherwise pitch the airframe up and loft it off the collision course. Because a
ramp can only *attenuate* early commands (multiplier \(\le 1\)), it cannot
destabilize the loop. A value of `0.0` disables the ramp.

### 2.2 `autopilot_fmu.py` — Inner-loop three-loop acceleration autopilot

Converts the guidance NED acceleration command into fin deflections. Skid-to-turn
(no bank-to-turn); roll is held wings-level.

**Inputs** (aux): guidance `a_cmd_{n,e,d}`, `guidance_active`; ego EKF
`nav_q{w,x,y,z}`, `nav_p/q/r`, `nav_vel_{n,e,d}`, `nav_valid`; **raw strapdown
IMU** `gyro_{p,q,r}`, `accel_{x,y,z}` (low-latency body FRD); flight condition
`qbar_pa`, `airspeed_mps`, `mass_kg`, `Iyy`, `Izz`.

**Outputs** (aux): `elevator_cmd_rad`, `aileron_cmd_rad`, `rudder_cmd_rad`,
`throttle_cmd` (held at `1.0`); reported `az_cmd_mps2`, `ay_cmd_mps2` (commanded
aerodynamic specific force) and `az_ach_mps2`, `ay_ach_mps2` (model-based
achieved specific force from the trimmed incidence).

**Parameters** (defaults):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `controller_type` | `"lqr"` | `"lqr"` (on-line Riccati rate gain) or `"pid"` (fixed gain) |
| `use_gyro_rate` | `1.0` | `>0.5` feeds back the raw rate gyro; else the nav-filter rate estimate |
| `ref_area_m2` | `0.0314` | Aerodynamic reference area \(S\) |
| `ref_diameter_m` | `0.2` | Reference diameter \(d\) |
| `CN_alpha` | `15.0` | Normal-force curve slope \(C_{N_\alpha}\) |
| `CN_de` | `5.0` | Normal force per elevator \(C_{N_{\delta e}}\) |
| `Cm_alpha` | `-3.0` | Pitch static-stability derivative \(C_{m_\alpha}\) |
| `Cm_de` | `-8.0` | Pitch control power \(C_{m_{\delta e}}\) |
| `Cm_q` | `-50.0` | Pitch-rate damping \(C_{m_q}\) |
| `CY_beta` | `-15.0` | Side-force derivative \(C_{Y_\beta}\) |
| `CY_dr` | `5.0` | Side force per rudder \(C_{Y_{\delta r}}\) |
| `Cn_beta` | `3.0` | Yaw static-stability derivative \(C_{n_\beta}\) |
| `Cn_dr` | `-8.0` | Yaw control power \(C_{n_{\delta r}}\) |
| `Cn_r` | `-50.0` | Yaw-rate damping \(C_{n_r}\) |
| `Cl_da` | `-2.0` | Roll control power \(C_{l_{\delta a}}\) |
| `ki_accel` | `3.0` | Incidence-trim integral gain (1/s) |
| `kq_rate` | `0.30` | Fixed rate-damping gain (rad fin per rad/s), PID option |
| `max_incidence_rad` | `0.30` | Incidence-command clamp (~17°) |
| `lqr_q_angle` | `5.0` | LQR state weight on incidence |
| `lqr_q_rate` | `2.0` | LQR state weight on body rate |
| `lqr_r_fin` | `200.0` | LQR control weight on fin deflection |
| `kp_roll` | `0.4` | Roll-angle proportional gain |
| `kd_roll` | `0.15` | Roll-rate derivative gain |
| `max_fin_rad` | `0.436332` | Fin-deflection clamp (25°) |
| `max_accel_cmd_mps2` | `300.0` | Body specific-force command clamp (~30 g) |

The rate-damping loop feeds back the **raw rate gyro** (`use_gyro_rate`), which is
low-latency and minimum-phase, rather than the lagged nav-filter estimate. The
loop gain is automatically gain-scheduled with dynamic pressure through the
analytic incidence→fin trim mapping.

### 2.3 `ego_nav_ekf_fmu.py` — Ego 16-state quaternion INS (error-state EKF)

A full 6-DOF strapdown inertial navigation system. The nominal state carries 16
quantities; the covariance is propagated on the 15 error states with a
multiplicative attitude error (MEKF), the numerically correct formulation for a
quaternion INS.

- **Nominal state (16):** \(p(3), v(3), q(4), b_a(3), b_g(3)\).
- **Error state (15):** \(\delta p, \delta v, \delta\theta, \delta b_a, \delta b_g\).

**Prediction** runs whenever `imu_measurement_ready` is set (IMU strapdown, every
step). **Updates:** GNSS (position + velocity, linear in \(\delta p, \delta v\))
whenever `gnss_measurement_ready` is set; baro (down-position, linear in
\(\delta p_D\)) whenever `baro_measurement_ready` is set. Each sensor is fused
only on its own ready flag, so mismatched/asynchronous rates are handled
naturally. After each batch of updates the error state is injected into the
nominal state and reset to zero. The first GNSS fix seeds position/velocity
directly and sets `nav_valid`.

**Inputs** (aux): IMU `accel_{x,y,z}_mps2`, `gyro_{x,y,z}_rps`,
`imu_measurement_ready`; GNSS `pos_{n,e,d}_m`, `vel_{n,e,d}_mps`,
`gnss_measurement_ready`; baro `baro_alt_m`, `baro_measurement_ready`.

**Outputs** (aux): `nav_pos_{n,e,d}`, `nav_vel_{n,e,d}`, `nav_q{w,x,y,z}`,
`nav_p/q/r` (last body rates), `nav_a{x,y,z}` (last body specific force),
`nav_valid`, and a full `nav_vehicle_state` (`VehicleState`) message.

**Parameters** (defaults): `world_origin_altitude` (`0.0`), `gnss_pos_std_m`
(`4.0`), `gnss_vel_std_mps` (`0.15`), `baro_alt_std_m` (`3.0`), `accel_noise_std`
(`0.05`), `gyro_noise_std` (`0.002`), `accel_bias_rw` (`1e-3`), `gyro_bias_rw`
(`1e-5`), `init_pos_std_m` (`50.0`), `init_vel_std_mps` (`5.0`), `init_att_std_rad`
(`0.1`). (The GNSS down-position uses a \(1.5\times\) inflated standard
deviation to reflect poorer vertical GNSS geometry.)

### 2.4 `target_nav_ekf_fmu.py` — Threat 9-state kinematic tracker

Estimates the threat missile's NED trajectory from the ego seekers with a
nearly-constant-acceleration (Singer-like) process model.

- **State (9):** \(x = [p_N, p_E, p_D,\ v_N, v_E, v_D,\ a_N, a_E, a_D]\).
- **Process:** \(\dot p = v,\ \dot v = a,\ \dot a = \text{white}\).
- **Measurements** (relative to the ego nav solution): radar →
  \([\text{range}, \text{az}, \text{el}, \dot{\text{range}}]\) (full 3-D fix);
  IR → \([\text{az}, \text{el}]\) (bearing-only). Measurement Jacobians are formed
  by central-free forward finite differences (robust for the mixed range/angle
  observation). The tracker locks (and sets `tgt_valid`) on the first radar
  return with range `> 1 m`, initializing target position along the measured
  line of sight. Angle innovations are wrapped to \([-\pi, \pi]\).

**Inputs** (aux): `ego_pos_{n,e,d}`, `ego_vel_{n,e,d}`; radar `radar_range_m`,
`radar_az_rad`, `radar_el_rad`, `radar_range_rate_mps`, `radar_ready`; IR
`ir_az_rad`, `ir_el_rad`, `ir_ready`.

**Outputs** (aux): `tgt_pos_{n,e,d}`, `tgt_vel_{n,e,d}`, `tgt_acc_{n,e,d}`,
`tgt_valid`, `est_range_m`.

**Parameters** (defaults): `accel_process_std` (`30.0`), `radar_range_std_m`
(`15.0`), `radar_angle_std_rad` (`0.005`), `radar_rate_std_mps` (`3.0`),
`ir_angle_std_rad` (`0.001`), `init_pos_std_m` (`100.0`), `init_vel_std_mps`
(`150.0`), `init_acc_std_mps2` (`100.0`).

### 2.5 `qp_solver.py` — Compact convex QP solver

A pure-NumPy primal-dual interior-point method (Mehrotra predictor-corrector)
solving

\[
\min_x\ \tfrac12 x^\top P x + q^\top x \quad\text{s.t.}\quad Gx \le h,
\]

with \(P\) symmetric positive semi-definite. `solve_qp(P, q, G, h, max_iter=30,
tol=1e-8)` returns `(x, info)` where `info` reports `converged`, `iterations`,
and `status`. Design choices: it symmetrizes and mildly regularizes \(P\); with
no constraints it returns the unconstrained minimizer directly; and it always
returns a usable control even without full convergence (falling back to the
projected unconstrained solution) so real-time guidance never stalls. This is
exactly the QP class produced by the condensed terminal-guidance MPC.

### 2.6 `uncertainty.py` — UQ & probability-of-kill helpers

Dependency-light statistical primitives:

- `carleton_b_from_lethal_radius(R_L)` — Carleton parameter \(b = R_L/\sqrt{2\ln 2}\).
- `pk_given_miss(miss, R_L, model)` — conditional kill probability; `"carleton"`
  diffuse-Gaussian \(\exp(-r^2/2b^2)\) or `"cookie"` cookie-cutter (1 inside
  \(R_L\), else 0).
- `sspk_rayleigh_carleton(sigma, R_L)` — closed-form SSPK \(= b^2/(b^2+\sigma^2)\).
- `sspk_monte_carlo(miss_distances, R_L, model, confidence)` — Monte-Carlo SSPK
  with a confidence interval (Wilson interval for cookie-cutter; CLT interval on
  the mean of per-run kill probabilities for Carleton), plus reported CEP, mean
  miss, and \(\sigma\).
- `wilson_interval(k, n, confidence)` — Wilson score interval for a binomial
  proportion \(k/n\).
- `sample_params(spec, rng)` — draw parameters from `normal` / `uniform` /
  `lognormal` specs.
- `propagate_linear(f, x0, cov_x, eps)` — first-order (delta-method)
  mean/covariance propagation \(\Sigma_y = J\,\Sigma_x J^\top\).

### 2.7 `sixdof.py` and `atmosphere.py` — Shared helpers

`sixdof.py` provides quaternion utilities (`quat_normalize`, `rot_from_quat`,
`quat_from_euler`, `euler_from_quat`, Hamilton product `quat_mul`, kinematics
`quat_deriv`), aerodynamic-angle extraction `alpha_beta`, and full 6-DOF
rigid-body integrators — a fourth-order Runge-Kutta step
(`integrate_6dof_rk4`) and a semi-implicit Euler step (`integrate_6dof`) — using
the 13-element state \([p, v, q_{xyzw}, \omega]\).

`atmosphere.py` implements the ICAO International Standard Atmosphere over four
layers (sea level to 47 km): `isa(h)` returns \((T, P, \rho, a)\), with
convenience wrappers `temperature_K`, `pressure_Pa`, `density_kgm3`,
`speed_of_sound_mps`, `mach`, `dynamic_pressure_Pa`, and the inverse
`pressure_altitude_m`.

---

## 3. Mathematical Formulation

Symbols: \(p_e, v_e\) ego position/velocity (NED); \(p_t, v_t, a_t\) target
position/velocity/acceleration; relative range vector \(r = p_t - p_e\) and
relative velocity \(v = v_t - v_e\); \(\hat r = r/\lVert r\rVert\).

### 3.1 Fire-control phase logic and engagement kinematics

Every active step the computer forms the range, closing speed, time-to-go, and
line-of-sight (LOS) rotation rate:

\[
\rho = \lVert r\rVert,\qquad
V_c = \max\!\Big(-\tfrac{r\cdot v}{\rho},\ V_{c,\min}\Big),\qquad
t_{go} = \frac{\rho}{V_c},
\]

\[
\boldsymbol{\omega}_{\text{LOS}} = \frac{r \times v}{\rho^2},\qquad
\dot\lambda = \lVert \boldsymbol{\omega}_{\text{LOS}}\rVert .
\]

The phase is `TERMINAL` when a valid track exists and \(\rho \le
R_{\text{term}}\) (`terminal_range_m`), otherwise `MIDCOURSE`; with neither track
nor cue (or `nav_valid` false) the phase is `IDLE` and \(a_{\text{cmd}} = 0\).
The final command is magnitude-limited to \(a_{\max} = \texttt{max\_accel\_g}\cdot
g\), then multiplied by the soft-start ramp factor \(\min(1, (t -
t_{\text{active},0})/\texttt{command\_ramp\_s})\).

### 3.2 Predicted Intercept Point (PIP)

For a constant-velocity target and a missile flying at its current speed \(V_m =
\lVert v_e\rVert\), the intercept time \(t\) solves
\(\lVert p_t + v_t t - p_e\rVert = V_m t\). With \(r_0 = p_t - p_e\) this is the
quadratic \(a t^2 + b t + c = 0\):

\[
a = v_t\!\cdot\! v_t - V_m^2,\qquad
b = 2\,r_0\!\cdot\! v_t,\qquad
c = r_0\!\cdot\! r_0 .
\]

The smallest positive root is taken (or the linear root \(t = -c/b\) when
\(|a|\) is negligible); if no valid root exists it falls back to the
straight-line estimate \(t = \lVert r_0\rVert / V_m\). Then

\[
\text{PIP} = p_t + v_t\, t .
\]

**Midcourse lead-collision steering.** With \(\hat v = v_e/\lVert v_e\rVert\), the
unit direction to the PIP \(\hat\ell\), and the heading-error component
\(\ell_\perp = \hat\ell - (\hat\ell\cdot\hat v)\hat v\):

\[
a_{\text{cmd}} = K_{\text{mid}}\,\frac{\lVert v_e\rVert^2}{\max(d, 1)}\,\ell_\perp ,
\qquad d = \lVert \text{PIP} - p_e\rVert,
\]

limited to \(\min(a_{\max}, \texttt{midcourse\_max\_g}\cdot g)\).

### 3.3 Terminal PropNav / Augmented PN

True vector Proportional Navigation with effective navigation ratio \(N'\):

\[
a_{\text{cmd}} = N'\, V_c\, \big(\boldsymbol{\omega}_{\text{LOS}} \times \hat r\big).
\]

When `augmented_propnav` is enabled, the component of the estimated target
acceleration perpendicular to the LOS is added:

\[
a_{t,\perp} = a_t - (a_t\cdot\hat r)\,\hat r,\qquad
a_{\text{cmd}} \mathrel{+}= \tfrac12\, N'\, a_{t,\perp}.
\]

**ZEM diagnostic (not a control law).** The perpendicular zero-effort-miss
reported in `zem_m` is

\[
\mathbf{z} = r + v\,t_{go} + \tfrac12 a_t\, t_{go}^2,\qquad
\text{ZEM} = \big\lVert \mathbf{z} - (\mathbf{z}\cdot\hat r)\,\hat r\big\rVert .
\]

### 3.4 Terminal MPC (condensed constrained QP)

Per NED axis the relative engagement is a double integrator with state
\(x_k = [r_k, \dot r_k]^\top\), missile acceleration \(u_k\) as control, and target
acceleration \(a_t\) as a known disturbance. With per-step interval
\(T = \max(\texttt{mpc\_min\_dt},\ t_{go}/N)\):

\[
x_{k+1} = A x_k + B\,(a_t - u_k),\qquad
A = \begin{bmatrix}1 & T\\ 0 & 1\end{bmatrix},\quad
B = \begin{bmatrix}\tfrac12 T^2\\ T\end{bmatrix}.
\]

The terminal relative position is condensed over the horizon. With
\(e_1 = [1, 0]\) and the control-sensitivity coefficients
\(g_k = e_1 A^{\,N-1-k} B\):

\[
r_N = \underbrace{e_1 A^N x_0 + \textstyle\sum_k g_k\, a_t}_{f_0}\ -\ g^\top U,
\qquad U = [u_0, \dots, u_{N-1}]^\top .
\]

The cost trades terminal miss against control effort and control rate (slew),
using the first-difference matrix \(D = I - I_{-1}\):

\[
J = w_{\text{miss}}\, r_N^2 + w_{\text{eff}}\lVert U\rVert^2 + w_{\text{rate}}\lVert D U\rVert^2 .
\]

This is the QP \(\min_U \tfrac12 U^\top P U + q^\top U\) s.t. \(|u_k| \le a_{\max}\),
with

\[
P = 2\big(w_{\text{miss}}\, g g^\top + w_{\text{eff}} I + w_{\text{rate}} D^\top D\big),
\]
\[
q = 2\big(-w_{\text{miss}} f_0\, g + w_{\text{rate}} D^\top r_{\text{lin}}\big),
\qquad r_{\text{lin}} = [-u_{\text{prev}}, 0, \dots, 0]^\top,
\]

where the \(r_{\text{lin}}\) term couples the first move \(u_0\) to the previously
applied command (rate penalty). The actuator limits are encoded as
\(G = [I;\,-I]\), \(h = a_{\max}\mathbf 1\). The QP is solved independently for each
axis and only the first move \(u_0\) is applied (receding horizon).

### 3.5 QP interior-point solver (KKT / Newton + Mehrotra)

For \(\min \tfrac12 x^\top P x + q^\top x\) s.t. \(Gx \le h\), introduce slacks
\(s = h - Gx \ge 0\) and duals \(z \ge 0\). The perturbed KKT conditions are

\[
Px + q + G^\top z = 0,\qquad
Gx + s - h = 0,\qquad
s_i z_i = \mu,\ \ s, z > 0 .
\]

The solver forms the residuals \(r_d = Px + q + G^\top z\), \(r_p = Gx + s - h\),
and duality measure \(\mu = s^\top z/m\). Eliminating \((\Delta s, \Delta z)\) gives
the normal-equation (Schur-complement) system for \(\Delta x\)

\[
\underbrace{\big(P + G^\top W G\big)}_{H}\,\Delta x = -\big(r_d + G^\top(W r_p - r_{\text{cent}}\oslash s)\big),
\qquad W = \operatorname{diag}(z_i/s_i),
\]

solved by a Cholesky factorization \(H = LL^\top\) (with adaptive regularization
if indefinite). The **Mehrotra predictor-corrector** first takes an affine
(predictor) step with \(r_{\text{cent}} = s\odot z\), estimates the achievable
duality measure \(\mu_{\text{aff}}\), sets the centering parameter
\(\sigma = (\mu_{\text{aff}}/\mu)^3\), then solves a corrector step with
\(r_{\text{cent}} = s\odot z + \Delta s_{\text{aff}}\odot\Delta z_{\text{aff}} -
\sigma\mu\mathbf 1\). Steps are limited to the boundary (\(s, z \ge 0\)) with a
0.99 fraction-to-boundary rule.

### 3.6 Three-loop autopilot

**Command rotation.** The required aerodynamic body specific force subtracts
gravity, since the fins must both maneuver and hold the airframe against gravity:

\[
f_{\text{req}}^{b} = R(q)^{-1}\big(a_{\text{cmd}}^{\text{NED}} - g^{\text{NED}}\big),
\qquad g^{\text{NED}} = [0,0,g]^\top,
\]

clamped per axis to `max_accel_cmd_mps2`, giving \(a_{z,\text{req}} = f^b_{z}\) and
\(a_{y,\text{req}} = f^b_{y}\).

**Outer loop — acceleration → incidence trim.** The quasi-static trim
\(a_z = -(\bar q S/m)\, C_{N_\alpha}\,\alpha\) inverts to the feed-forward incidence,
clamped to \(\alpha_{\max}\) (`max_incidence_rad`):

\[
\alpha_{ff} = \operatorname{clamp}\!\Big(\!-\frac{a_{z,\text{req}}\, m}{\bar q S\, C_{N_\alpha}}\Big),
\qquad
\beta_{ff} = \operatorname{clamp}\!\Big(\frac{a_{y,\text{req}}\, m}{\bar q S\, C_{Y_\beta}}\Big).
\]

The incidence error is closed by a slow integral (trimming out steady model error
and gravity), with anti-windup limit \(\alpha_{\max}/k_i\):

\[
I_\theta \mathrel{+}= (\alpha_{ff} - \alpha)\,\Delta t,\qquad
\alpha_{\text{cmd}} = \operatorname{clamp}\big(\alpha_{ff} + k_i\, I_\theta\big),
\]

and identically for \(\beta_{\text{cmd}}\) with \(I_\psi\), where \(k_i =
\texttt{ki\_accel}\). Incidences are measured from the nav velocity resolved into
body axes: \(\alpha = \operatorname{atan2}(w_b, u_b)\),
\(\beta = \arcsin(v_b/V)\).

**Inner loop — trim feed-forward + rate damping.** A statically stable airframe
self-trims to the commanded incidence via the feed-forward fin, so no fast
incidence/acceleration feedback is used; only the low-latency gyro closes the
fast (minimum-phase) loop:

\[
\delta_{e} = -\frac{C_{m_\alpha}}{C_{m_{\delta e}}}\,\alpha_{\text{cmd}} + k_q^{p}\, q_{\text{gyro}},
\qquad
\delta_{r} = -\frac{C_{n_\beta}}{C_{n_{\delta r}}}\,\beta_{\text{cmd}} + k_q^{r}\, r_{\text{gyro}} .
\]

For `controller_type = "pid"`, \(k_q^p = k_q^r = \texttt{kq\_rate}\). For `"lqr"`
the gains are computed on-line (§3.7).

**Roll (wings-level P-D).** With bank angle \(\phi\) and roll rate \(p\):

\[
C_{l,\text{cmd}} = -\big(k_{p,\phi}\,\phi + k_{d,\phi}\, p\big),
\qquad
\delta_a = \frac{C_{l,\text{cmd}}}{C_{l_{\delta a}}} .
\]

All fin commands are clamped to `max_fin_rad`. The reported achieved specific
forces are \(a_{z,\text{ach}} = -(\bar q S/m) C_{N_\alpha}\alpha\) and
\(a_{y,\text{ach}} = (\bar q S/m) C_{Y_\beta}\beta\).

### 3.7 LQR short-period rate gain

At the live flight condition the short-period model \(\{\text{incidence},
\text{rate}\}\) is built with dimensional derivatives (pitch axis shown;
\(V\) = airspeed, \(I\) = \(I_{yy}\)):

\[
Z_\alpha = -\frac{\bar q S\, C_{N_\alpha}}{m V},\quad
M_\alpha = \frac{\bar q S d\, C_{m_\alpha}}{I},\quad
M_q = \frac{\bar q S d^2\, C_{m_q}}{2 V I},
\]
\[
M_{\delta} = \frac{\bar q S d\, C_{m_{\delta e}}}{I},\quad
Z_{\delta} = -\frac{\bar q S\, C_{N_{\delta e}}}{m V},
\]
\[
A = \begin{bmatrix} Z_\alpha & 1\\ M_\alpha & M_q\end{bmatrix},\qquad
B = \begin{bmatrix} Z_{\delta}\\ M_{\delta}\end{bmatrix}.
\]

With \(Q = \operatorname{diag}(\texttt{lqr\_q\_angle},\ \texttt{lqr\_q\_rate})\) and
\(R = [\texttt{lqr\_r\_fin}]\), the continuous-time algebraic Riccati equation

\[
A^\top P + P A - P B R^{-1} B^\top P + Q = 0
\]

is solved (`scipy.linalg.solve_continuous_are`) for the optimal gain
\(K = R^{-1} B^\top P\); the rate-feedback term used for damping is \(k_q = -K_2\)
(the sign opposes the body rate through the negative fin-moment derivative). If
the Riccati solve fails it falls back to `kq_rate`. The yaw axis is identical with
\(\{C_{Y_\beta}, C_{n_\beta}, C_{n_r}, C_{Y_{\delta r}}, C_{n_{\delta r}}, I_{zz}\}\).

### 3.8 Ego INS EKF (quaternion, error-state)

**Kinematics / propagation** with \(\omega = \omega_{\text{meas}} - b_g\) and
\(f = f_{\text{meas}} - b_a\):

\[
\dot q = \tfrac12\, q \otimes \begin{bmatrix}\omega\\ 0\end{bmatrix},\qquad
\dot v = R(q) f - g,\qquad
\dot p = v .
\]

The attitude increment is applied as \(q \leftarrow q \otimes \operatorname{Exp}(\omega
\Delta t)\). The local (body) **error dynamics** are

\[
\dot{\delta p} = \delta v,\quad
\dot{\delta v} = -R[f\times]\,\delta\theta - R\,\delta b_a,\quad
\dot{\delta\theta} = -[\omega\times]\,\delta\theta - \delta b_g,
\]

with biases as random walks, discretized to first order:

\[
F = \begin{bmatrix}
I & I\Delta t & 0 & 0 & 0\\
0 & I & -R[f\times]\Delta t & -R\Delta t & 0\\
0 & 0 & I - [\omega\times]\Delta t & 0 & -I\Delta t\\
0 & 0 & 0 & I & 0\\
0 & 0 & 0 & 0 & I
\end{bmatrix},
\qquad P \leftarrow F P F^\top + Q .
\]

**GNSS update** (position + velocity, \(H = [I_6\ \ 0]\)); **baro update**
(down-position, \(H = e_3^\top\) on \(\delta p\)). The Joseph-form covariance
update is used for stability:

\[
K = P H^\top (H P H^\top + R)^{-1},\quad
P \leftarrow (I - KH) P (I - KH)^\top + K R K^\top,
\]

and the error state is injected (quaternion via \(q \leftarrow q \otimes
\operatorname{Exp}(\delta\theta)\)) then reset.

### 3.9 Target kinematic EKF (constant acceleration)

Constant-acceleration transition over \(\Delta t\):

\[
F = \begin{bmatrix} I & I\Delta t & \tfrac12 I\Delta t^2\\ 0 & I & I\Delta t\\ 0 & 0 & I\end{bmatrix},
\qquad P \leftarrow F P F^\top + Q,
\]

with process noise loaded on acceleration (and a coupled velocity term). The
nonlinear measurement model on the relative vector \(r = p_t - p_e\),
\(v = v_t - v_e\) is

\[
\text{range} = \lVert r\rVert,\quad
\text{az} = \operatorname{atan2}(r_E, r_N),\quad
\text{el} = \operatorname{atan2}(-r_D, \sqrt{r_N^2+r_E^2}),\quad
\dot{\text{range}} = \frac{r\cdot v}{\lVert r\rVert},
\]

with radar observing all four and IR observing \((\text{az}, \text{el})\). The
Jacobian \(H = \partial h/\partial x\) is formed by forward finite differences,
angle innovations are wrapped to \([-\pi,\pi]\), and the same Joseph-form Kalman
update is applied.

### 3.10 Probability of kill

The **Carleton** diffuse-Gaussian damage function maps a miss distance \(r\) to a
conditional kill probability, with \(b\) chosen so \(P(R_L) = 0.5\):

\[
b = \frac{R_L}{\sqrt{2\ln 2}},\qquad P_{k|\text{miss}}(r) = \exp\!\Big(-\frac{r^2}{2 b^2}\Big).
\]

For a circular-Gaussian miss with per-axis standard deviation \(\sigma\), the
single-shot kill probability integrates in closed form to

\[
\text{SSPK} = \frac{b^2}{b^2 + \sigma^2}.
\]

The **Monte-Carlo** estimator averages the per-run kill probability
\(\hat p = \tfrac1n\sum_i P_{k|\text{miss}}(r_i)\). For the cookie-cutter model each
run is Bernoulli and the interval is the **Wilson score interval**

\[
\text{CI} = \frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} ,
\]

while for the continuous Carleton model a CLT interval \(\hat p \pm z\,
\text{SE}\) is reported. The two-sided normal quantile \(z\) is evaluated with
Acklam's inverse-normal (probit) approximation.

---

## 4. Uncertainty Quantification & \(P_{\text{kill}}\)

The UQ workflow connects surrogate/model uncertainty to a probability of
interceptor success with confidence bounds:

1. **Parameter sampling.** `sample_params` draws uncertain inputs (aero
   coefficients, sensor noise, target maneuver, etc.) from `normal`, `uniform`,
   or `lognormal` distributions defined in an uncertainty spec.
2. **Miss-distance generation.** Each sampled parameter set is run through the
   engagement (closed-loop simulation, or a cheaper surrogate) to produce a
   terminal miss distance. When a full Monte-Carlo pass is too expensive,
   `propagate_linear` gives a first-order (delta-method) mean and covariance
   \(\Sigma_y = J\Sigma_x J^\top\) of the outcome using finite-difference
   sensitivities \(J\).
3. **Lethality mapping.** Each miss is converted to a conditional kill
   probability with the Carleton (or cookie-cutter) damage function keyed to the
   warhead lethal radius \(R_L\).
4. **\(P_{\text{kill}}\) with bounds.** `sspk_monte_carlo` reports the point
   estimate \(\hat p\), a confidence interval (Wilson for cookie-cutter, CLT for
   Carleton), and summary miss statistics (CEP, mean miss, \(\sigma\)). When the
   miss distribution is approximately circular-Gaussian, the closed-form
   `sspk_rayleigh_carleton` provides an analytic cross-check \(b^2/(b^2+\sigma^2)\).

The result is a defensible statement of the form "SSPK = \(\hat p\)
(95% CI \([p_{\text{low}}, p_{\text{high}}]\)) at lethal radius \(R_L\), from \(n\)
samples," directly usable for engagement-effectiveness analysis and design
trade studies.

---

## 5. References

1. P. Zarchan, *Tactical and Strategic Missile Guidance*, 7th ed., Progress in
   Astronautics and Aeronautics, AIAA, 2019 — Proportional Navigation, Augmented
   PN, ZEM formulation, and missile autopilot design.
2. G. M. Siouris, *Missile Guidance and Control Systems*, Springer, 2004 —
   guidance-law derivations and inner-loop autopilot architecture.
3. N. A. Shneydor, *Missile Guidance and Pursuit: Kinematics, Dynamics and
   Control*, Horwood Publishing, 1998 — pursuit / lead-collision geometry and
   LOS kinematics.
4. F. W. Nesline and P. Zarchan, "Why Modern Controllers Can Go Unstable in
   Practice," *Journal of Guidance, Control, and Dynamics*, 7(4):495–500, 1984;
   and F. W. Nesline, B. H. Wells, and P. Zarchan, "Combined Optimal/Classical
   Approach to Robust Missile Autopilot Design," *J. Guidance and Control*,
   4(3):316–322, 1981 — three-loop acceleration autopilot design.
5. J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, *Model Predictive Control:
   Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2017 —
   receding-horizon control and condensed QP formulation.
6. S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University
   Press, 2004 — Ch. 11 (interior-point methods), §16.1 (QP).
7. J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
   2006 — Ch. 16 (QP), Mehrotra predictor-corrector (Algorithm 16.4).
8. S. Mehrotra, "On the Implementation of a Primal-Dual Interior Point Method,"
   *SIAM Journal on Optimization*, 2(4):575–601, 1992.
9. A. E. Bryson and Y.-C. Ho, *Applied Optimal Control: Optimization,
   Estimation, and Control*, Hemisphere/Taylor & Francis, 1975 — LQR and the
   algebraic Riccati equation.
10. P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated
    Navigation Systems*, 2nd ed., Artech House, 2013 — strapdown INS mechanization
    and error-state (multiplicative) EKF.
11. R. E. Ball, *The Fundamentals of Aircraft Combat Survivability Analysis and
    Design*, 2nd ed., AIAA, 2003 — Carleton damage function, lethal radius, and
    probability-of-kill methodology.
12. E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical
    Inference," *Journal of the American Statistical Association*,
    22(158):209–212, 1927 — Wilson score confidence interval.

*Additional context:* the ICAO Standard Atmosphere follows ICAO Doc 7488; the
inverse-normal quantile uses P. J. Acklam's rational approximation of the probit
function.
