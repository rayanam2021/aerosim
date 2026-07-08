# SHIFT Interceptor — Sensor FMUs

Sensor Functional Mock-up Units (FMUs) for the **SHIFT** missile-interception
scenario, part of the AeroSim distributed co-simulation. Every model in this
folder is a [`pythonfmu3`](https://pypi.org/project/pythonfmu3/) `Fmi3Slave`
that turns the perfect-truth plant kinematics into realistic, noisy, biased,
and **asynchronously sampled** measurements. Those measurements are the only
information the ego and target navigation filters are allowed to see.

---

## 1. Executive Summary

The interception scenario pits an **ego interceptor** against a maneuvering
**threat missile**. The interceptor cannot act on truth; it must estimate its
own state and the threat's state from imperfect sensors. This folder provides
that sensor suite, split into two functional groups:

**Ego inertial navigation system (INS) sensors** — self-localization:

| Sensor | Measures | Default rate | Frame |
| --- | --- | --- | --- |
| `imu_fmu` | Body-frame specific force + angular rate | 100 Hz | Body |
| `gnss_fmu` | NED position + velocity | 10 Hz | NED |
| `baro_fmu` | Static pressure + pressure altitude | 25 Hz | Vertical (MSL) |

**Target-tracking seekers** — relative geometry to the threat:

| Sensor | Measures | Default rate | Frame |
| --- | --- | --- | --- |
| `ir_seeker_fmu` | LOS azimuth + elevation (bearing-only) | 100 Hz | NED (INS-stabilized) |
| `semi_active_radar_fmu` | Range, azimuth, elevation, range-rate | 20 Hz | NED (INS-stabilized) |

Two design themes run through the whole suite:

- **Realistic error models.** Each sensor applies the error mechanisms that
  actually dominate its class of hardware: the IMU carries bias, scale-factor,
  and white noise (whose per-sample sigma is derived from a noise *density*);
  GNSS carries per-axis Gaussian position/velocity noise plus optional signal
  dropout; the barometer injects pressure noise and a fixed calibration bias;
  the seekers add angular (and, for radar, range and Doppler) white noise plus
  detection-range and field-of-view gating.

- **Mismatched, asynchronous sample rates.** No two sensors are guaranteed to
  produce a fresh sample on the same simulation tick. Every FMU steps on every
  tick but only emits a new measurement at its own `update_rate_hz`, raising a
  `measurement_ready` strobe (`1.0`) on the tick it fires and holding it at
  `0.0` otherwise. The downstream Extended Kalman Filters (EKFs) in the
  `aerosim-controllers` package key their measurement-update steps off these
  strobes, so the fast IMU drives strapdown propagation while the slower GNSS,
  baro, radar, and IR updates fold in opportunistically whenever they arrive.
  This reproduces the real-world engineering problem of fusing a fast dead-
  reckoning channel with several slow, heterogeneous aiding channels.

The `atmosphere.py` module provides the shared ICAO Standard Atmosphere used by
the barometer (and elsewhere in the missile stack) to relate altitude and
static pressure.

---

## 2. Component Descriptions

All FMUs share the same lifecycle conventions:

- Random draws use a per-FMU `numpy` generator seeded from an `rng_seed`
  parameter, re-seeded in `enter_initialization_mode`, so runs are repeatable.
- `_elapsed_since_update` (or `_elapsed`) is pre-loaded to the full update
  interval in initialization, which **forces a measurement on the first step**.
- The `time` variable is `independent` (FMI clock); each sensor also publishes
  its own timestamp output (`imu_time_s`, `gnss_time_s`, ...).

### 2.1 `imu_fmu` — Strapdown Inertial Measurement Unit

Emulates a tactical-grade strapdown MEMS IMU: it reports **body-frame specific
force** (accelerometer) and **body-frame angular rate** (gyroscope).

- **Input** (component topic): `vehicle_state` (`aerosim::types::VehicleState`)
  — provides NED `velocity` and body `orientation` quaternion.
- **Outputs** (aux): `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2`,
  `gyro_x_rps`, `gyro_y_rps`, `gyro_z_rps`, `measurement_ready`, `imu_time_s`.
- **Parameters and defaults:**

  | Parameter | Default | Meaning |
  | --- | --- | --- |
  | `update_rate_hz` | `100.0` | Sample rate |
  | `accel_noise_density_mps2_per_sqrthz` | `3e-3` | Accel noise density (m/s²/√Hz) |
  | `accel_bias_std_mps2` | `0.05` | Std. dev. of the fixed accel bias draw |
  | `accel_scale_error_ppm` | `500.0` | Accel scale-factor error (ppm) |
  | `gyro_noise_density_rps_per_sqrthz` | `1e-4` | Gyro angle-random-walk density (rad/s/√Hz) |
  | `gyro_bias_std_rps` | `5e-4` | Std. dev. of the fixed gyro bias draw |
  | `gyro_scale_error_ppm` | `300.0` | Gyro scale-factor error (ppm) |
  | `rng_seed` | `42` | Random seed |

- **Behavior.** The accelerometer output is the **specific force**
  \(f = R^{\mathsf T}(a - g)\) (see §3.1), not the kinematic acceleration.
  Kinematic acceleration is differentiated numerically from the NED velocity
  (\(a = \Delta v / \Delta t\)). The gyroscope rate is obtained from the
  **body-frame** attitude increment between consecutive samples (a recent bug
  fix — see §3.2). The white-noise per-sample sigma is derived from the noise
  density as \(\sigma = \text{density}\times\sqrt{\text{update\_rate\_hz}}\).
  Biases are drawn once at initialization and held fixed for the run.

### 2.2 `gnss_fmu` — GNSS / GPS Receiver

Low-rate absolute NED position and Doppler velocity fix.

- **Input** (component topic): `vehicle_state` — NED `position` and `velocity`.
- **Outputs** (aux): `pos_n_m`, `pos_e_m`, `pos_d_m`, `vel_n_mps`, `vel_e_mps`,
  `vel_d_mps`, `measurement_ready`, `gnss_time_s`.
- **Parameters and defaults:**

  | Parameter | Default | Meaning |
  | --- | --- | --- |
  | `update_rate_hz` | `10.0` | Sample rate |
  | `pos_noise_horizontal_std_m` | `3.0` | Horizontal (N, E) position noise std. dev. |
  | `pos_noise_vertical_std_m` | `5.0` | Vertical (D) position noise std. dev. (~1.5× worse) |
  | `vel_noise_std_mps` | `0.1` | Per-axis velocity noise std. dev. |
  | `dropout_probability` | `0.0` | Per-update probability of a dropped fix |
  | `rng_seed` | `7` | Random seed |

- **Behavior.** Each fresh update adds independent zero-mean Gaussian noise to
  every NED position and velocity axis. When `dropout_probability > 0`, a
  uniform draw per update interval may suppress the fix entirely
  (`measurement_ready` stays `0.0`), simulating jamming or blockage.

### 2.3 `baro_fmu` — Barometric Altimeter

Reports static pressure and **pressure altitude** via the ISA relation.

- **Input** (component topic): `vehicle_state` — NED `position.z` (Down).
- **Outputs** (aux): `pressure_pa`, `baro_alt_m`, `measurement_ready`,
  `baro_time_s`.
- **Parameters and defaults:**

  | Parameter | Default | Meaning |
  | --- | --- | --- |
  | `update_rate_hz` | `25.0` | Sample rate |
  | `pressure_noise_std_pa` | `5.0` | Static-pressure noise std. dev. (Pa) |
  | `altitude_bias_std_m` | `2.0` | Std. dev. of the fixed calibration-offset draw (m) |
  | `world_origin_altitude` | `0.0` | MSL altitude of the NED origin (m) |
  | `rng_seed` | `99` | Random seed |

- **Behavior.** True MSL altitude is `world_origin_altitude − pos.z`. The ISA
  model gives the true static pressure at that altitude; Gaussian pressure
  noise is added (floored at 1 Pa), and the noisy pressure is inverted back
  through the ISA pressure-altitude relation. A fixed altitude bias (drawn once
  at initialization) is then added, mimicking a residual calibration offset.
  The barometer is immune to GNSS jamming, so it complements GNSS on the
  altitude channel.

### 2.4 `ir_seeker_fmu` — Passive Infrared Seeker

A passive, body-fixed/gimbaled IR seeker. Being passive it provides **no
range** — only line-of-sight (LOS) bearing to the threat — which is precisely
why the target EKF fuses it with the radar's range channel.

- **Inputs** (aux): `ego_pos_{n,e,d}`, `ego_q{w,x,y,z}`, `tgt_pos_{n,e,d}`.
  (Ego and threat kinematics arrive as scalar aux signals because a single FMU
  cannot bind two `VehicleState` component inputs in AeroSim.)
- **Outputs** (aux): `ir_az_rad`, `ir_el_rad`, `ir_locked`,
  `measurement_ready`, `ir_time_s`.
- **Parameters and defaults:**

  | Parameter | Default | Meaning |
  | --- | --- | --- |
  | `update_rate_hz` | `100.0` | Sample rate |
  | `max_lock_range_m` | `15000.0` | Maximum lock range (thermal roll-off, hard-gated) |
  | `fov_halfangle_rad` | `0.785398` | Gimbal/FOV half-angle (45°) about boresight |
  | `angle_noise_std_rad` | `0.001` | Angular white-noise std. dev. (rad) |
  | `rng_seed` | `123` | Random seed |

- **Behavior.** The relative position \(r = p_\text{tgt} - p_\text{ego}\) sets
  the range and LOS unit vector. Lock requires both `range ≤ max_lock_range_m`
  and the off-boresight angle (angle between LOS and the ego body +x axis) to be
  within `fov_halfangle_rad`. The reported azimuth/elevation are computed
  directly from the NED components of \(r\) (an INS-stabilized output), with
  independent Gaussian angular noise on each. When gating fails, `ir_locked`
  and `measurement_ready` are both cleared.

### 2.5 `semi_active_radar_fmu` — Semi-Active Radar Seeker

A semi-active radar homing (SARH) seeker: an external illuminator floods the
threat and the missile receiver measures the reflected return, yielding a full
3-D fix — range, LOS bearing, and Doppler range-rate — at coarser angular
accuracy and a lower rate than the IR seeker.

- **Inputs** (aux): `ego_pos_{n,e,d}`, `ego_vel_{n,e,d}`, `ego_q{w,x,y,z}`,
  `tgt_pos_{n,e,d}`, `tgt_vel_{n,e,d}`.
- **Outputs** (aux): `radar_range_m`, `radar_az_rad`, `radar_el_rad`,
  `radar_range_rate_mps`, `radar_locked`, `measurement_ready`, `radar_time_s`.
- **Parameters and defaults:**

  | Parameter | Default | Meaning |
  | --- | --- | --- |
  | `update_rate_hz` | `20.0` | Sample rate |
  | `max_detection_range_m` | `40000.0` | Maximum detection range (radar-equation roll-off, hard-gated) |
  | `fov_halfangle_rad` | `0.523599` | FOV half-angle (30°) about boresight |
  | `range_noise_std_m` | `15.0` | Range noise std. dev. (m) |
  | `range_rate_noise_std_mps` | `3.0` | Range-rate (Doppler) noise std. dev. (m/s) |
  | `angle_noise_std_rad` | `0.005` | Angular white-noise std. dev. (rad) |
  | `rng_seed` | `321` | Random seed |

- **Behavior.** Uses the relative position \(r\) and relative velocity
  \(v = v_\text{tgt} - v_\text{ego}\). Gating is identical in form to the IR
  seeker but with radar limits. Range, azimuth, elevation, and closing
  range-rate are each corrupted by independent Gaussian noise. The angular
  channel is deliberately noisier than the IR seeker's, which is why the two are
  complementary: IR gives precise bearing, radar gives range and Doppler.

### 2.6 `atmosphere.py` — ICAO Standard Atmosphere

Shared library (not an FMU). Implements the ICAO/ISA layered model from sea
level to 47 km MSL, plus helpers.

- **Constants:** `GAMMA_AIR = 1.4`, `R_AIR = 287.05287` J/(kg·K),
  `G0 = 9.80665` m/s².
- **Layers** `(base altitude m, base temperature K, lapse rate K/m)`:
  `(0, 288.15, −0.0065)`, `(11000, 216.65, 0.0)`, `(20000, 216.65, +0.0010)`,
  `(32000, 228.65, +0.0028)`, `(47000, 270.65, 0.0)`. Base pressures at each
  boundary are pre-computed once via the barometric formula.
- **API:** `isa(altitude_m) → (T, P, ρ, a)`; `temperature_K`, `pressure_Pa`,
  `density_kgm3`, `speed_of_sound_mps`, `mach`, `dynamic_pressure_Pa`, and the
  inverse `pressure_altitude_m(pressure_Pa)` used by the barometer.
- Below sea level the input is clamped to 0 m; the running `__main__` block
  self-tests the model against tabulated ICAO values.

---

## 3. Mathematical Formulation

Notation: NED (North-East-Down) is the world/navigation frame; the body frame
is right-handed with +x forward (boresight). \(R = R(q)\) is the body-to-NED
rotation matrix from the vehicle attitude quaternion \(q\); \(R^{\mathsf T}\)
rotates NED into body. Gravity in NED is \(g = [0,\,0,\,g_0]^{\mathsf T}\) with
\(g_0 = 9.80665\ \text{m/s}^2\) (Down positive).

### 3.1 IMU — Accelerometer Specific Force

An accelerometer measures **specific force**, the non-gravitational force per
unit mass, expressed in the body frame. From the NED kinematic acceleration
\(a\) and gravity \(g\):

\[
f_b \;=\; R^{\mathsf T}\,(a - g).
\]

The kinematic acceleration is obtained by numerically differentiating NED
velocity between successive IMU samples,

\[
a \;\approx\; \frac{v_k - v_{k-1}}{\Delta t},
\]

and the specific force is formed as \(f_b = R^{\mathsf T}(a + g)\) in the code,
which is the same relation with \(g = [0,0,g_0]^{\mathsf T}\) written on the
additive side — the accelerometer senses the reaction to gravity, so a vehicle
in free fall reads zero and a vehicle at rest reads \(+g_0\) on the Down axis
rotated into the body frame.

### 3.2 IMU — Body-Frame Angular Rate (Frame-Correct Increment)

The gyroscope measures the angular rate of the **body relative to the
navigation frame, resolved in the body frame**. Given the previous and current
attitudes \(q_{k-1}, q_k\) (as rotations \(R_{k-1}, R_k\)), the correct
body-frame increment is the *right* composition

\[
\Delta R \;=\; R_{k-1}^{\mathsf T}\,R_k
\quad\Longleftrightarrow\quad
\Delta R = R_{k-1}^{-1} \, R_k,
\]

and the body angular rate is the rotation vector of that increment divided by
the sample interval:

\[
\omega_b \;=\; \frac{\operatorname{Log}(\Delta R)}{\Delta t}
\;=\; \frac{\text{rotvec}\!\left(R_{k-1}^{-1} R_k\right)}{\Delta t}.
\]

> **Important (recent bug fix).** The body-frame increment is
> `delta_rot = prev_rot.inv() * rot`, i.e. \(R_{k-1}^{-1} R_k\). Using the
> *left* composition \(R_k\, R_{k-1}^{-1}\) instead would yield the increment
> resolved in the **navigation frame**, which is wrong for a body-mounted
> strapdown gyro. This distinction matters as soon as the vehicle is rotating
> and the two frames diverge.

### 3.3 IMU — Sensor Error Model

Both channels share the same error structure: scale-factor error, a fixed bias,
and additive white Gaussian noise. For a true body-frame quantity \(u\)
(\(u = f_b\) for the accelerometer, \(u = \omega_b\) for the gyro):

\[
\tilde{u} \;=\; (1 + s)\,u \;+\; b \;+\; n,
\qquad n \sim \mathcal N(0,\, \sigma^2 I),
\]

where the **scale-factor** term is \(s = \text{scale\_error\_ppm}\times 10^{-6}\)
(so `scale_a = 1 + accel_scale_error_ppm·1e-6`, `scale_g` analogously), the
**bias** \(b\) is drawn once at initialization,

\[
b_a \sim \mathcal N(0,\, \sigma_{b,a}^2 I),\qquad
b_g \sim \mathcal N(0,\, \sigma_{b,g}^2 I),
\]

(with \(\sigma_{b,a} = \) `accel_bias_std_mps2`, \(\sigma_{b,g} = \)
`gyro_bias_std_rps`), and the per-sample **white-noise** standard deviation is
derived from the spectral noise *density* and the sample rate:

\[
\sigma \;=\; \rho\,\sqrt{f_s},
\]

with \(\rho = \) `accel_noise_density_mps2_per_sqrthz` (m/s²/√Hz) or
`gyro_noise_density_rps_per_sqrthz` (rad/s/√Hz), and \(f_s = \)
`update_rate_hz`. This is the standard conversion from a continuous noise
density to a discrete per-sample sigma at the sampling bandwidth. Physically,
the bias is a slowly varying quantity that the model approximates as constant
over an engagement, and \(\rho\) captures the gyro's angle-random-walk /
accelerometer's velocity-random-walk coefficient.

### 3.4 GNSS — NED Error Model

The receiver adds independent zero-mean Gaussian noise to each true NED
position and velocity component:

\[
\begin{aligned}
\tilde{p}_N &= p_N + \eta_h, &
\tilde{p}_E &= p_E + \eta_h, &
\tilde{p}_D &= p_D + \eta_v, \\
\tilde{v}_i &= v_i + \nu_i, & & &
\end{aligned}
\]

with \(\eta_h \sim \mathcal N(0, \sigma_h^2)\) (horizontal,
`pos_noise_horizontal_std_m`), \(\eta_v \sim \mathcal N(0, \sigma_v^2)\)
(vertical, `pos_noise_vertical_std_m`, worse than horizontal), and
\(\nu_i \sim \mathcal N(0, \sigma_{v\!el}^2)\) per axis (`vel_noise_std_mps`).
Dropouts are Bernoulli per update interval with probability
`dropout_probability`; a dropped update suppresses `measurement_ready`.

### 3.5 Barometer — ISA Pressure Altitude

The ISA relates altitude and pressure per layer. Within a layer with base
altitude \(h_b\), base temperature \(T_b\), base pressure \(P_b\), and lapse
rate \(L\):

- **Gradient layer** (\(L \neq 0\)), temperature \(T = T_b + L\,(h - h_b)\):

\[
P(h) \;=\; P_b\left(\frac{T}{T_b}\right)^{\!\!-\,g_0 / (R\,L)}
\;=\; P_b\left(\frac{T_b + L(h-h_b)}{T_b}\right)^{\!\!g_0 / (R(-L))}.
\]

- **Isothermal layer** (\(L = 0\)):

\[
P(h) \;=\; P_b\,\exp\!\left(-\,\frac{g_0\,(h - h_b)}{R\,T_b}\right).
\]

The barometer inverts these to obtain **pressure altitude** from a measured
pressure \(P\) (function `pressure_altitude_m`):

\[
h \;=\; h_b + \frac{T_b}{L}\!\left[\left(\frac{P}{P_b}\right)^{\!\!R(-L)/g_0} - 1\right]
\quad (L \neq 0),
\qquad
h \;=\; h_b - \frac{R\,T_b}{g_0}\,\ln\!\frac{P}{P_b}
\quad (L = 0),
\]

selecting the layer whose base pressure still exceeds \(P\) (pressure decreases
monotonically with altitude). Here \(R = R_\text{air} = 287.05287\)
J/(kg·K). The full measurement chain is

\[
h_\text{true} = h_0 - p_D,\quad
P_\text{true} = P(h_\text{true}),\quad
\tilde P = \max(P_\text{true} + n_P,\, 1),\quad
\tilde h = h(\tilde P) + b_h,
\]

with \(n_P \sim \mathcal N(0, \sigma_P^2)\) (`pressure_noise_std_pa`), fixed
bias \(b_h \sim \mathcal N(0, \sigma_{b_h}^2)\) (`altitude_bias_std_m`) drawn at
initialization, and \(h_0 = \) `world_origin_altitude`.

The ISA density and speed of sound follow from the ideal-gas and acoustic
relations \(\rho = P/(R\,T)\) and \(a = \sqrt{\gamma R T}\), with \(\gamma = 1.4\).

### 3.6 IR Seeker — Bearing-Only LOS Geometry

Relative position in NED, range, and LOS unit vector:

\[
r = p_\text{tgt} - p_\text{ego},\qquad
\|r\| = \sqrt{r_N^2 + r_E^2 + r_D^2},\qquad
\hat{\ell} = \frac{r}{\|r\|}.
\]

**Gating.** The boresight is the ego body +x axis expressed in NED,
\(\hat{b} = R\,[1,0,0]^{\mathsf T}\). The off-boresight angle is

\[
\theta_\text{off} = \arccos\!\big(\operatorname{clip}(\hat{\ell}\cdot\hat{b},\,-1,\,1)\big),
\]

and a valid lock requires \(\|r\| \le r_\text{max}\) (`max_lock_range_m`) **and**
\(\theta_\text{off} \le \phi_\text{fov}\) (`fov_halfangle_rad`).

**Angles.** The reported azimuth and elevation are formed directly from the NED
components of \(r\) (i.e. an INS-stabilized, navigation-frame LOS), with the
Down axis negated so positive elevation is upward:

\[
\text{az} = \operatorname{atan2}(r_E,\, r_N),\qquad
\text{el} = \operatorname{atan2}\!\big(-r_D,\ \sqrt{r_N^2 + r_E^2}\big),
\]

and each is corrupted by independent zero-mean Gaussian noise:

\[
\widetilde{\text{az}} = \text{az} + n_\theta,\quad
\widetilde{\text{el}} = \text{el} + n_\theta,\quad
n_\theta \sim \mathcal N(0, \sigma_\theta^2),
\]

with \(\sigma_\theta = \) `angle_noise_std_rad`. (The body attitude \(R\) enters
only through the FOV gate; the emitted angles themselves are NED-referenced.)

### 3.7 Radar Seeker — Range, Bearing, and Range-Rate

Relative position and velocity:

\[
r = p_\text{tgt} - p_\text{ego},\qquad
v = v_\text{tgt} - v_\text{ego}.
\]

Gating uses the same off-boresight test as §3.6 with radar limits
(`max_detection_range_m`, `fov_halfangle_rad`). The measured quantities are

\[
\text{range} = \|r\|,\qquad
\text{az} = \operatorname{atan2}(r_E, r_N),\qquad
\text{el} = \operatorname{atan2}\!\big(-r_D,\ \sqrt{r_N^2 + r_E^2}\big),
\]

and the **range-rate** (rate of change of range along the LOS) is the projection
of relative velocity onto the LOS unit vector:

\[
\dot r \;=\; \frac{r \cdot v}{\|r\|} \;=\; \hat{\ell}\cdot v .
\]

> The code computes `range_rate = (r · v)/‖r‖`. This is the standard opening/
> closing-rate convention: a **negative** value means the range is decreasing
> (the interceptor and threat are closing). Equivalently one may write the
> closing rate as \(-\,(r\cdot v)/\|r\|\); the sign convention is simply a
> matter of whether "closing" is reported positive or negative, and the target
> EKF's measurement model is consistent with the FMU output.

Each channel gets independent Gaussian noise:

\[
\widetilde{\text{range}} = \text{range} + n_r,\quad
\widetilde{\text{az}} = \text{az} + n_\theta,\quad
\widetilde{\text{el}} = \text{el} + n_\theta,\quad
\widetilde{\dot r} = \dot r + n_{\dot r},
\]

with \(n_r \sim \mathcal N(0, \sigma_r^2)\) (`range_noise_std_m`),
\(n_\theta \sim \mathcal N(0, \sigma_\theta^2)\) (`angle_noise_std_rad`), and
\(n_{\dot r} \sim \mathcal N(0, \sigma_{\dot r}^2)\)
(`range_rate_noise_std_mps`).

### 3.8 Asynchronous `measurement_ready` Timing

Every FMU keeps an accumulator \(\tau\) advanced by the solver step \(\Delta t\)
each `do_step`, and a fixed interval \(T = 1/f_s\). A new measurement is emitted
only when the accumulator has filled:

\[
\tau \mathrel{+}= \Delta t;\qquad
\text{if } \tau < T - \epsilon:\ \text{measurement\_ready} = 0,\ \text{hold outputs};
\]
\[
\text{else}:\ \Delta t_\text{meas} = \tau,\ \tau \leftarrow 0,\ \text{emit sample},\ \text{measurement\_ready} = 1,
\]

with a small tolerance \(\epsilon = 10^{-9}\). The accumulator is pre-loaded to
\(T\) at initialization so the first step always fires. Because the IMU
(100 Hz), baro (25 Hz), radar (20 Hz), GNSS (10 Hz), and IR (100 Hz) each carry
their own \(T\), the strobes fire on generally different ticks — the
asynchronous, mismatched-rate behavior the navigation filters are designed to
absorb. Note the IMU passes its *actual* elapsed interval
\(\Delta t_\text{meas}\) into the measurement so its finite-difference
derivatives use the true sample spacing rather than a single solver step.

---

## 4. Sensor Fusion Context

These FMUs feed two Extended Kalman Filters that live in
`aerosim-controllers/python/aerosim_controllers/shift_missile_controller_fmus/`.
Each filter keys its updates off the `measurement_ready` strobes, so the
mismatched rates are handled naturally.

**Ego INS EKF (`ego_nav_ekf_fmu`)** — a 16-state / 15-error-state quaternion
inertial navigator (multiplicative/error-state EKF):

- **IMU** drives the strapdown **prediction** on every tick its strobe is set:
  \(\omega = \text{gyro} - b_g\), \(f = \text{accel} - b_a\), with
  \(\dot v = R f - g\) — the same specific-force sign convention emitted by
  `imu_fmu` (§3.1). This is why the accelerometer must output specific force,
  not kinematic acceleration.
- **GNSS** provides a linear position + velocity **update** (directly on the
  \(dp, dv\) error states); the first GNSS fix also initializes the nominal
  position/velocity.
- **Baro** provides a linear Down-position (\(p_D\)) **update**, complementing
  GNSS on the altitude channel and remaining available under GNSS dropout.

**Target tracking EKF (`target_nav_ekf_fmu`)** — a 9-state constant-
acceleration (Singer-like) kinematic tracker of the threat in NED, referenced to
the ego navigation solution:

- **Radar** provides a full nonlinear 3-D **update**
  \(z = [\text{range}, \text{az}, \text{el}, \dot r]\); the first radar return
  initializes the track by projecting range along the LOS. Its \(h(\cdot)\)
  mirrors §3.7 exactly (including \(\dot r = (r\cdot v)/\|r\|\)).
- **IR** provides a bearing-only **update** \(z = [\text{az}, \text{el}]\)
  mirroring §3.6, tightening the angular estimate between/around radar returns
  thanks to its lower angular noise and higher rate.
- Angular innovations are wrapped to \((-\pi, \pi]\), and Jacobians are formed
  numerically, robustly handling the mixed range/angle observation.

The measurement-noise parameters configured in the EKFs are intended to be
consistent with the sensor FMU noise settings (e.g. the radar/IR angle and range
sigmas, GNSS position/velocity sigmas, baro altitude sigma), so the filters are
approximately statistically matched to the sensors that produced the data.

---

## 5. References

1. P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated
   Navigation Systems*, 2nd ed., Artech House, 2013. — Strapdown INS mechanization,
   inertial error models (bias, scale factor, random walk), and GNSS/INS
   integration.
2. D. H. Titterton and J. L. Weston, *Strapdown Inertial Navigation Technology*,
   2nd ed., IET/AIAA, 2004. — Body-frame specific force and angular-rate
   measurement models and strapdown attitude propagation.
3. J. A. Farrell, *Aided Navigation: GPS with High Rate Sensors*, McGraw-Hill,
   2008. — Error-state (indirect) Kalman filtering for aided inertial navigation.
4. M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001. —
   Radar range measurement, Doppler/range-rate, and detection-range concepts.
5. M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
   McGraw-Hill, 2014. — Range, angle, and Doppler measurement and their error
   characteristics.
6. *U.S. Standard Atmosphere, 1976*, NOAA/NASA/USAF, U.S. Government Printing
   Office, 1976; and ICAO Doc 7488, *Manual of the ICAO Standard Atmosphere*,
   3rd ed., 1993. — Layered temperature/pressure model and the barometric
   pressure-altitude relation implemented in `atmosphere.py`.

*Citations identify the authoritative sources for each model; specific page
numbers are intentionally omitted rather than invented.*
