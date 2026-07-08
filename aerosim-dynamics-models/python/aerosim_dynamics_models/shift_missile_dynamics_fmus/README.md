# SHIFT Interceptor — Plant / Dynamics FMUs

Physics-based plant (dynamics) Functional Mock-up Units for the **SHIFT
interceptor** digital twin. Every module in this folder is a
[`pythonfmu3`](https://github.com/NTNU-IHB/PythonFMU3) `Fmi3Slave` that is built
into an FMU and co-simulated inside AeroSim's distributed architecture: the FMUs
exchange signals over **Apache Kafka** topics and are stepped by a **Rust
orchestrator**. Together they model the ego missile's airframe, motor, structure,
and force/moment estimation as the interceptor flies toward its target.

---

## 1. Executive Summary

### What this folder is

This folder is the **plant** half of the SHIFT interceptor twin — the set of FMUs
that own the *truth* physics of the vehicle:

| Module | Role |
| --- | --- |
| `aerodynamics_sm_fmu.py` | 6-DOF rigid-body plant (RK4) + partial aero surrogate + ISA atmosphere |
| `propulsion_sm_fmu.py` | 0-D internal-ballistics rocket motor (solid / liquid) + UQ |
| `structures_sm_fmu.py` | Mass properties + Euler–Bernoulli airframe load/stress solver + UQ |
| `corrector_fmu.py` | Full 6-DOF Ensemble Kalman Filter force/moment corrector |
| `servo_sm_fmu.py` | 3-axis fin/actuator second-order servo model |
| `sixdof.py` | Quaternion 6-DOF kinematics/dynamics helpers + RK4 integrator |
| `atmosphere.py` | ICAO Standard Atmosphere (ISA) model |

The controller FMUs (guidance, autopilot, navigation filters) live in a sibling
package; the modules here consume actuator commands and publish the resulting
vehicle state and internal loads.

### Its role in the interceptor digital twin

The aerodynamics FMU is the **perfect ground-truth plant** for the ego
interceptor (`actor1`). Each co-simulation tick it integrates the full
six-degree-of-freedom (6-DOF) Newton–Euler rigid-body equations of motion,
driven by:

- **aerodynamic forces/moments** built up from a coefficient model,
- **thrust** from the propulsion FMU (`thrust_n`),
- **mass and inertia** (`mass_kg`, `Ixx`, `Iyy`, `Izz`) from the structures FMU,
- **fin deflections** (`elevator_rad`, `aileron_rad`, `rudder_rad`) from the servo
  FMU,

and publishes a noise-free `VehicleState` on `aerosim.actor1.vehicle_state`. That
truth state feeds the sensor FMUs (which add noise), the corrector, and the
structures load solver, closing the loop.

### Design decision: full 6-DOF EOM *fed by* a surrogate

A central architectural choice is that the twin is **neither pure-analytic nor
pure-surrogate**. Instead:

- The **Newton–Euler rigid-body EOM, integrated by RK4, is the plant.** This is
  what actually advances position, velocity, attitude, and body rates, and it
  captures the attitude-dependent gravity/force rotation and the gyroscopic
  coupling of a spinning, maneuvering airframe.
- An **aerodynamic model supplies the force/moment coefficients** into those
  equations. This is the reference architecture in the missile-simulation
  literature (e.g. the JHU/APL 6-DOF GNC digital simulations and the open SRAAM
  6-DOF simulation), where the coefficients come from wind-tunnel/DATCOM tables,
  CFD, or a learned surrogate.

In this build there are two coefficient tiers:

1. A **physically complete analytic aerodynamic model** (all six force/moment
   components, static + control + damping derivatives) that drives the plant
   integration every step — this guarantees the twin always has a full 6-DOF load
   set.
2. A **Luminary `aero_sm` learned surrogate** (`mlp_model.pt`, a PyTorch MLP)
   that — in this first iteration — predicts three of the six channels
   (`force_x`, `force_z`, `moment_y`, the pitch-plane channels) and is exported
   as auxiliary telemetry with per-channel validity flags and per-channel
   1‑σ uncertainty. The unknown channels are reconstructed downstream by the
   EnKF corrector from ground-truth kinematics.

**Why this yields a high-fidelity, faster-than-real-time twin.** Integrating the
true rigid-body EOM (rather than a reduced analytic point-mass model) preserves
the coupled translational/rotational dynamics needed for realistic
endgame maneuvering and miss-distance prediction. At the same time, delegating
the expensive aerodynamic-coefficient evaluation to a compact MLP surrogate
(with an analytic fallback tier) keeps each step cheap enough to run
faster than real time. RK4 with internal sub-stepping (default `max_substep_s =
0.001 s`) provides the numerical stability the stiff, high-dynamic-pressure
airframe needs regardless of the coarser co-simulation step size. The result is
a physically faithful plant that still respects a real-time (or better) budget.

---

## 2. Component Descriptions

### 2.1 `aerodynamics_sm_fmu.py` — 6-DOF plant + partial aero surrogate

**Purpose.** Ground-truth 6-DOF rigid-body plant for the ego interceptor. Builds
the full analytic aerodynamic force/moment set, adds thrust, integrates the EOM
with RK4 sub-stepping, refreshes the ISA atmosphere at the live altitude every
step, and separately evaluates the partial ML surrogate (with UQ) for the
corrector to consume.

**Inputs (aux):**

| Name | Meaning |
| --- | --- |
| `elevator_rad`, `aileron_rad`, `rudder_rad` | Fin deflections (from servo) |
| `thrust_n` | Axial thrust along body +x (from propulsion) |
| `mass_kg`, `Ixx`, `Iyy`, `Izz` | Mass / principal inertias (from structures) |

**Outputs:**

- `vehicle_state` — the truth `VehicleState` component (position, orientation
  quaternion, velocity, angular velocity, acceleration).
- True 6-DOF body-frame loads: `true_fx_n`, `true_fy_n`, `true_fz_n`,
  `true_mx_nm`, `true_my_nm`, `true_mz_nm`.
- Surrogate 6-DOF loads: `sm_fx_n` … `sm_mz_nm` (unknown channels carry a `0.0`
  sentinel).
- Per-channel validity flags: `sm_valid_fx` … `sm_valid_mz`
  (1.0 = surrogate-predicted, 0.0 = unknown). Defaults mark `fx`, `fz`, `my`
  valid.
- Per-channel surrogate 1‑σ uncertainty: `sm_sigma_fx` … `sm_sigma_mz`.
- Flight condition + atmosphere: `alpha_deg`, `beta_deg`, `mach_number`,
  `dynamic_pressure_pa`, `air_density_kgm3`, `air_pressure_pa`,
  `speed_of_sound_mps`, `altitude_msl_m`, `model_source`.
- Mirrored scalar ego kinematics for seekers: `ego_pos_n/e/d`,
  `ego_vel_n/e/d`, `ego_qw/qx/qy/qz`, `airspeed_mps`.

**Key parameters (defaults):** `mlp_model_path = "mlp_model.pt"`,
`world_origin_altitude = 0.0`, `ref_area_m2 = 0.0314` (≈ π/4·d², d = 0.2 m),
`ref_diameter_m = 0.2`; aerodynamic derivatives `CA0 = 0.30`, `CN_alpha = 15.0`,
`CN_de = 5.0`, `CY_beta = -15.0`, `CY_dr = 5.0`, `Cm_alpha = -3.0`,
`Cm_de = -8.0`, `Cm_q = -50.0`, `Cn_beta = 3.0`, `Cn_dr = -8.0`, `Cn_r = -50.0`,
`Cl_p = -5.0`, `Cl_da = -2.0`; surrogate domain clamps `min_mach = 0.3`,
`max_mach = 4.0`, `max_abs_alpha_deg = 20.0`; UQ model `sm_rel_sigma = 0.05`,
`sm_abs_sigma_force_n = 50.0`, `sm_abs_sigma_moment_nm = 20.0`,
`sm_edge_inflation = 3.0`, `sm_unknown_sigma_force_n = 4000.0`,
`sm_unknown_sigma_moment_nm = 1500.0`; initial conditions
`init_pos_down_m = -5000.0` (≈ 5 km altitude in NED), `init_speed_mps = 600.0`;
`mass_fallback_kg = 500.0`; `max_substep_s = 0.001`.

**Notes.** If PyTorch or `mlp_model.pt` is unavailable, the surrogate path falls
back to an analytic stand-in consistent with the trainer, and `model_source`
records the reason. The surrogate is evaluated on the *final* sub-step flight
condition with features `[mach, alpha_deg, elevator_deg]`, clamped to the trained
domain.

### 2.2 `propulsion_sm_fmu.py` — 0-D internal-ballistics rocket motor

**Purpose.** Physics-based lumped (0-D) internal-ballistics motor that computes
chamber pressure, thrust, and mass flow from first principles for a **solid**
(BATES/cylindrical grain) or an optional **liquid** motor, and rolls up the
quantities of interest (QoI) that drive interceptor performance.

**Inputs (aux):** `throttle` (0–1); `ambient_pressure_pa` (from the aero
atmosphere, enabling altitude-corrected thrust).

**Outputs (aux):** `thrust_n`, `thrust_sigma_n` (UQ 1‑σ), `propellant_fraction`,
`mass_flow_kg_s`, `chamber_pressure_pa`, `isp_s`, `total_impulse_ns`,
`burn_time_s`, `thrust_coeff`, `motor_burning`.

**Key parameters (defaults):** `motor_type = "solid"`,
`propellant_mass_kg = 80.0`; solid grain `grain_type = "neutral"`
("neutral" | "progressive" | "regressive"), `grain_outer_radius_m = 0.09`,
`grain_port_radius_m = 0.035`, `propellant_density_kgm3 = 1800.0`,
`burn_rate_coeff_a = 5.0e-5`, `burn_rate_exponent_n = 0.35`,
`c_star_mps = 1550.0`; nozzle `auto_size_throat = 1.0`,
`design_chamber_pressure_pa = 9.0e6`, `throat_area_m2 = 0.0025`,
`expansion_ratio = 8.0` (Aₑ/Aₜ), `gamma_exhaust = 1.22`; liquid reference
`liquid_pc_ref_pa = 7.0e6`, `liquid_mdot_ref_kg_s = 12.0`; UQ
`sigma_a_rel = 0.05`, `sigma_cstar_rel = 0.02`.

**Notes.** At init, the grain length is derived from the loaded propellant mass so
the motor is self-consistent with the structures FMU, and (if
`auto_size_throat`) the throat is sized so the initial burn area yields the
design chamber pressure. Mass flow is capped by remaining propellant; the motor
cuts off when the load is exhausted.

### 2.3 `structures_sm_fmu.py` — mass properties + beam load/stress solver

**Purpose.** Tracks mass/inertia/CG as propellant burns and runs a
station-discretized **free–free Euler–Bernoulli beam** load analysis of the
airframe every step, from the aerodynamic/control forces fed back by the
aerodynamics FMU.

**Inputs (aux):** `propellant_fraction` (from propulsion); `true_fz_n`,
`true_fy_n` (aero normal/side force); `thrust_n` (propulsion).

**Outputs (aux):** `mass_kg`, `Ixx`, `Iyy`, `Izz`, `cg_x_m`, `load_factor_g`,
`max_bending_moment_nm`, `max_bending_stress_pa`, `axial_stress_pa`,
`combined_stress_pa`, `margin_of_safety`, `first_bending_freq_hz`,
`prob_structural_failure`.

**Key parameters (defaults):** `dry_mass_kg = 150.0`, `propellant_mass_kg = 80.0`,
`roll_inertia_per_kg = 0.01`, `pitch_inertia_per_kg = 1.02`,
`cg_full_x_m = 1.75`, `cg_empty_x_m = 1.55`; airframe `body_length_m = 3.5`,
`body_outer_radius_m = 0.10`, `wall_thickness_m = 0.004`, `cp_fraction = 0.55`;
material (aluminum 7075-T6) `youngs_modulus_pa = 71.0e9`,
`yield_strength_pa = 480.0e6`, `factor_of_safety = 1.5`, `n_stations = 40`; UQ
`sigma_yield_rel = 0.06`, `sigma_load_rel = 0.10`.

### 2.4 `corrector_fmu.py` — full 6-DOF Ensemble Kalman Filter corrector

**Purpose.** Assimilates the ego's ground-truth 6-DOF kinematics to estimate the
**complete** body-frame force/moment vector, reconstructing exactly the three
channels the surrogate is blind to (`fy`, `mx`, `mz`) and estimating the
surrogate's per-channel bias on the valid channels. Ingests the per-channel
surrogate 1‑σ to drive the forecast process noise.

**Inputs:** `vehicle_state` (ego ground truth, VehicleState topic); surrogate
predictions `sm_fx_n` … `sm_mz_nm`, validity flags `sm_valid_*`, and 1‑σ
`sm_sigma_*` (aux, from aero); `thrust_n` (propulsion); `mass_kg`, `Ixx`, `Iyy`,
`Izz` (structures).

**Outputs (aux):** corrected loads `fx_corrected_n` … `mz_corrected_nm`;
per-channel surrogate bias `bias_fx_n` … `bias_mz_nm`; posterior per-channel 1‑σ
`std_fx_n` … `std_mz_nm`; diagnostics `innovation_norm`, `ensemble_spread`,
`ground_truth_valid`.

**Key parameters (defaults):** `ensemble_size = 40`;
`proc_std_force_valid_n = 200.0`, `proc_std_moment_valid_nm = 50.0`,
`proc_std_force_unknown_n = 4000.0`, `proc_std_moment_unknown_nm = 1500.0`;
`obs_std_accel_mps2 = 0.4`, `obs_std_angaccel_rps2 = 0.2`; `rng_seed = 2024`.

**Notes.** The observation vector is derived from the truth state by finite
difference (specific force in body frame + angular acceleration). The analysis is
a stochastic perturbed-observation EnKF and is regularized/robustly solved so it
can never blow up numerically; a pathological update reinitializes the ensemble
around the surrogate.

### 2.5 `servo_sm_fmu.py` — 3-axis fin actuator model

**Purpose.** Models the finite bandwidth and authority of the elevator (pitch),
aileron (roll), and rudder (yaw) fin actuators. Each channel is an independent
first-order lag with a slew-rate limit and a hard deflection limit; throttle is
passed through so downstream FMUs can source it from one actuator topic.

**Inputs (aux):** `elevator_cmd_rad`, `aileron_cmd_rad`, `rudder_cmd_rad`,
`throttle_cmd`.

**Outputs (aux):** `elevator_rad`, `aileron_rad`, `rudder_rad`, `throttle`.

**Key parameters (defaults):** `time_constant_s = 0.03`, `max_rate_rps = 10.0`,
`max_deflection_rad = 0.436332` (25°).

### 2.6 `sixdof.py` — quaternion 6-DOF core

**Purpose.** Dependency-light (numpy + scipy `Rotation`) shared library bundled
into each plant FMU. Frames: NED world (x-North, y-East, z-Down) and body FRD
(x-forward, y-right, z-down). Quaternions are stored **scalar-last** `[x, y, z, w]`
and represent the body→NED rotation.

**Public helpers:** `quat_normalize`, `rot_from_quat`, `quat_from_euler`
(3-2-1 ZYX), `euler_from_quat`, `body_to_ned`, `ned_to_body`, `quat_to_msg`
(→ w,x,y,z), `quat_from_msg`, `alpha_beta` (α, β from body velocity),
`quat_mul` (Hamilton product), `quat_deriv` (q̇ = ½ q ⊗ [ω,0]),
`integrate_6dof_rk4` (classical RK4 with zero-order-hold body loads; returns
`pos, vel, q, omega, accel_ned`), and `integrate_6dof` (semi-implicit Euler
alternative).

### 2.7 `atmosphere.py` — ICAO Standard Atmosphere

**Purpose.** ICAO ISA model covering four standard layers from sea level to 47 km
MSL, the full range for supersonic missile profiles. Below sea level returns
sea-level values; above 47 km the stratospheric trend is linearly extrapolated.

**Public API:** `isa(altitude_m) -> (T, P, rho, a)`; convenience
`temperature_K`, `pressure_Pa`, `density_kgm3`, `speed_of_sound_mps`, `mach`,
`dynamic_pressure_Pa`, and the inverse `pressure_altitude_m`. Constants:
`GAMMA_AIR = 1.4`, `R_AIR = 287.05287 J/(kg·K)`, `G0 = 9.80665 m/s²`.

---

## 3. Mathematical Formulation

Notation: body-frame FRD vectors carry subscript \(b\); NED world vectors carry
subscript \(n\). \(R(q)\) is the body→NED rotation matrix from the (scalar-last)
attitude quaternion \(q\).

### 3.1 Rigid-body 6-DOF equations of motion (quaternion attitude)

The 13-element state is

\[
\mathbf{s} = \big[\, \mathbf{p}_n,\ \mathbf{v}_n,\ q,\ \boldsymbol{\omega}_b \,\big],
\qquad
\mathbf{p}_n = [p_N, p_E, p_D],\ \mathbf{v}_n = [v_N, v_E, v_D],\ q = [q_x,q_y,q_z,q_w].
\]

**Translational kinematics and dynamics.** With body force
\(\mathbf{F}_b = [F_x, F_y, F_z]\) (aerodynamic loads plus thrust along \(+x\)),
mass \(m\), and gravity \(\mathbf{g}_n = [0,0,g]\), \(g = 9.80665\ \mathrm{m/s^2}\):

\[
\dot{\mathbf{p}}_n = \mathbf{v}_n,
\qquad
\dot{\mathbf{v}}_n = \frac{1}{m}\,R(q)\,\mathbf{F}_b + \mathbf{g}_n .
\]

**Attitude kinematics.** For body rates \(\boldsymbol{\omega}_b = [p,q_r,r]\) the
quaternion derivative uses the Hamilton product with the pure-vector rate
quaternion:

\[
\dot{q} = \tfrac{1}{2}\, q \otimes
\begin{bmatrix} \boldsymbol{\omega}_b \\ 0 \end{bmatrix}.
\]

**Rotational dynamics (Euler's equations).** With a diagonal principal-inertia
tensor \(\mathbf{I} = \operatorname{diag}(I_{xx}, I_{yy}, I_{zz})\) and body moment
\(\mathbf{M}_b\):

\[
\dot{\boldsymbol{\omega}}_b = \mathbf{I}^{-1}\Big( \mathbf{M}_b -
\boldsymbol{\omega}_b \times (\mathbf{I}\,\boldsymbol{\omega}_b) \Big).
\]

The angle of attack and sideslip are recovered from the body velocity
\(\mathbf{v}_b = R(q)^{\top}\mathbf{v}_n = [u,v,w]\):

\[
\alpha = \operatorname{atan2}(w, u),
\qquad
\beta = \arcsin\!\left(\frac{v}{\lVert \mathbf{v}_b \rVert}\right).
\]

### 3.2 RK4 integration

Body loads are held constant over a step (co-simulation zero-order hold). With
\(\mathbf{s}\) the packed state and \(\mathbf{f}(\mathbf{s})\) the derivative
above, one classical fourth-order Runge–Kutta step of size \(h\) is

\[
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(\mathbf{s}_0), &
\mathbf{k}_2 &= \mathbf{f}\!\left(\mathbf{s}_0 + \tfrac{h}{2}\mathbf{k}_1\right), \\
\mathbf{k}_3 &= \mathbf{f}\!\left(\mathbf{s}_0 + \tfrac{h}{2}\mathbf{k}_2\right), &
\mathbf{k}_4 &= \mathbf{f}\!\left(\mathbf{s}_0 + h\,\mathbf{k}_3\right),
\end{aligned}
\qquad
\mathbf{s}_1 = \mathbf{s}_0 + \frac{h}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big).
\]

The quaternion is renormalized after the update. The plant sub-steps each
co-simulation tick into \(n_{\text{sub}} = \lceil \Delta t / h_{\max} \rceil\)
segments (default \(h_{\max} = 0.001\ \mathrm{s}\)) for stability of the stiff,
high-\(\bar q\) airframe.

### 3.3 ICAO Standard Atmosphere

For a layer with base altitude \(h_b\), base temperature \(T_b\), base pressure
\(P_b\), and lapse rate \(L\), at geometric altitude \(h\):

**Temperature.**
\[
T(h) = T_b + L\,(h - h_b).
\]

**Pressure.** For a gradient layer (\(L \neq 0\)) and an isothermal layer
(\(L = 0\)) respectively,

\[
P(h) = P_b \left(\frac{T}{T_b}\right)^{\!-\frac{g_0}{R\,L}},
\qquad
P(h) = P_b \, \exp\!\left(-\frac{g_0\,(h - h_b)}{R\,T_b}\right).
\]

(The code writes the gradient exponent as \(g_0 / (R(-L))\), which is identical.)

**Density and speed of sound.**
\[
\rho = \frac{P}{R\,T},
\qquad
a = \sqrt{\gamma\,R\,T},
\]

with \(\gamma = 1.4\), \(R = 287.05287\ \mathrm{J\,kg^{-1}K^{-1}}\),
\(g_0 = 9.80665\ \mathrm{m/s^2}\). Layer bases: \((0\,\mathrm{m}, 288.15\,\mathrm K,
-0.0065)\), \((11\,\mathrm{km}, 216.65\,\mathrm K, 0)\), \((20\,\mathrm{km},
216.65\,\mathrm K, +0.0010)\), \((32\,\mathrm{km}, 228.65\,\mathrm K, +0.0028)\),
capped at \(47\,\mathrm{km}\). Base pressures are precomputed by chaining the
barometric formula across boundaries.

### 3.4 Aerodynamic coefficient buildup, surrogate blending, and UQ

**Analytic buildup (all six channels).** With dynamic pressure
\(\bar q = \tfrac{1}{2}\rho V^2\), reference area \(S\), diameter \(d\),
\(qS = \bar q S\), \(qSd = \bar q S d\), and non-dimensional body rates
\(\hat p = p\,d/2V\), \(\hat q = q_r\,d/2V\), \(\hat r = r\,d/2V\), the coefficients
are

\[
\begin{aligned}
C_A &= C_{A0}, &
C_N &= C_{N\alpha}\,\alpha + C_{N\delta_e}\,\delta_e, &
C_Y &= C_{Y\beta}\,\beta + C_{Y\delta_r}\,\delta_r, \\
C_l &= C_{l p}\,\hat p + C_{l\delta_a}\,\delta_a, &
C_m &= C_{m\alpha}\,\alpha + C_{m\delta_e}\,\delta_e + C_{mq}\,\hat q, &
C_n &= C_{n\beta}\,\beta + C_{n\delta_r}\,\delta_r + C_{nr}\,\hat r,
\end{aligned}
\]

giving body-frame forces and moments

\[
F_x = -C_A\,qS,\quad
F_y = C_Y\,qS,\quad
F_z = -C_N\,qS,\quad
M_x = C_l\,qSd,\quad
M_y = C_m\,qSd,\quad
M_z = C_n\,qSd,
\]

where \(F_x\) is axial drag (rearward), and a positive \(\alpha\) produces an
upward (\(-z\)) normal force. Thrust is added separately along \(+x\) before
integration.

**Surrogate.** The MLP \(g_\theta\) maps the clamped features
\(\mathbf{x} = [\,M_c,\ \alpha_c,\ \delta_{e,\deg}\,]\) to the three pitch-plane
channels,

\[
[\,\hat F_x,\ \hat F_z,\ \hat M_y\,] = g_\theta(\mathbf{x}),
\qquad
M_c = \operatorname{clip}(M, M_{\min}, M_{\max}),\quad
\alpha_c = \operatorname{clip}(\alpha, -\alpha_{\max}, \alpha_{\max}),
\]

with the analytic stand-in used when the model is unavailable. The unknown
channels \((F_y, M_x, M_z)\) are published as a finite \(0.0\) sentinel with
validity flag \(0\) (JSON aux transport cannot carry NaN/Inf); the predicted
channels get validity flag \(1\).

**Per-channel UQ (\(\sigma\)) model.** An edge-extrapolation inflation grows from
unity at the domain center to \(k_{\text{edge}}\) at the clamp limits,

\[
e = \max\!\left(
\frac{\lvert M_c - \tfrac{1}{2}(M_{\min}+M_{\max}) \rvert}{\tfrac{1}{2}(M_{\max}-M_{\min})},\
\frac{\lvert \alpha_c \rvert}{\alpha_{\max}}
\right),
\qquad
\lambda = 1 + (k_{\text{edge}} - 1)\,\min(\max(e,0),1),
\]

and the predicted-channel sigmas combine a relative and an absolute term:

\[
\sigma_{F} = \lambda\sqrt{(\sigma_{\mathrm{rel}}\,\lvert F \rvert)^2 + \sigma_{\mathrm{abs},F}^2},
\qquad
\sigma_{M} = \lambda\sqrt{(\sigma_{\mathrm{rel}}\,\lvert M \rvert)^2 + \sigma_{\mathrm{abs},M}^2}.
\]

Unknown channels take a large diffuse sigma (`sm_unknown_sigma_force_n`,
`sm_unknown_sigma_moment_nm`) so downstream consumers treat them as effectively
unobserved.

### 3.5 Solid-motor internal ballistics

**Grain geometry (BATES/cylindrical).** The grain length follows from the loaded
mass, \(L = m_p / [\rho_p\,\pi(r_o^2 - r_p^2)]\), and the reference internal burn
area is \(A_{b0} = 2\pi r_p L\). The instantaneous burn area depends on
`grain_type`: neutral holds \(A_b = A_{b0}\); progressive uses
\(A_b = 2\pi (r_p + w) L\) growing with web \(w\); regressive uses
\(A_b = 2\pi (r_o - w) L\).

**Chamber-pressure equilibrium.** Equating propellant mass generation to nozzle
mass flow with St. Robert's burn-rate law \(r = a\,p_c^{\,n}\) and klemmung
\(K_n = A_b/A_t\) gives the equilibrium chamber pressure

\[
p_c = \big(\rho_p\, a\, c^{*}\, K_n\big)^{\frac{1}{1-n}},
\qquad
\dot r = a\, p_c^{\,n},
\qquad
\dot m = \rho_p\, A_b\, \dot r,
\qquad
w \mathrel{+}= \dot r\,\Delta t .
\]

**Auto-sized throat.** When enabled, the throat is sized at init so the initial
burn area yields the design pressure:

\[
A_t = \frac{A_{b0}\,\rho_p\, a\, c^{*}}{p_{c,\text{design}}^{\,1-n}} .
\]

**Nozzle and thrust.** The supersonic exit Mach \(M_e\) solves the isentropic
area–Mach relation for expansion ratio \(\varepsilon = A_e/A_t\),

\[
\varepsilon = \frac{1}{M_e}\left[\frac{2}{\gamma+1}\left(1 + \tfrac{\gamma-1}{2}M_e^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}},
\]

(solved by Newton iteration), with exit-to-chamber pressure ratio
\(p_e/p_c = \left(1 + \tfrac{\gamma-1}{2}M_e^2\right)^{-\gamma/(\gamma-1)}\). The
vacuum thrust coefficient and its ambient correction are

\[
C_{f,\text{vac}} = \sqrt{\frac{2\gamma^2}{\gamma-1}\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}}\!\left(1 - \left(\tfrac{p_e}{p_c}\right)^{\frac{\gamma-1}{\gamma}}\right)} + \frac{p_e}{p_c}\,\varepsilon,
\qquad
C_f = C_{f,\text{vac}} - \frac{p_a}{p_c}\,\varepsilon,
\]

so thrust and specific impulse are

\[
F = C_f\, p_c\, A_t,
\qquad
I_{sp} = \frac{F}{\dot m\, g_0},
\qquad
I_{\text{tot}} = \int F\,dt \approx \sum F\,\Delta t .
\]

The liquid abstraction uses \(p_c = \tau\, p_{c,\text{ref}}\),
\(\dot m = \tau\, \dot m_{\text{ref}}\) with the same \(C_f\) relation.

### 3.6 Euler–Bernoulli beam theory (airframe loads)

The airframe is a slender free–free beam of length \(L\), discretized into \(n\)
uniform-mass stations. The aerodynamic normal-force resultant
\(N = \sqrt{F_z^2 + F_y^2}\) is lumped at the center-of-pressure station; the body
accelerates at \(a = N/m\) so d'Alembert inertial relief distributes
\(-(m_i/m)N\) along the body, giving a self-equilibrated transverse load

\[
q_i = N_{\text{aero},i} - \frac{m_i}{m}\,N,
\qquad \sum_i q_i = 0 .
\]

Shear and bending moment follow by cumulative integration (\(V = \mathrm dM/\mathrm dx\),
\(M = -EI\,w''\)):

\[
V_i = \sum_{k \le i} q_k,
\qquad
M_i = \Delta x \sum_{k \le i} V_k,
\qquad
M_{\max} = \max_i \lvert M_i \rvert .
\]

For the thin-wall tube with second moment
\(I = \tfrac{\pi}{4}(r_o^4 - r_i^4)\) and cross-section area
\(A = \pi(r_o^2 - r_i^2)\), the peak bending stress, axial stress, and combined
(compression-fiber) stress are

\[
\sigma_b = \frac{M_{\max}\, r_o}{I},
\qquad
\sigma_a = \frac{\lvert T \rvert}{A},
\qquad
\sigma_{\max} = \sigma_b + \sigma_a .
\]

The load factor is \(n_g = N/(mg_0)\). With allowable \(\sigma_{\text{allow}}\)
(material yield) and factor of safety \(\mathrm{FoS} = 1.5\), the margin of safety
is

\[
\mathrm{MoS} = \frac{\sigma_{\text{allow}}}{\mathrm{FoS}\cdot\sigma_{\max}} - 1 .
\]

The free–free first bending natural frequency uses \(\beta_1 L = 4.730\) and mass
per unit length \(\mu = m/L\):

\[
f_1 = \frac{(\beta_1 L)^2}{2\pi L^2}\sqrt{\frac{EI}{\mu}} .
\]

**Probabilistic failure.** Treating applied stress and allowable as Gaussian with
relative scatters \(\sigma_{\text{load}}\) and \(\sigma_{\text{yield}}\), the
failure probability is \(P(\text{margin}<0)\):

\[
\text{applied} = \mathrm{FoS}\cdot\sigma_{\max},
\quad
m_g = \sigma_{\text{allow}} - \text{applied},
\quad
\sigma = \sqrt{(\sigma_{\text{load}}\,\text{applied})^2 + (\sigma_{\text{yield}}\,\sigma_{\text{allow}})^2},
\]
\[
P_f = \Phi\!\left(-\frac{m_g}{\sigma}\right) = \tfrac{1}{2}\,\operatorname{erfc}\!\left(\frac{m_g}{\sigma\sqrt{2}}\right).
\]

### 3.7 Ensemble Kalman Filter (corrector)

**State and ensemble.** The state is the full body-frame load vector
\(\mathbf{x} = [F_x, F_y, F_z, M_x, M_y, M_z]\), represented by an ensemble
\(\{\mathbf{x}^{(i)}\}_{i=1}^{N}\) (default \(N = 40\)).

**Forecast (recenter + inflation).** Surrogate-valid channels re-center on the
surrogate value; unknown channels persist the current ensemble mean. The ensemble
is then perturbed by the per-channel process covariance
\(\mathbf{Q} = \operatorname{diag}(\sigma_{q}^2)\):

\[
\mathbf{x}^{(i)} \leftarrow \bar{\mathbf{c}} + \boldsymbol{\eta}^{(i)},
\qquad
\boldsymbol{\eta}^{(i)} \sim \mathcal N(\mathbf 0, \mathbf Q),
\qquad
c_j = \begin{cases} \text{surrogate}_j & \text{valid} \\ \bar x_j & \text{unknown} \end{cases}.
\]

The process std is data-driven: on valid channels with a positive surrogate 1‑σ,
\(\sigma_q = \max(\sigma_{\mathrm{sm}}, 10^{-6})\); otherwise fixed valid/unknown
defaults are used.

**Observation operator.** Predicted observations map each ensemble member to a
specific force and angular acceleration,

\[
\mathbf{a}_b = \frac{[F_x + T,\ F_y,\ F_z]}{m},
\qquad
\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1}\big([M_x, M_y, M_z] - \boldsymbol{\omega}\times(\mathbf{I}\boldsymbol{\omega})\big),
\qquad
h(\mathbf{x}) = [\mathbf{a}_b,\ \dot{\boldsymbol{\omega}}].
\]

The actual observation \(\mathbf y\) is derived from the ground-truth state:
\(\mathbf a_n = (\mathbf v_n - \mathbf v_n^{-})/\Delta t\), specific force
\(\mathbf a_n - [0,0,g]\) rotated into the body frame, and
\(\dot{\boldsymbol\omega} = (\boldsymbol\omega - \boldsymbol\omega^{-})/\Delta t\).

**Analysis (stochastic EnKF).** With ensemble anomalies \(\mathbf X'\) (state)
and \(\mathbf Y'\) (predicted obs), cross- and innovation covariances

\[
\mathbf P_{xy} = \frac{\mathbf X' \mathbf Y'^{\top}}{N-1},
\qquad
\mathbf P_{yy} = \frac{\mathbf Y' \mathbf Y'^{\top}}{N-1} + \mathbf R,
\]

the Kalman gain and perturbed-observation update are

\[
\mathbf K = \mathbf P_{xy}\,\mathbf P_{yy}^{-1},
\qquad
\mathbf x^{(i)} \leftarrow \mathbf x^{(i)} + \mathbf K\big(\mathbf y + \boldsymbol\epsilon^{(i)} - h(\mathbf x^{(i)})\big),
\quad
\boldsymbol\epsilon^{(i)}\sim\mathcal N(\mathbf 0, \mathbf R),
\]

with \(\mathbf R = \operatorname{diag}(\sigma_a^2,\sigma_a^2,\sigma_a^2,
\sigma_{\dot\omega}^2,\sigma_{\dot\omega}^2,\sigma_{\dot\omega}^2)\). \(\mathbf
P_{yy}\) is Tikhonov-regularized before the solve. The posterior estimate is the
ensemble mean; the posterior per-channel std \(\operatorname{std}_j\) and the
mean spread are reported as UQ, and the surrogate bias is
\(\text{analysis}_j - \text{surrogate}_j\) on valid channels only.

---

## 4. Uncertainty Quantification (UQ) Flow

The plant is instrumented end-to-end so that aleatory/epistemic uncertainty
propagates from the aerodynamics surrogate through the rest of the stack and into
the downstream probability-of-kill (\(P_{\text{kill}}\)) analysis.

1. **Source — aero surrogate \(\sigma\).** `aerodynamics_sm_fmu` emits a
   per-channel 1‑σ (`sm_sigma_*`) that combines a relative term
   (\(\sigma_{\mathrm{rel}}|value|\)) and an absolute floor, inflated up to
   \(k_{\text{edge}} = 3\times\) toward the trained-domain edges in Mach and
   \(\alpha\). Unknown channels carry a large diffuse sigma.

2. **Into the corrector.** `corrector_fmu` reads `sm_sigma_*` and uses it directly
   as the forecast **process noise** on valid channels (data-driven UQ replacing
   the fixed defaults). Larger surrogate sigma → looser forecast prior → the
   ground-truth observations pull the estimate harder. The corrector publishes the
   **posterior per-channel std** (`std_*`), `ensemble_spread`, and the surrogate
   `bias_*` — a calibrated uncertainty on the reconstructed 6-DOF loads, including
   the channels the surrogate never modeled.

3. **Into propulsion.** `propulsion_sm_fmu` reports a first-order (delta-method)
   thrust 1‑σ, `thrust_sigma_n`. For a solid, \(p_c \sim (a\,c^{*})^{1/(1-n)}\) and
   \(F \sim p_c\), so
   \(\sigma_F/F = \tfrac{1}{1-n}\sqrt{(\sigma_a/a)^2 + (\sigma_{c^*}/c^*)^2}\); for a
   liquid, \(\sigma_F/F = \sigma_{c^*}/c^*\).

4. **Into structures.** `structures_sm_fmu` converts load/material scatter
   (\(\sigma_{\text{load}}\), \(\sigma_{\text{yield}}\)) into an explicit
   **structural failure probability** `prob_structural_failure` alongside the
   deterministic margin of safety.

5. **Into \(P_{\text{kill}}\).** The corrected-load posterior std, the thrust
   1‑σ, and the structural failure probability are all consumable by the
   downstream Monte-Carlo \(P_{\text{kill}}\) estimator, so miss-distance/lethality
   confidence reflects the actual modeling uncertainty rather than a point
   estimate.

---

## 5. References

- B. L. Stevens, F. L. Lewis, and E. N. Johnson, *Aircraft Control and
  Simulation: Dynamics, Controls Design, and Autonomous Systems*, 3rd ed., Wiley,
  2015 — quaternion 6-DOF rigid-body equations of motion, body/NED frames, and
  numerical integration of the flight-vehicle EOM.
- P. H. Zipfel, *Modeling and Simulation of Aerospace Vehicle Dynamics*, 3rd ed.,
  AIAA Education Series, 2014 — Newton–Euler formulation, quaternion attitude
  kinematics, and coefficient-driven aerodynamic force/moment buildup for missile
  6-DOF simulation.
- J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd
  ed., Wiley, 2016 — classical fourth-order Runge–Kutta method.
- *U.S. Standard Atmosphere, 1976*, NOAA/NASA/USAF, U.S. Government Printing
  Office, 1976; and ICAO Doc 7488, *Manual of the ICAO Standard Atmosphere*, 3rd
  ed., 1993 — layer definitions, lapse rates, and the barometric equations
  implemented in `atmosphere.py`.
- G. P. Sutton and O. Biblarz, *Rocket Propulsion Elements*, 9th ed., Wiley, 2017
  — St. Robert's (Vieille's) burn-rate law, solid-motor chamber-pressure
  equilibrium and klemmung, thrust coefficient \(C_f\), characteristic velocity
  \(c^*\), specific impulse, and the isentropic nozzle area–Mach relation.
- H. D. Humble, G. N. Henry, and W. J. Larson (eds.), *Space Propulsion Analysis
  and Design*, McGraw-Hill, 1995 — internal-ballistics sizing, nozzle expansion,
  and motor performance parameters.
- A. Davenas, *Solid Rocket Propulsion Technology*, Pergamon, 1993; and N.
  Kubota, *Propellants and Explosives: Thermochemical Aspects of Combustion*,
  3rd ed., Wiley-VCH, 2015 — solid-propellant grain design and burn-rate
  characterization.
- J. M. Gere and B. J. Goodno, *Mechanics of Materials*, 8th ed., Cengage, 2012 —
  Euler–Bernoulli beam theory, the flexure formula \(\sigma = Mc/I\), shear/moment
  relations, and combined stress.
- R. D. Blevins, *Formulas for Natural Frequency and Mode Shape*, Krieger, 1979 —
  free–free beam first-bending eigenvalue \(\beta_1 L = 4.730\) and natural
  frequency formula.
- NASA SP-8007, *Buckling of Thin-Walled Circular Cylinders* (rev. 1968), and
  E. F. Bruhn, *Analysis and Design of Flight Vehicle Structures*, Jacobs, 1973 —
  thin-wall airframe strength, factor of safety, and margin-of-safety practice.
- G. Evensen, *Data Assimilation: The Ensemble Kalman Filter*, 2nd ed., Springer,
  2009 (and G. Evensen, "The Ensemble Kalman Filter: theoretical formulation and
  practical implementation," *Ocean Dynamics*, 53(4), 2003) — stochastic
  perturbed-observation EnKF forecast/analysis equations.
