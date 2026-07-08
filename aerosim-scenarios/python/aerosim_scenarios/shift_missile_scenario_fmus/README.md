# SHIFT Missile Interception — Scenario FMUs

Scenario actors and shared numerical helpers for the **SHIFT missile‑interception
digital twin**, packaged as [PythonFMU3](https://github.com/NTNU-IHB/PythonFMU3)
co‑simulation units for the AeroSim distributed runtime (FMI 3.0).

---

## 1. Executive Summary

This folder implements the **threat (target) side** of a hit‑to‑kill air‑defense
engagement: an ego **SHIFT interceptor** (a ~200 mm‑class agile missile modeled
elsewhere in AeroSim) is launched against a **configurable, weaving threat
missile** that ingresses toward a defended asset. The threat is a perfect
ground‑truth plant — everything the ego's seekers and the force/moment corrector
"see" about the target originates here.

The scenario is deliberately built from **small, modular configuration files**
rather than one monolithic sim‑config. The composition hierarchy is:

```
master_intercept.json          which scenario + which Monte-Carlo campaign
  └─ scenario_headon_intercept.json   world / clock / engagement geometry
       ├─ missile_shift_interceptor.json   ego interceptor design + GNC params
       └─ target_cruise_missile.json       threat missile parameters
  └─ monte_carlo.json           dispersed-parameter P_kill campaign
```

Each JSON owns one concern, so many engagements (different threats, geometries,
guidance laws, or uncertainty campaigns) can be expressed by swapping or editing
one small file instead of duplicating a large config. The **FMU topology** (the
wiring graph between FMUs) is fixed in Python (`compose_sim_config.py`); the JSON
files carry only **parameters**, world/clock settings, and engagement geometry.

There are **two ways to run** the same engagement, driven by the *same* modular
config so results are consistent:

1. **Distributed co‑simulation** — compose the modular files into a full AeroSim
   sim‑config with `compose_sim_config.py`, then run it on the AeroSim
   orchestrator (Kafka‑backed, real‑time‑paced, renderer‑capable).
2. **Standalone, faster‑than‑real‑time** — run the entire FMU stack in a single
   Python process via `engagement.py` (one shot) or `run_monte_carlo.py` (a
   full probability‑of‑kill campaign), with no Kafka/orchestrator dependency.

---

## 2. Component Descriptions

### 2.1 `threat_missile_fmu.py` — the threat/target actor (`actor2`)

A guided **point‑mass** with a **6‑DOF‑consistent attitude**: the translational
motion is integrated as a point mass, and a quaternion is synthesized from the
flight‑path angles plus the bank required for the commanded turn, so downstream
consumers receive a complete `VehicleState` (position, orientation, velocity,
body rates, acceleration). It is an **autonomous scenario actor** and takes **no
component inputs**.

**Guidance behavior.** The threat combines three saturated, superimposed
commands:

- **Speed hold** — axial acceleration along the velocity vector that drives speed
  toward `cruise_speed_mps`.
- **Turn‑to‑waypoint** — a horizontal‑plane lateral command that steers the
  heading toward the ingress waypoint `(target_north_m, target_east_m)`.
- **Altitude hold** — a vertical (proportional‑derivative) command that holds
  `cruise_altitude_m`.
- **Evasive weave** — a sinusoidal lateral acceleration superimposed on the turn
  command, giving the classic weaving‑target maneuver.

All lateral/vertical commands are clipped by the control‑surface authority
`max_lateral_accel_g × g`, so a less‑agile threat (small `max_lateral_accel_g`)
visibly under‑turns and under‑weaves.

**Parameters** (FMI parameters, with in‑code defaults):

| Parameter | Default | Meaning |
|---|---|---|
| `launch_range_m` | `20000.0` | Down‑range distance of the threat from the world origin at start |
| `launch_bearing_deg` | `0.0` | Bearing of the threat's start point from the origin |
| `cruise_altitude_m` | `5000.0` | Held cruise altitude (MSL, via origin altitude) |
| `cruise_speed_mps` | `300.0` | Commanded cruise speed |
| `max_axial_accel_mps2` | `30.0` | Axial (speed‑hold) acceleration limit |
| `max_lateral_accel_g` | `8.0` | Lateral/vertical acceleration authority (in g) |
| `target_north_m` | `0.0` | North coordinate of the defended asset (ingress waypoint) |
| `target_east_m` | `0.0` | East coordinate of the defended asset |
| `weave_amplitude_g` | `0.0` | Weave lateral‑acceleration amplitude (in g); `0` disables |
| `weave_frequency_hz` | `0.2` | Weave frequency |
| `onboard_nav_error_std_m` | `10.0` | Documented threat nav/seeker grade (not used by the point‑mass plant) |
| `world_origin_altitude` | `0.0` | Origin altitude for MSL/NED conversion |

> Note: the scenario/target configs set `cruise_altitude_m = 8000.0` and
> `weave_amplitude_g = 2.0`, overriding the in‑code defaults above.

**Outputs.** A `VehicleState` component output (published on
`aerosim.actor2.vehicle_state`) plus auxiliary scalar truth channels consumed by
the ego's seekers and the corrector:

- Position: `threat_pos_n`, `threat_pos_e`, `threat_pos_d` (NED, meters)
- Velocity: `threat_vel_n`, `threat_vel_e`, `threat_vel_d` (NED, m/s)
- `threat_speed_mps` — speed magnitude
- `threat_altitude_m` — altitude MSL (`world_origin_altitude − pD`)

### 2.2 `sixdof.py` — shared 6‑DOF quaternion helpers

A dependency‑light (numpy + `scipy.spatial.transform.Rotation`) copy of the
shared rigid‑body core, bundled as a project resource so the FMU is
self‑contained. Frames are **NED** world (z down) and **FRD** body; quaternions
are stored **scalar‑last** `[x, y, z, w]` (scipy convention) and represent the
body→NED rotation. It provides:

- Quaternion utilities: `quat_normalize`, `rot_from_quat`, `quat_from_euler`
  (3‑2‑1 yaw‑pitch‑roll), `euler_from_quat`, `quat_to_msg` / `quat_from_msg`
  (convert to/from the AeroSim `w,x,y,z` `Orientation` message).
- Frame transforms `body_to_ned` / `ned_to_body`, aerodynamic angles
  `alpha_beta`, and quaternion kinematics `quat_mul`, `quat_deriv`
  (\( \dot q = \tfrac12\, q \otimes [\omega,\,0] \)).
- Full rigid‑body integrators `integrate_6dof_rk4` (classical RK4) and
  `integrate_6dof` (semi‑implicit Euler) over the 13‑element state
  `[pN pE pD, vN vE vD, qx qy qz qw, wx wy wz]`.

The threat FMU uses only the `quat_from_euler` / `quat_to_msg` helpers (it
synthesizes attitude from kinematics rather than integrating the full 6‑DOF
equations); the remaining functions are shared with the ego dynamics/GNC FMUs.

### 2.3 `atmosphere.py` — ICAO Standard Atmosphere

A four‑layer ICAO/ISA model (sea level → 47 km MSL) covering the full supersonic
flight envelope, with monotonic barometric pressure precomputation. It returns
temperature, pressure, density, and speed of sound, plus derived helpers
`mach`, `dynamic_pressure_Pa`, and the altimeter inverse `pressure_altitude_m`.
This is the same shared copy used by the ego aero/sensor FMUs; the threat
point‑mass plant does not currently call it, but it is bundled so any
higher‑fidelity threat iteration has ISA available.

### 2.4 Modular configuration hierarchy (what each JSON owns)

| File | Owns |
|---|---|
| `master_intercept.json` | Top‑level run: which `scenario`, which `monte_carlo` campaign, and the output sim‑config filename. |
| `scenario_headon_intercept.json` | World origin/weather, clock (`step_size_ms`, pacing), engagement geometry (ego initial conditions + target launch geometry), free‑form `overrides` (e.g. `guidance.guidance_law`, `autopilot.controller_type`), and which `missile` + `target` files to use. |
| `missile_shift_interceptor.json` | The ego interceptor design: per‑FMU `fmus` parameter blocks for aerodynamics surrogate, servo, solid motor, structures, EnKF corrector, sensors, seekers, EKFs, guidance, and autopilot. |
| `target_cruise_missile.json` | The threat design: the `threat_missile` parameter block (a subset of §2.1). |
| `monte_carlo.json` | The dispersed‑parameter P_kill campaign: run count, seed, sim time, step, lethal radius, damage model, confidence, and the `uncertain_params` distributions. |

`compose_sim_config.py` merges these into the concrete `fmu_models` graph. Two
naming conventions are resolved during the merge: engagement `target.*` and
`uncertain_params` keys prefixed with `target.` are routed to the
`threat_missile` FMU, and `world.origin.altitude` is propagated as
`world_origin_altitude` to every FMU that needs the MSL/NED datum.

---

## 3. Mathematical Formulation

Frames are NED (North‑East‑Down). Let \( \mathbf{p} = [p_N, p_E, p_D]^\top \)
and \( \mathbf{v} = [v_N, v_E, v_D]^\top \) be the threat position and velocity,
\( V = \lVert \mathbf{v} \rVert \) its speed, \( \hat{\mathbf{v}} =
\mathbf{v}/V \) the unit velocity, and \( g = 9.80665\ \mathrm{m/s^2} \).

### 3.1 Engagement geometry and initial conditions

The threat starts down‑range at the configured range and bearing, at the cruise
altitude, heading toward the defended waypoint. With bearing
\( \psi_0 = \mathrm{launch\_bearing\_deg} \) and origin altitude
\( h_0 = \mathrm{world\_origin\_altitude} \):

\[
\mathbf{p}(0) =
\begin{bmatrix}
R_\text{launch}\cos\psi_0 \\
R_\text{launch}\sin\psi_0 \\
h_0 - h_\text{cruise}
\end{bmatrix},
\qquad
\chi_0 = \operatorname{atan2}\!\big(w_E - p_E,\; w_N - p_N\big),
\]

where \( R_\text{launch} = \mathrm{launch\_range\_m} \),
\( h_\text{cruise} = \mathrm{cruise\_altitude\_m} \), and
\( \mathbf{w} = (\mathrm{target\_north\_m}, \mathrm{target\_east\_m}) \) is the
ingress waypoint. The initial velocity is horizontal at cruise speed along the
initial heading \( \chi_0 \):

\[
\mathbf{v}(0) = V_\text{cruise}\,[\cos\chi_0,\; \sin\chi_0,\; 0]^\top .
\]

In the baseline head‑on scenario, the ego interceptor starts at the origin at
8 km altitude (`init_pos_down_m = -8000`) at 600 m/s, and the threat starts
20 km down‑range (`launch_range_m = 20000`) on bearing 0, cruising at 8 km and
300 m/s toward the origin — a nearly head‑on closing geometry.

### 3.2 Threat motion model and weaving maneuver

**Speed‑hold (axial) acceleration** along the velocity direction, saturated by
the axial limit \( a_x^{\max} = \mathrm{max\_axial\_accel\_mps2} \):

\[
\mathbf{a}_\text{axial} =
\operatorname{clip}\!\big(V_\text{cruise} - V,\; -a_x^{\max},\; a_x^{\max}\big)\,
\hat{\mathbf{v}} .
\]

**Turn‑to‑waypoint** in the horizontal plane. With desired heading
\( \chi_d = \operatorname{atan2}(w_E - p_E,\; w_N - p_N) \), current heading
\( \chi = \operatorname{atan2}(v_E, v_N) \), and wrapped heading error
\( \Delta\chi = \operatorname{atan2}(\sin(\chi_d-\chi), \cos(\chi_d-\chi)) \),
the (proportional) turn command is saturated by the lateral authority
\( a_\perp^{\max} = \mathrm{max\_lateral\_accel\_g}\cdot g \):

\[
a_\text{turn} =
\operatorname{clip}\!\big(4\,\Delta\chi\,V,\; -a_\perp^{\max},\; a_\perp^{\max}\big).
\]

**Evasive weave** — a sinusoidal lateral acceleration superimposed on the turn:

\[
a_\text{weave}(t) = A\,g\,\sin\!\big(2\pi f\, t\big),
\]

where \( A = \mathrm{weave\_amplitude\_g} \) and
\( f = \mathrm{weave\_frequency\_hz} \). The total horizontal lateral
acceleration acts along the left‑perpendicular unit vector
\( \hat{\mathbf{l}} = [-\hat v_E,\; \hat v_N,\; 0]^\top \):

\[
\mathbf{a}_\text{lat} = \hat{\mathbf{l}}\,\big(a_\text{turn} + a_\text{weave}\big).
\]

**Altitude‑hold** (vertical, NED‑down positive) is a PD law on the altitude
error \( e_h = h_\text{cruise} - (h_0 - p_D) \), also clipped by
\( a_\perp^{\max} \):

\[
a_\text{vert} = \operatorname{clip}\!\big(2\,e_h - 1.5\,(-v_D),\; -a_\perp^{\max},\; a_\perp^{\max}\big),
\qquad
\mathbf{a}_\text{vert} = [0,\; 0,\; -a_\text{vert}]^\top .
\]

The total specific force and the explicit (Euler) integration over a step
\( \Delta t \) are:

\[
\mathbf{a} = \mathbf{a}_\text{axial} + \mathbf{a}_\text{lat} + \mathbf{a}_\text{vert},
\qquad
\mathbf{v} \mathrel{+}= \mathbf{a}\,\Delta t,
\qquad
\mathbf{p} \mathrel{+}= \mathbf{v}\,\Delta t .
\]

**Attitude synthesis.** The quaternion is built from flight‑path angles plus a
coordinated‑turn bank from the commanded horizontal lateral acceleration:

\[
\psi = \operatorname{atan2}(v_E, v_N),\quad
\theta = \operatorname{atan2}\!\big(-v_D,\ \sqrt{v_N^2 + v_E^2}\big),\quad
\phi = \operatorname{atan2}\!\big(a_\text{turn}+a_\text{weave},\ g\big),
\]

with body rates recovered from the finite rotation increment
\( \boldsymbol{\omega} = \operatorname{rotvec}\!\big(R_k R_{k-1}^{-1}\big)/\Delta t \).

### 3.3 Closing geometry and miss distance

The relative (line‑of‑sight) vector from ego to threat and the range are:

\[
\mathbf{r}(t) = \mathbf{p}_\text{threat}(t) - \mathbf{p}_\text{ego}(t),
\qquad
r(t) = \lVert \mathbf{r}(t) \rVert .
\]

The **miss distance** is the closest approach over the engagement,

\[
m = \min_{t}\, r(t),
\]

detected in the harness as the range minimum just before \( r(t) \) begins to
increase again (closest‑approach pass). The closing speed is
\( V_c = -\dot r = -\,\mathbf{r}\cdot\dot{\mathbf{r}}/r \).

### 3.4 Probability‑of‑kill (P_kill) Monte‑Carlo pipeline

The scenarios in this folder drive the single‑shot probability of kill (SSPK /
P_kill). The pipeline (implemented in `uncertainty.py`, orchestrated by
`run_monte_carlo.py`) is:

1. **Sample** uncertain parameters per run from their configured distributions
   (normal / uniform / lognormal).
2. **Simulate** one closed‑loop engagement per sample and record its miss
   distance, producing an empirical **miss distribution**.
3. **Score** each miss with a lethality (damage) function against the lethal
   radius \( r_L \).

The **Carleton** diffuse‑Gaussian damage function maps a miss \( r \) to a
conditional kill probability

\[
P_k(r) = \exp\!\left(-\frac{r^2}{2 b^2}\right),
\qquad
b = \frac{r_L}{\sqrt{2\ln 2}},
\]

with \( b \) chosen so that \( P_k(r_L) = \tfrac12 \). The Monte‑Carlo Carleton
P_kill is the mean of the per‑run \( P_k \) with a central‑limit‑theorem
interval. As a cross‑check, for a circular‑Gaussian miss with per‑axis standard
deviation \( \sigma \) the closed‑form SSPK integrates to

\[
P_\text{kill} = \frac{b^2}{b^2 + \sigma^2}.
\]

The **cookie‑cutter** model scores each run as a Bernoulli trial
(\( P_k = 1 \) iff \( r \le r_L \), else \( 0 \)), and the P_kill estimate is
reported with a **Wilson score** confidence interval

\[
\hat p_\pm = \frac{\hat p + \dfrac{z^2}{2n} \pm z\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}},
\]

for \( \hat p = k/n \) hits in \( n \) runs at confidence level implied by
\( z \). Diverged / non‑finite runs are scored as clean misses (kill
probability 0) rather than dropped, so numerical instability appears as reduced
P_kill instead of survivor bias.

---

## 4. How to Run

All commands are run from the repository root
(`c:\Users\rayan\Rahul\Github_Projects\aerosim`). The standalone paths import
the FMU source directly and require only `numpy` + `scipy`.

### 4.1 Compose a distributed sim‑config

Merge the modular config into a full AeroSim sim‑config for the orchestrator:

```bash
python examples/shift_missile/compose_sim_config.py examples/config/shift_missile/master_intercept.json
```

This writes the generated file named by the master's `output_sim_config`
(default `sim_config_shift_missile_intercept.generated.json`) into
`examples/config/`. Omit the argument to use `master_intercept.json` by default.
Run the generated sim‑config with the AeroSim orchestrator as with any other
AeroSim scenario.

### 4.2 Standalone single‑shot engagement

Run the entire FMU stack in one process and print a trajectory trace plus the
miss distance, final Mach, motor impulse, and structural margin:

```bash
python examples/shift_missile/engagement.py propnav lqr
```

The two positional arguments select the terminal **guidance law**
(`propnav` | `mpc`) and the inner‑loop **controller** (`lqr` | `pid`); both
default to `propnav lqr`. The run prints a `HIT`/`MISS` verdict (HIT if miss
< 20 m) — useful as a quick smoke test.

### 4.3 Monte‑Carlo P_kill campaign

Run the dispersed‑parameter campaign and report P_kill with confidence bounds:

```bash
python examples/shift_missile/run_monte_carlo.py --runs 200
```

You may pass a master config path and `--runs N` to override the config's
`n_runs`. The run prints the miss‑distance statistics (min, CEP50, mean, p90,
max, std) and three P_kill estimates — Carleton (CLT interval), cookie‑cutter
(Wilson interval), and the closed‑form Rayleigh/Carleton cross‑check — then
writes a `<master-name>.results.json` summary next to the config. With the
shipped `monte_carlo.json`, the campaign uses 200 runs, an 8 m lethal radius,
40 s engagements at a 10 ms step, and disperses aerodynamic, propulsion, and
target (`launch_range_m`, `max_lateral_accel_g`, `weave_frequency_hz`)
parameters.

---

## 5. Known Limitations

- **Airframe stability is not the limiter.** The interceptor is stable through
  the full flight envelope and accelerates to roughly Mach 5 under boost; the
  6‑DOF plant, three‑loop/LQR autopilot, and structural model remain
  well‑behaved throughout.
- **Nominal miss distance is currently a few hundred meters**, dominated by a
  **boost‑phase loft**. The autopilot's gravity‑trim incidence pitches the
  thrust vector up during boost (a gravity‑bias/thrust‑vector coupling),
  lofting the missile off the collision course. This is a **guidance‑refinement
  item** — it needs thrust‑aware gravity‑bias compensation — **not** an
  airframe‑stability defect.
- **Consequently, P_kill against a maneuvering target at the 8 m hit‑to‑kill
  lethal radius is currently low.** The miss is a systematic guidance bias, not
  random dispersion, so tightening the parameter distributions will not raise
  P_kill until the loft is corrected.
- **The Monte‑Carlo P_kill framework is complete and correct.** The sampling,
  miss‑distribution accumulation, Carleton/cookie damage scoring, and
  Wilson/CLT confidence bounds are all implemented and validated; the framework
  will report a high P_kill once the guidance loft is fixed. Diverged runs are
  conservatively scored as misses.
- The threat is a **point‑mass** plant with synthesized attitude and no
  aerodynamic drag/atmosphere coupling; `onboard_nav_error_std_m` is documented
  but not yet exercised by the plant.

---

## 6. References

- P. Zarchan, *Tactical and Strategic Missile Guidance*, 6th ed., Progress in
  Astronautics and Aeronautics, Vol. 239, AIAA, 2012. (Engagement kinematics,
  proportional‑navigation intercept, and weaving‑target engagement analysis.)
- N. A. Shneydor, *Missile Guidance and Pursuit: Kinematics, Dynamics and
  Control*, Horwood Publishing, 1998. (Line‑of‑sight geometry and pursuit/
  proportional‑navigation guidance foundations.)
- R. E. Ball, *The Fundamentals of Aircraft Combat Survivability Analysis and
  Design*, 2nd ed., AIAA Education Series, 2003. (Probability of kill, lethal
  radius, and the Carleton diffuse‑Gaussian damage function.)
- E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical
  Inference," *Journal of the American Statistical Association*, Vol. 22,
  pp. 209–212, 1927. (Wilson score interval for a binomial proportion.)
- Modelica Association, *Functional Mock‑up Interface Specification, Version
  3.0*, 2022. (FMU co‑simulation standard underlying the AeroSim runtime; see
  <https://fmi-standard.org>.)
- International Civil Aviation Organization, *Manual of the ICAO Standard
  Atmosphere* (Doc 7488), 3rd ed. (ISA model used by `atmosphere.py`.)
```