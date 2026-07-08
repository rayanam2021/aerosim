# SHIFT missile interception scenario

This guide documents the **SHIFT missile** co-simulation stack in AeroSim: the ego interceptor plant, guidance/navigation/control (GNC), sensors, threat scenario, how the FMUs connect, and how to build and run the interception example.

For general AeroSim startup (Kafka, orchestrator, renderer), see [Running AeroSim simulations](running_aerosim.md). For sim-config JSON structure, see [Simulation configuration reference](sim_config.md).

---

## Overview

The interception scenario simulates:

1. **Ego vehicle (`actor1`)** — a SHIFT interceptor with a full 6-DOF quaternion plant integrated by **RK4**, a physics-based solid rocket motor, modular **4-canard + 4-tail** (diamond/trapezoid) geometry driving equivalent elevator/aileron/rudder control derivatives, an Euler–Bernoulli structural model with canard/tail load stations, and a partial Luminary `aero_sm` surrogate with per-channel uncertainty.
2. **Threat vehicle (`actor2`)** — a configurable guided missile flying toward a defended asset, with optional evasive weave.
3. **Perfect ground truth** — both vehicles publish noise-free `VehicleState` on component topics.
4. **Estimated states** — ego 16-state quaternion INS and a 9-state threat tracker fuse noisy, asynchronous sensors at mismatched rates.
5. **Fire control + GNC** — a phased fire-control computer (IDLE → MIDCOURSE → TERMINAL) runs initial guidance to a predicted intercept point, then terminal homing with a selectable law — true vector **Proportional Navigation / Augmented PN** or a full **receding-horizon MPC** (QP solved by an interior-point solver; the old Zero-Effort-Miss shortcut has been removed) — and an inner-loop **three-loop autopilot** (PID or online LQR) drives the ego fins.
6. **Corrector** — a 6-DOF Ensemble Kalman Filter reconstructs all force/moment channels from ground-truth kinematics, driven by the surrogate's per-channel uncertainty, and exposes posterior standard deviations.
7. **Uncertainty quantification → P_kill** — surrogate/propulsion/structural uncertainty is propagated into a single-shot probability of kill (P_kill) with confidence bounds via a Monte-Carlo campaign.

Simulation clock: **10 ms** (100 Hz). All FMUs step on the same clock; sensors expose a `measurement_ready` flag when a new sample is available so filters can handle asynchronous rates.

> **Detailed math and references** live in a `README.md` inside each of the four `shift_missile_*_fmus` folders (dynamics, controllers, sensors, scenarios). This guide is the operational overview; those READMEs are the technical reference.

---

## Repository layout

The stack is split across four existing AeroSim packages (not a single monolithic folder):

| Package | Python path | Role |
|---------|-------------|------|
| `aerosim-dynamics-models` | `python/aerosim_dynamics_models/shift_missile_dynamics_fmus/` | Ego plant + EnKF corrector |
| `aerosim-controllers` | `python/aerosim_controllers/shift_missile_controller_fmus/` | Guidance, autopilot, navigation EKFs |
| `aerosim-sensors` | `python/aerosim_sensors/shift_missile_sensor_fmus/` | IMU, GNSS, baro, IR, semi-active radar |
| `aerosim-scenarios` | `python/aerosim_scenarios/shift_missile_scenario_fmus/` | Configurable threat missile |

Built FMUs are copied to:

```text
examples/fmu/
```

Shared utility modules (`atmosphere.py`, `sixdof.py`) are **bundled into each FMU** that needs them at build time so every FMU remains self-contained.

**Modular configuration** (small, composable JSON files — see [Configuring the scenario](#configuring-the-scenario)):

```text
examples/config/shift_missile/master_intercept.json      high-level run selector
examples/config/shift_missile/scenario_headon_intercept.json
examples/config/shift_missile/missile_shift_interceptor.json
examples/config/shift_missile/target_cruise_missile.json
examples/config/shift_missile/monte_carlo.json            P_kill campaign
examples/config/shift_missile/sweep_intercept.json        multi-config sweep
```

**Standalone tooling** (no Kafka/orchestrator; stubs the FMU runtime in-process, faster than real time):

```text
examples/shift_missile/compose_sim_config.py   modular JSON -> full AeroSim sim-config
examples/shift_missile/engagement.py           one closed-loop engagement + telemetry
examples/shift_missile/run_monte_carlo.py       dispersed P_kill campaign
examples/shift_missile/run_sweep.py             many configurations via delta storage
```

---

## Architecture and data flow

```text
                         THREAT (actor2)                    EGO (actor1)
                         -------------                    -----------

  threat_missile_fmu  -->  vehicle_state (truth)
        |                  threat_pos/vel (aux)
        |                           |
        |                           +-----> ir_seeker_fmu ----+
        |                           +-----> semi_active_radar -+--> target_nav_ekf_fmu
        |                           |                              (9-state tracker)
        |                           |                                    |
        |                           |                                    v
        |                           |                            guidance_fmu
        |                           |                           (PropNav | MPC)
        |                           |                                    |
        |                           |                                    v
        |                           |                            autopilot_fmu
        |                           |                           (PID | LQR)
        |                           |                                    |
        |                           |                                    v
        |                           |                             servo_sm_fmu
        |                           |                                    |
        |                           +<--- aero mirrors ego truth ------+---> propulsion_sm_fmu
        |                                                                |
        |                                                                v
        |                                                         structures_sm_fmu
        |                                                                |
        v                                                                v
  (seekers use aux pos/vel)                              aerodynamics_sm_fmu (6-DOF plant)
                                                                        |
                                        +-------------------------------+
                                        | vehicle_state (truth)
                                        v
                              imu / gnss / baro  -->  ego_nav_ekf_fmu (16-state INS)
                                                                        |
                                        corrector_fmu (6-DOF EnKF) <----+
                                        (uses truth + partial surrogate)
```

**Important wiring constraint:** the FMU driver maps every `VehicleState` component input to the variable prefix `vehicle_state.`. A single FMU cannot subscribe to two different `VehicleState` topics. Seekers therefore read ego and threat kinematics as **scalar aux signals** (the ego plant mirrors its truth to aux; the threat FMU publishes its own aux outputs).

---

## Dynamics FMUs (`shift_missile_dynamics_fmus`)

### `aerodynamics_sm_fmu.py`

**Role:** Perfect ground-truth 6-DOF plant for the ego interceptor.

- Integrates full rigid-body 6-DOF equations of motion with quaternions using **RK4** (`integrate_6dof_rk4` in `sixdof.py`). The EOM are *fed by* the Luminary surrogate for aerodynamic coefficients/forces, with the analytic model as fallback — a high-fidelity, faster-than-real-time hybrid rather than pure-analytic or pure-surrogate.
- Uses a complete **analytic** aerodynamic model (all six force/moment components) plus rocket thrust when the surrogate is unavailable.
- Evaluates the local **`aero_sm` MLP surrogate**, which in this iteration predicts only:
  - `force_x`, `force_z`, `moment_y` (pitch-plane channels)
- Unknown surrogate channels are published as `0.0` with **`sm_valid_* = 0.0`**. Known channels have **`sm_valid_* = 1.0`**. (JSON aux topics cannot carry `NaN`/`Inf`, so validity flags are the transport-safe equivalent of null.)
- Emits **per-channel surrogate uncertainty** `sm_sigma_fx/fy/fz/mx/my/mz` (relative + absolute σ with edge inflation) consumed by the corrector.
- Atmosphere (density, pressure, speed of sound) comes from **ICAO ISA** via `atmosphere.py` at live altitude — nothing is hardcoded. Also outputs `air_pressure_pa` for the propulsion ambient correction.
- Internal sub-stepping (`max_substep_s`, default 1 ms) keeps the stiff plant stable at the 10 ms co-simulation step.

**Key aux inputs:** `elevator_rad`, `aileron_rad`, `rudder_rad`, `thrust_n`, `mass_kg`, `Ixx`, `Iyy`, `Izz`

**Key outputs:** `vehicle_state` (component), true and surrogate 6-DOF forces/moments + validity flags, flight condition (`alpha_deg`, `mach_number`, `dynamic_pressure_pa`, …), mirrored ego kinematics for seekers (`ego_pos_*`, `ego_vel_*`, `ego_q*`, `airspeed_mps`).

### `servo_sm_fmu.py`

Three independent first-order fin actuators (elevator, aileron, rudder) with rate and deflection limits. Passes `throttle` through to propulsion.

### `propulsion_sm_fmu.py`

Physics-based **0-D internal-ballistics** solid rocket motor (with an optional liquid mode). A cylindrical BATES-style grain burns per **St. Robert's law** (\(r = a\,p_c^{\,n}\)); chamber pressure is solved from the mass balance at equilibrium, and thrust is ambient-pressure corrected. The nozzle throat can be auto-sized for a design chamber pressure. Outputs `thrust_n`, `thrust_sigma_n` (uncertainty), `propellant_fraction`, `mass_flow_kg_s`, `chamber_pressure_pa`, and quantities of interest `isp_s`, `total_impulse_ns`, `burn_time_s`.

### `structures_sm_fmu.py`

**Euler–Bernoulli beam** load solver. From the aerodynamic normal/side forces and thrust it computes internal shear and bending moment, peak bending stress (\(\sigma = Mc/I\)), axial stress, combined von-Mises stress, **margin of safety** (1.5 factor of safety), and the **first bending natural frequency**, plus a probabilistic `prob_structural_failure` from load/material scatter. Also outputs mass and principal inertias (`Ixx`, `Iyy`, `Izz`) and CG vs. propellant fraction.

### `corrector_fmu.py`

**Role:** Full 6-DOF Ensemble Kalman Filter.

- **State:** `[Fx, Fy, Fz, Mx, My, Mz]` in body FRD.
- **Observations:** specific force and angular acceleration derived from ego **ground-truth** `VehicleState`.
- Surrogate-valid channels are centered on the ML prediction; **per-channel surrogate 1-σ (`sm_sigma_*`) dynamically sets the process noise**. Unknown channels use a diffuse persistence prior so kinematics alone determine them.
- Outputs corrected forces/moments plus **posterior per-channel standard deviations** (`std_fx_n`, …) that feed downstream P_kill uncertainty.

### Shared: `atmosphere.py`, `sixdof.py`

- **`atmosphere.py`** — ICAO standard atmosphere (0–47 km).
- **`sixdof.py`** — quaternion kinematics helpers (`quat_mul`, `quat_deriv`) and a classical **RK4** 6-DOF integrator (`integrate_6dof_rk4`).

---

## Controller FMUs (`shift_missile_controller_fmus`)

Controllers live in **`aerosim-controllers`**, consistent with existing JSBSim autopilot / flight-controller FMUs in this repo.

### `guidance_fmu.py` — fire control + outer loop

Sequences the engagement through fire-control phases IDLE → MIDCOURSE (lead-collision toward the predicted intercept point) → TERMINAL (selected homing law).

| Parameter | Values | Description |
|-----------|--------|-------------|
| `guidance_law` | `"propnav"` or `"mpc"` | Terminal homing: true vector Proportional Navigation vs. full receding-horizon MPC |
| `nav_gain` | float | PropNav effective navigation ratio \(N'\) |
| `augmented_propnav` | 0/1 | Add the APN target-acceleration feed-forward term |
| `terminal_range_m` | float | Range at which midcourse hands over to terminal homing |
| `midcourse_gain`, `midcourse_max_g` | floats | Lead-collision gain and energy-managed accel cap |
| `mpc_horizon`, `mpc_w_miss`, `mpc_w_effort`, `mpc_w_rate` | — | MPC horizon length and QP weights |
| `command_ramp_s` | float | Soft-start ramp on the command at guidance hand-off (default 0 = off) |
| `max_accel_g` | float | Acceleration command limit |

Inputs: ego nav estimate + target nav estimate (+ optional launcher cue). Outputs: `a_cmd_n/e/d`, range, closing speed, time-to-go, LOS rate, PIP, `guidance_phase`, `guidance_active`. **The Zero-Effort-Miss law has been removed**; `zem_m` remains as a diagnostic only.

### `autopilot_fmu.py` — inner loop

Classical **three-loop** skid-to-turn autopilot: the guidance NED acceleration command is converted to incidence commands (\(\alpha,\beta\)) via quasi-static trim, an incidence-trim integral outer loop removes steady error, and a low-latency **rate-gyro** inner loop damps the short-period mode.

| Parameter | Values | Description |
|-----------|--------|-------------|
| `controller_type` | `"pid"` or `"lqr"` | Fixed rate-damping gain vs. online LQR short-period Riccati gain |
| `use_gyro_rate` | 0/1 | Close the fast loop on the raw IMU gyro (1, default) or nav estimate (0) |
| `max_incidence_rad` | float | Incidence (α/β) command clamp (~17° default) |
| `ki_accel` | float | Incidence-trim integral gain [1/s] |
| `kq_rate` | float | Fixed rate-damping gain (PID option) |
| `lqr_q_angle`, `lqr_q_rate`, `lqr_r_fin` | floats | LQR short-period weights |

Roll channel holds wings level via moment-coefficient feedback inverted through `Cl_da`.

### `ego_nav_ekf_fmu.py` — ego navigation

16-state quaternion INS implemented as a **15-error-state MEKF**:

- Nominal: position, velocity, quaternion, accel bias, gyro bias.
- Fuses IMU (100 Hz), GNSS (10 Hz), baro (25 Hz) using each sensor's `measurement_ready` flag.

### `target_nav_ekf_fmu.py` — threat tracking

9-state constant-acceleration tracker: `[pN, pE, pD, vN, vE, vD, aN, aE, aD]`.

- **Radar update:** range, azimuth, elevation, range-rate (nonlinear observation, numeric Jacobian).
- **IR update:** azimuth, elevation only (bearing-only).
- Initialized on first radar lock; requires ego nav for relative geometry.

---

## Sensor FMUs (`shift_missile_sensor_fmus`)

| FMU | Rate (default) | Measures | Notes |
|-----|----------------|----------|-------|
| `imu_fmu` | 100 Hz | Specific force + body rates | Noise, bias, scale error |
| `gnss_fmu` | 10 Hz | NED position + velocity | Optional dropout |
| `baro_fmu` | 25 Hz | Pressure altitude | ISA + calibration bias |
| `ir_seeker_fmu` | 100 Hz | LOS az/el (NED) | Passive, bearing-only, FOV + max range |
| `semi_active_radar_fmu` | 20 Hz | Range, az/el, range-rate | Semi-active homing, FOV + max range |

All navigation sensors read ego **`vehicle_state`** (truth). Seekers read **aux** ego/threat kinematics (see wiring constraint above).

---

## Scenario FMU (`shift_missile_scenario_fmus`)

### `threat_missile_fmu.py`

Autonomous guided threat with 6-DOF-consistent attitude (quaternion from flight-path + coordinated turn bank).

**Tunable parameters** (via `fmu_initial_vals` in sim config):

| Parameter | Effect |
|-----------|--------|
| `launch_range_m`, `launch_bearing_deg` | Initial engagement geometry |
| `cruise_altitude_m`, `cruise_speed_mps` | Cruise profile |
| `max_lateral_accel_g` | Agility / control-surface authority |
| `max_axial_accel_mps2` | Speed-hold authority |
| `target_north_m`, `target_east_m` | Ingress waypoint |
| `weave_amplitude_g`, `weave_frequency_hz` | Evasive weave |
| `onboard_nav_error_std_m` | Documented nav grade (future use) |

---

## Luminary `mlp_model.pt`

Place the Luminary **`aero_sm`** TorchScript weights here **before building** the aerodynamics FMU:

```text
aerosim-dynamics-models/python/aerosim_dynamics_models/shift_missile_dynamics_fmus/mlp_model.pt
```

Full path on this machine (adjust for your clone):

```text
<repo-root>/aerosim-dynamics-models/python/aerosim_dynamics_models/shift_missile_dynamics_fmus/mlp_model.pt
```

**How it is used:**

1. **`aerodynamics_sm_fmu`** loads the file via parameter `mlp_model_path` (default `"mlp_model.pt"`). It searches the path as given, then the FMU source directory.
2. **`build_shift_missile_dynamics_fmus.bat` / `.sh`** bundles `mlp_model.pt` into `aerodynamics_sm_fmu.fmu` when present. If the file is missing, the FMU falls back to analytic aero at runtime.
3. **`missile_shift_interceptor.json`** sets `"mlp_model_path": "mlp_model.pt"` under the aerodynamics FMU block (filename inside the built FMU bundle); the composer injects it into the generated sim-config.

**Expected model I/O:**

- **Inputs:** `[mach, alpha_deg, elevator_deg]` (TorchScript tensor)
- **Outputs:** `[force_x_n, force_z_n, moment_y_nm]` in body FRD

Do **not** commit proprietary Luminary weights to git unless your license allows it; keep `mlp_model.pt` local and rebuild the FMU after updating weights.

---

## Prerequisites

1. AeroSim built and virtual environment activated — see [Build AeroSim in Windows](build_windows.md) or [Build AeroSim in Linux](build_linux.md).
2. Python packages for FMU build (in the AeroSim venv):
   - `pythonfmu3`
   - `numpy`, `scipy`
   - `torch` (for aerodynamics FMU; analytic fallback works without a working torch load at runtime)
3. Running simulation infrastructure: Kafka, orchestrator, FMU driver — see [Running AeroSim simulations](running_aerosim.md).

Dynamics FMU Python requirements file:

```text
aerosim-dynamics-models/python/aerosim_dynamics_models/shift_missile_dynamics_fmus/requirements_shift_missile.txt
```

---

## Building the FMUs

From the **repository root**, run all four build scripts (order does not matter):

**Windows (PowerShell or cmd):**

```bat
aerosim-dynamics-models\build_shift_missile_dynamics_fmus.bat
aerosim-controllers\build_shift_missile_controller_fmus.bat
aerosim-sensors\build_shift_missile_sensor_fmus.bat
aerosim-scenarios\build_shift_missile_scenario_fmus.bat
```

**Linux / macOS:**

```sh
./aerosim-dynamics-models/build_shift_missile_dynamics_fmus.sh
./aerosim-controllers/build_shift_missile_controller_fmus.sh
./aerosim-sensors/build_shift_missile_sensor_fmus.sh
./aerosim-scenarios/build_shift_missile_scenario_fmus.sh
```

Expected outputs in `examples/fmu/`:

| FMU file | Source module |
|----------|---------------|
| `aerodynamics_sm_fmu.fmu` | dynamics |
| `servo_sm_fmu.fmu` | dynamics |
| `propulsion_sm_fmu.fmu` | dynamics |
| `structures_sm_fmu.fmu` | dynamics |
| `corrector_fmu.fmu` | dynamics |
| `guidance_fmu.fmu` | controllers |
| `autopilot_fmu.fmu` | controllers |
| `ego_nav_ekf_fmu.fmu` | controllers |
| `target_nav_ekf_fmu.fmu` | controllers |
| `imu_fmu.fmu` | sensors |
| `gnss_fmu.fmu` | sensors |
| `baro_fmu.fmu` | sensors |
| `ir_seeker_fmu.fmu` | sensors |
| `semi_active_radar_fmu.fmu` | sensors |
| `threat_missile_fmu.fmu` | scenarios |

---

## Running the interception scenario

### 1. Start AeroSim services

```sh
# From repo root (see launch script options in running_aerosim.md)
./launch_aerosim.sh          # Linux
launch_aerosim.bat           # Windows
```

Start the renderer if you want a visual scene (optional for headless FMU-only runs).

### 2. Activate the Python environment

```sh
source .venv/bin/activate          # Linux
.venv\Scripts\activate             # Windows
```

### 3. Compose and launch the scenario

The modular config is composed into a full AeroSim sim-config, then launched:

```sh
# From repo root — generate the sim-config from the modular master file
python examples/shift_missile/compose_sim_config.py examples/config/shift_missile/master_intercept.json
# -> writes examples/config/sim_config_shift_missile_intercept.generated.json
```

Then run it through the standard AeroSim launcher (registers all 15 FMUs). See [Running AeroSim simulations](running_aerosim.md).

### 4. Headless standalone runs (no Kafka, faster than real time)

The standalone harness stubs the FMU runtime and runs the entire wired stack in one Python process — ideal for tuning, CI, and Monte-Carlo.

**Single engagement + telemetry:**

```sh
# From repo root:  python examples/shift_missile/engagement.py [propnav|mpc] [lqr|pid]
python examples/shift_missile/engagement.py propnav lqr
```

Expected output (nominal head-on case; abbreviated):

```text
[engagement] law=propnav controller=lqr model=analytic
  t[s]   range[m]   alt[m]  mach  ph    MoS nav   |acmd|   alpha    elev
  1.01    19030.1   7998.0  2.34   1   6.22   1      9.7    6.83   -1.46
  ...
 14.01     1148.9   8135.8  2.76   2   7.87   1    294.2   -2.00   -3.88

  miss distance   : 367.4 m at t=14.96 s
  final mach      : 2.72
  total impulse   : 207.6 kN.s over 7.24 s
  min struct MoS  : 31.80
  RESULT: MISS
```

> **Note on `model=analytic`:** if `torch` cannot load `mlp_model.pt`, the plant uses the analytic aero fallback and prints `model=analytic`. This is expected without the proprietary weights.

**Monte-Carlo P_kill campaign:**

```sh
python examples/shift_missile/run_monte_carlo.py --runs 200
```

Expected output (abbreviated):

```text
[monte-carlo] Head-on intercept P_kill Monte-Carlo
  runs=200  lethal_radius=8.0 m  damage_model=carleton  confidence=95%
  ...
  miss distance   min=... CEP=... mean=... p90=... max=... m
  P_kill (Carleton, r_L=8 m) = 0.0xx  [0.0xx, 0.0xx]  (95% CI)
  P_kill (cookie-cutter)     = 0.0xx  [0.0xx, 0.0xx]  (Wilson 95%)
  wrote summary -> examples/config/master_intercept.results.json
```

---

## Configuring the scenario

The scenario is described by **small, composable JSON files** so many scenarios can be built without duplicating a large monolithic config. The topology (the FMU graph) is fixed in `compose_sim_config.py`; the JSON files carry only parameters, world/clock, and engagement geometry.

```text
master_intercept.json          high-level run: which scenario + which Monte-Carlo campaign
  └─ scenario_headon_intercept.json   world/clock + engagement geometry + which missile & target
       ├─ missile_shift_interceptor.json   interceptor design + GNC parameters (per-FMU)
       └─ target_cruise_missile.json       threat missile parameters
  └─ monte_carlo.json           P_kill campaign: n_runs, lethal radius, damage model, uncertain params
```

- **`missile_shift_interceptor.json`** — per-FMU `fmus` blocks (aerodynamics, servo, propulsion, structures, corrector, sensors, guidance, autopilot, EKFs). Guidance/autopilot law selection, gains, motor geometry, structural properties, sensor noise, EKF tuning.
- **`target_cruise_missile.json`** — threat `threat_missile` parameters (see the threat table above).
- **`scenario_headon_intercept.json`** — `world.origin`, `clock`, `engagement.ego` (initial position/attitude/speed; NED down positive, `-8000` ≈ 8000 m MSL when origin altitude is 0), `engagement.target` (launch geometry, cruise, weave), and free-form `overrides` (e.g. `"guidance.guidance_law": "mpc"`).
- **`master_intercept.json`** — references a scenario and a Monte-Carlo file and names the composed output; the single entry point for a run.

Parameter-only edits take effect on the next composition/run (no FMU rebuild required). Signals are mapped through `fmu_aux_input_mapping` / `fmu_aux_output_mapping` — see [Simulation configuration reference](sim_config.md).

### Many configurations: delta-based sweeps

To run many varying configurations **without duplicating full config files**, use the sweep runner. A sweep manifest references the base master **once** (content-hashed for provenance) and stores each case as a small `overrides` delta; cases can be listed explicitly and/or generated as the Cartesian product of a `grid`:

```sh
python examples/shift_missile/run_sweep.py examples/config/shift_missile/sweep_intercept.json
python examples/shift_missile/run_sweep.py --montecarlo     # P_kill campaign per case
```

Outputs go to `examples/runs/<sweep>_<timestamp>/`:

- **`manifest.json`** — base reference + base content hash + every case delta and its scalar result summary (human-readable campaign index).
- **`results.jsonl`** — one JSON record per run (append-friendly, streaming; loads directly into pandas/DuckDB). A 10 000-case campaign costs one base file plus a compact override list, not 10 000 monolithic JSON blobs, and every resolved config is reproducible from `base + delta`.

---

## Ground truth vs. estimates

| Signal | Source | Used by |
|--------|--------|---------|
| `aerosim.actor1.vehicle_state` | `aerodynamics_sm_fmu` plant | Renderer, IMU/GNSS/baro truth, corrector |
| `aerosim.actor2.vehicle_state` | `threat_missile_fmu` | Renderer, threat aux for seekers |
| `aerosim.actor1.egonav.aux_out` | `ego_nav_ekf_fmu` | Guidance, autopilot, target EKF |
| `aerosim.actor1.tgtnav.aux_out` | `target_nav_ekf_fmu` | Guidance |

Guidance and autopilot **never** consume ground truth directly — only navigation estimates. The corrector **does** use ego truth kinematics to estimate the full force/moment vector and surrogate bias.

---

## Coordinate frames

- **World:** NED (North-East-Down), right-handed, **z** down.
- **Body:** FRD (x forward, y right, z down), origin at CG.
- **Quaternion:** scalar-last `[x, y, z, w]` internally; AeroSim `Orientation` messages use `[w, x, y, z]` on the wire.

---

## Partial surrogate and corrector behavior

The ML surrogate provides only three of six force/moment channels. The aerodynamics FMU publishes:

```text
sm_valid_fx = sm_valid_fz = sm_valid_my = 1.0
sm_valid_fy = sm_valid_mx = sm_valid_mz = 0.0
sm_fy_n = sm_mx_nm = sm_mz_nm = 0.0   (sentinel values)
```

The corrector EnKF uses tight process noise on valid channels and diffuse noise on unknown channels, then assimilates accelerations from truth. In practice this recovers side force and roll/yaw moments that the surrogate omits.

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `analytic (mlp_model.pt not found)` in `model_source` | Weights not bundled | Copy `mlp_model.pt` to dynamics folder and rebuild aerodynamics FMU |
| Torch load error on Windows / Python 3.13 | Environment issue | Use supported Python/torch combo, or rely on analytic fallback |
| Ego loses energy / misses target | Aggressive guidance | Reduce `max_accel_g`, `max_incidence_rad`, `midcourse_max_g`; tune `nav_gain` |
| `guidance_active = 0` | Nav or target track not valid | Wait for GNSS fix and radar lock; check seeker FOV/range |
| Seekers never lock | Geometry / FOV | Reduce `launch_range_m` or increase `max_lock_range_m` / FOV params |

---

## Related documentation

- [FMU reference](fmu_reference.md) — creating and registering FMUs
- [Simulation configuration reference](sim_config.md) — JSON schema and topic mapping
- [Conventions](conventions.md) — frames and message types

**Per-folder technical READMEs** (executive summary, detailed math, component descriptions, references):

- `aerosim-dynamics-models/python/aerosim_dynamics_models/shift_missile_dynamics_fmus/README.md`
- `aerosim-controllers/python/aerosim_controllers/shift_missile_controller_fmus/README.md`
- `aerosim-sensors/python/aerosim_sensors/shift_missile_sensor_fmus/README.md`
- `aerosim-scenarios/python/aerosim_scenarios/shift_missile_scenario_fmus/README.md`
