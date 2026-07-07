# SHIFT missile interception scenario

This guide documents the **SHIFT missile** co-simulation stack in AeroSim: the ego interceptor plant, guidance/navigation/control (GNC), sensors, threat scenario, how the FMUs connect, and how to build and run the interception example.

For general AeroSim startup (Kafka, orchestrator, renderer), see [Running AeroSim simulations](running_aerosim.md). For sim-config JSON structure, see [Simulation configuration reference](sim_config.md).

---

## Overview

The interception scenario simulates:

1. **Ego vehicle (`actor1`)** — a SHIFT interceptor with a full 6-DOF quaternion plant, rocket propulsion, 3-axis fin actuators, and a partial Luminary `aero_sm` surrogate.
2. **Threat vehicle (`actor2`)** — a configurable guided missile flying toward a defended asset, with optional evasive weave.
3. **Perfect ground truth** — both vehicles publish noise-free `VehicleState` on component topics.
4. **Estimated states** — ego 16-state quaternion INS and a 9-state threat tracker fuse sensors at mismatched rates.
5. **GNC** — outer-loop guidance (Proportional Navigation or MPC-style ZEM) and inner-loop autopilot (PID or LQR) drive the ego fins.
6. **Corrector** — a 6-DOF Ensemble Kalman Filter reconstructs all force/moment channels from ground-truth kinematics, including channels the ML surrogate does not predict.

Simulation clock: **10 ms** (100 Hz). All FMUs step on the same clock; sensors expose a `measurement_ready` flag when a new sample is available so filters can handle asynchronous rates.

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

Example configuration and launcher:

```text
examples/config/sim_config_shift_missile_intercept.json
examples/run_shift_missile_intercept.py
```

Developer closed-loop smoke test (no Kafka; stubs FMU runtime):

```text
_shift_intercept_smoke_test.py
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

- Integrates full rigid-body equations with quaternions (`sixdof.py`).
- Uses a complete **analytic** aerodynamic model (all six force/moment components) plus rocket thrust.
- Evaluates the local **`aero_sm` MLP surrogate**, which in this iteration predicts only:
  - `force_x`, `force_z`, `moment_y` (pitch-plane channels)
- Unknown surrogate channels are published as `0.0` with **`sm_valid_* = 0.0`**. Known channels have **`sm_valid_* = 1.0`**. (JSON aux topics cannot carry `NaN`/`Inf`, so validity flags are the transport-safe equivalent of null.)
- Atmosphere (density, pressure, speed of sound) comes from **ICAO ISA** via `atmosphere.py` at live altitude — nothing is hardcoded.
- Internal sub-stepping (`max_substep_s`, default 1 ms) keeps the stiff plant stable at the 10 ms co-simulation step.

**Key aux inputs:** `elevator_rad`, `aileron_rad`, `rudder_rad`, `thrust_n`, `mass_kg`, `Ixx`, `Iyy`, `Izz`

**Key outputs:** `vehicle_state` (component), true and surrogate 6-DOF forces/moments + validity flags, flight condition (`alpha_deg`, `mach_number`, `dynamic_pressure_pa`, …), mirrored ego kinematics for seekers (`ego_pos_*`, `ego_vel_*`, `ego_q*`, `airspeed_mps`).

### `servo_sm_fmu.py`

Three independent first-order fin actuators (elevator, aileron, rudder) with rate and deflection limits. Passes `throttle` through to propulsion.

### `propulsion_sm_fmu.py`

Boost/sustain solid rocket motor with finite propellant; outputs `thrust_n`, `propellant_fraction`, `mass_flow_kg_s`.

### `structures_sm_fmu.py`

Mass and principal inertias (`Ixx`, `Iyy`, `Izz`) vs. propellant fraction; structural load factor from normal force.

### `corrector_fmu.py`

**Role:** Full 6-DOF Ensemble Kalman Filter.

- **State:** `[Fx, Fy, Fz, Mx, My, Mz]` in body FRD.
- **Observations:** specific force and angular acceleration derived from ego **ground-truth** `VehicleState`.
- Surrogate-valid channels are centered on the ML prediction; unknown channels use a diffuse persistence prior so kinematics alone determine them.
- Outputs corrected forces/moments and estimated surrogate bias on valid channels.

### Shared: `atmosphere.py`, `sixdof.py`

- **`atmosphere.py`** — ICAO standard atmosphere (0–47 km).
- **`sixdof.py`** — quaternion helpers and semi-implicit 6-DOF integrator.

---

## Controller FMUs (`shift_missile_controller_fmus`)

Controllers live in **`aerosim-controllers`**, consistent with existing JSBSim autopilot / flight-controller FMUs in this repo.

### `guidance_fmu.py` — outer loop

| Parameter | Values | Description |
|-----------|--------|-------------|
| `guidance_law` | `"propnav"` or `"mpc"` | Proportional Navigation vs. optimal ZEM-style predictive guidance |
| `nav_gain` | float | PropNav navigation constant |
| `mpc_gain` | float | ZEM guidance gain |
| `max_accel_g` | float | Acceleration command limit |

Inputs: ego nav estimate + target nav estimate. Outputs: `a_cmd_n/e/d`, range, closing speed, time-to-go, LOS rate, ZEM, `guidance_active`.

### `autopilot_fmu.py` — inner loop

| Parameter | Values | Description |
|-----------|--------|-------------|
| `controller_type` | `"pid"` or `"lqr"` | Fixed-gain PID vs. online LQR from short-period linearization |
| `max_alpha_cmd_rad` | float | Angle-of-attack command limit |
| `lqr_q_angle`, `lqr_q_rate`, `lqr_r_fin` | floats | LQR weights |

Skid-to-turn: pitch/yaw channels track lateral acceleration from guidance; roll channel holds wings level via moment-coefficient feedback inverted through `Cl_da`.

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
3. **`sim_config_shift_missile_intercept.json`** sets `"mlp_model_path": "mlp_model.pt"` under the aerodynamics FMU's `fmu_initial_vals` (filename inside the built FMU bundle).

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

### 3. Launch the scenario

```sh
cd examples
python run_shift_missile_intercept.py
```

The script loads `config/sim_config_shift_missile_intercept.json`, registers all 15 FMUs, and runs until you press Enter.

### 4. Headless numerical test (optional)

Without Kafka or built FMUs, you can exercise the Python FMU logic directly:

```sh
# From repo root
python _shift_intercept_smoke_test.py              # default: propnav + lqr
python _shift_intercept_smoke_test.py mpc pid      # alternate laws
```

---

## Configuring the scenario

Edit `examples/config/sim_config_shift_missile_intercept.json`.

### Threat behavior

Under the `threat_missile` FMU block, `fmu_initial_vals`:

```json
"launch_range_m": 20000.0,
"cruise_altitude_m": 8000.0,
"cruise_speed_mps": 300.0,
"max_lateral_accel_g": 8.0,
"weave_amplitude_g": 2.0,
"weave_frequency_hz": 0.2
```

Changes take effect on the next simulation start (rebuild FMUs not required for parameter-only edits).

### Guidance and autopilot laws

```json
// guidance FMU
"guidance_law": "propnav",   // or "mpc"
"nav_gain": 3.0,
"max_accel_g": 15.0

// autopilot FMU
"controller_type": "lqr",    // or "pid"
"max_alpha_cmd_rad": 0.20
```

### Ego initial conditions

Under `aerodynamics_sm` `fmu_initial_vals`:

```json
"init_pos_down_m": -8000.0,
"init_speed_mps": 600.0,
"init_yaw_rad": 0.0
```

NED down is positive downward; `-8000` m down ≈ 8000 m MSL when `world_origin_altitude` is 0.

### Sensor and filter tuning

Each FMU block has its own `fmu_initial_vals` (noise standard deviations, update rates, EnKF ensemble size, etc.). Scalar signals are mapped through `fmu_aux_input_mapping` / `fmu_aux_output_mapping` — see [Simulation configuration reference](sim_config.md).

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
| Ego loses energy / misses target | Aggressive guidance | Reduce `max_accel_g`, `max_alpha_cmd_rad`; tune `nav_gain` |
| `guidance_active = 0` | Nav or target track not valid | Wait for GNSS fix and radar lock; check seeker FOV/range |
| Seekers never lock | Geometry / FOV | Reduce `launch_range_m` or increase `max_lock_range_m` / FOV params |

---

## Related documentation

- [FMU reference](fmu_reference.md) — creating and registering FMUs
- [Simulation configuration reference](sim_config.md) — JSON schema and topic mapping
- [Conventions](conventions.md) — frames and message types
