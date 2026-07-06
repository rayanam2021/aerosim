"""
Run the SHIFT missile interception scenario in AeroSim.

Scenario
--------
Ego interceptor (actor1) engages a configurable threat missile (actor2).  The
ego runs a full GNC stack; the threat is an autonomous, parameterised target.

Data flow (all inter-FMU scalars travel on auxiliary JsonData topics)
---------------------------------------------------------------------
    threat_missile ─(VehicleState + pos/vel aux)─> seekers / target EKF

    Ego plant loop:
      autopilot ─(fin + throttle cmd)─> servo_sm
      servo_sm ─(throttle)─> propulsion_sm
      servo_sm ─(fin deflections)─> aerodynamics_sm
      propulsion_sm ─(thrust, propellant)─> aerodynamics_sm, structures_sm
      structures_sm ─(mass, full inertia)─> aerodynamics_sm, corrector, autopilot
      aerodynamics_sm ─(VehicleState truth + 6-DOF true/surrogate forces)─> ...

    Sensors (from ego truth): imu (100 Hz), gnss (10 Hz), baro (25 Hz)
    Seekers (ego vs threat):  ir (bearing-only, 100 Hz), radar (range+bearing, 20 Hz)

    GNC:
      ego_nav_ekf   (IMU+GNSS+baro)          ─(nav state)─> guidance, autopilot, target EKF
      target_nav_ekf(radar+IR, ego nav)      ─(threat state)─> guidance
      guidance (PropNav | MPC)               ─(accel cmd)─> autopilot
      autopilot (PID | LQR)                  ─(fin cmds)─> servo_sm

    corrector: full 6-DOF EnKF fusing ego ground-truth kinematics with the
    partial (fx,fz,my) surrogate to reconstruct ALL six force/moment channels.

Rapidly adjust the engagement by editing ``fmu_initial_vals`` in
``config/sim_config_shift_missile_intercept.json`` — e.g. the threat's
``launch_range_m``, ``cruise_altitude_m``, ``cruise_speed_mps``,
``max_lateral_accel_g`` (agility), ``weave_amplitude_g`` (evasion); the
guidance ``guidance_law`` ("propnav"/"mpc") and the autopilot
``controller_type`` ("pid"/"lqr").

Usage (from repo root):
    python examples/run_shift_missile_intercept.py

Prerequisites:
    - Kafka running (launch_aerosim.sh / launch_aerosim.bat)
    - Built FMUs (from repo root, run each module's build script):
        aerosim-dynamics-models/build_shift_missile_dynamics_fmus.bat  (or .sh)
        aerosim-controllers/build_shift_missile_controller_fmus.bat    (or .sh)
        aerosim-sensors/build_shift_missile_sensor_fmus.bat            (or .sh)
        aerosim-scenarios/build_shift_missile_scenario_fmus.bat        (or .sh)
      (the dynamics build also generates a placeholder mlp_model.pt if the real
       Luminary aero_sm weights are not present)
"""

import os

from aerosim import AeroSim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join("config", "sim_config_shift_missile_intercept.json")


def main() -> None:
    os.chdir(SCRIPT_DIR)
    sim = AeroSim()
    try:
        sim.run(CONFIG_FILE, sim_config_dir=SCRIPT_DIR)
        input("Simulation running. Press Enter to stop...")
    except KeyboardInterrupt:
        print("Stopping simulation...")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
