"""
Run the SHIFT missile interception scenario in AeroSim (distributed).

Scenario
--------
Ego interceptor (actor1) engages a configurable threat missile (actor2).  The
ego runs a full GNC stack; the threat is an autonomous, parameterised target.

Configuration
-------------
Parameters live in the modular JSON files under ``config/shift_missile/``
(missile, target, scenario, master).  This script composes them into a full
sim-config immediately before launch so the distributed run always matches the
standalone ``engagement.py`` harness.

Edit, for example:
    config/shift_missile/scenario_headon_intercept.json   engagement geometry, overrides
    config/shift_missile/missile_shift_interceptor.json   interceptor + GNC tuning
    config/shift_missile/target_cruise_missile.json       threat behaviour

Usage (from repo root):
    python examples/run_shift_missile_intercept.py

Prerequisites:
    - Kafka running (launch_aerosim.sh / launch_aerosim.bat)
    - Built FMUs in examples/fmu/ (run all four build_shift_missile_* scripts)
    - Optional: mlp_model.pt in the dynamics FMU source folder before building
"""

import os
import sys

from aerosim import AeroSim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CONFIG = os.path.join("config", "shift_missile", "master_intercept.json")
CONFIG_FILE = os.path.join("config", "sim_config_shift_missile_intercept.generated.json")


def main() -> None:
    os.chdir(SCRIPT_DIR)
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "shift_missile"))
    import compose_sim_config as composer  # noqa: E402

    master = os.path.join(SCRIPT_DIR, MASTER_CONFIG)
    composer.main([master])
    print(f"Launching distributed sim with {CONFIG_FILE}")

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
