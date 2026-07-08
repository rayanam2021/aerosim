#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC=python/aerosim_controllers/shift_missile_controller_fmus
REQ=$SRC/requirements_shift_missile.txt
SIXDOF=$SRC/sixdof.py
QP=$SRC/qp_solver.py
GEOM=$SRC/airframe_geometry.py

mkdir -p ../examples/fmu

# guidance bundles the interior-point QP solver used by the MPC law.
pythonfmu3 build -f $SRC/guidance_fmu.py $QP $REQ
pythonfmu3 build -f $SRC/autopilot_fmu.py $SIXDOF $GEOM $REQ
pythonfmu3 build -f $SRC/ego_nav_ekf_fmu.py $REQ
pythonfmu3 build -f $SRC/target_nav_ekf_fmu.py $REQ

mv -f guidance_fmu.fmu autopilot_fmu.fmu ego_nav_ekf_fmu.fmu \
      target_nav_ekf_fmu.fmu ../examples/fmu/

echo "Built SHIFT missile controller FMUs into ../examples/fmu/"
