#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC=python/aerosim_dynamics_models/shift_missile_dynamics_fmus
REQ=$SRC/requirements_shift_missile.txt
ATM=$SRC/atmosphere.py
SIXDOF=$SRC/sixdof.py
GEOM=$SRC/airframe_geometry.py
MODEL=$SRC/mlp_model.pt

MODEL_ARG=""
if [ -f "$MODEL" ]; then
    MODEL_ARG="$MODEL"
else
    echo "mlp_model.pt not found; building aerodynamics with the analytic fallback tier."
fi

mkdir -p ../examples/fmu

# aerodynamics bundles atmosphere.py, sixdof.py, airframe_geometry.py and, if
# present, the Luminary aero_sm surrogate weights (mlp_model.pt).
pythonfmu3 build -f $SRC/aerodynamics_sm_fmu.py $ATM $SIXDOF $GEOM $MODEL_ARG $REQ
pythonfmu3 build -f $SRC/servo_sm_fmu.py $REQ
pythonfmu3 build -f $SRC/structures_sm_fmu.py $REQ
pythonfmu3 build -f $SRC/propulsion_sm_fmu.py $REQ
pythonfmu3 build -f $SRC/corrector_fmu.py $REQ

mv -f aerodynamics_sm_fmu.fmu servo_sm_fmu.fmu structures_sm_fmu.fmu \
      propulsion_sm_fmu.fmu corrector_fmu.fmu ../examples/fmu/

echo "Built SHIFT missile dynamics FMUs into ../examples/fmu/"
