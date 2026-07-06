#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC=python/aerosim_dynamics_models/shift_missile_dynamics_fmus
REQ=$SRC/requirements_shift_missile.txt
ATM=$SRC/atmosphere.py
SIXDOF=$SRC/sixdof.py
MODEL=$SRC/mlp_model.pt

if [ ! -f "$MODEL" ]; then
    echo "mlp_model.pt not found; generating placeholder surrogate..."
    python "$SRC/create_placeholder_mlp_model.py" --out "$MODEL" \
        || echo "WARNING: could not generate mlp_model.pt (aero FMU uses analytic fallback)"
fi

MODEL_ARG=""
[ -f "$MODEL" ] && MODEL_ARG="$MODEL"

mkdir -p ../examples/fmu

# aerodynamics bundles atmosphere.py, sixdof.py and the surrogate weights.
pythonfmu3 build -f $SRC/aerodynamics_sm_fmu.py $ATM $SIXDOF $MODEL_ARG $REQ
pythonfmu3 build -f $SRC/servo_sm_fmu.py $REQ
pythonfmu3 build -f $SRC/structures_sm_fmu.py $REQ
pythonfmu3 build -f $SRC/propulsion_sm_fmu.py $REQ
pythonfmu3 build -f $SRC/corrector_fmu.py $REQ

mv -f aerodynamics_sm_fmu.fmu servo_sm_fmu.fmu structures_sm_fmu.fmu \
      propulsion_sm_fmu.fmu corrector_fmu.fmu ../examples/fmu/

echo "Built SHIFT missile dynamics FMUs into ../examples/fmu/"
