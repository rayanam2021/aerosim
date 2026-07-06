#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC=python/aerosim_scenarios/shift_missile_scenario_fmus
REQ=$SRC/requirements_shift_missile.txt
SIXDOF=$SRC/sixdof.py

mkdir -p ../examples/fmu

pythonfmu3 build -f $SRC/threat_missile_fmu.py $SIXDOF $REQ

mv -f threat_missile_fmu.fmu ../examples/fmu/

echo "Built SHIFT missile scenario FMUs into ../examples/fmu/"
