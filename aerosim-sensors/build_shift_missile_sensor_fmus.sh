#!/bin/bash
set -e
cd "$(dirname "$0")"

SRC=python/aerosim_sensors/shift_missile_sensor_fmus
REQ=$SRC/requirements_shift_missile.txt
ATM=$SRC/atmosphere.py

mkdir -p ../examples/fmu

pythonfmu3 build -f $SRC/imu_fmu.py $REQ
pythonfmu3 build -f $SRC/gnss_fmu.py $REQ
pythonfmu3 build -f $SRC/baro_fmu.py $ATM $REQ
pythonfmu3 build -f $SRC/ir_seeker_fmu.py $REQ
pythonfmu3 build -f $SRC/semi_active_radar_fmu.py $REQ

mv -f imu_fmu.fmu gnss_fmu.fmu baro_fmu.fmu ir_seeker_fmu.fmu \
      semi_active_radar_fmu.fmu ../examples/fmu/

echo "Built SHIFT missile sensor FMUs into ../examples/fmu/"
