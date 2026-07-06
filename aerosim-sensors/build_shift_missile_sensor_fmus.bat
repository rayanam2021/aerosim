@echo off
setlocal
cd /d "%~dp0"

set SRC=python\aerosim_sensors\shift_missile_sensor_fmus
set REQ=%SRC%\requirements_shift_missile.txt
set ATM=%SRC%\atmosphere.py

if not exist "..\examples\fmu" mkdir "..\examples\fmu"

pythonfmu3 build -f %SRC%\imu_fmu.py %REQ%
pythonfmu3 build -f %SRC%\gnss_fmu.py %REQ%
pythonfmu3 build -f %SRC%\baro_fmu.py %ATM% %REQ%
pythonfmu3 build -f %SRC%\ir_seeker_fmu.py %REQ%
pythonfmu3 build -f %SRC%\semi_active_radar_fmu.py %REQ%

move /y imu_fmu.fmu ..\examples\fmu\ >nul
move /y gnss_fmu.fmu ..\examples\fmu\ >nul
move /y baro_fmu.fmu ..\examples\fmu\ >nul
move /y ir_seeker_fmu.fmu ..\examples\fmu\ >nul
move /y semi_active_radar_fmu.fmu ..\examples\fmu\ >nul

echo Built SHIFT missile sensor FMUs into ..\examples\fmu\
