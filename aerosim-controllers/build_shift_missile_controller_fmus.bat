@echo off
setlocal
cd /d "%~dp0"

set SRC=python\aerosim_controllers\shift_missile_controller_fmus
set REQ=%SRC%\requirements_shift_missile.txt
set SIXDOF=%SRC%\sixdof.py

if not exist "..\examples\fmu" mkdir "..\examples\fmu"

pythonfmu3 build -f %SRC%\guidance_fmu.py %REQ%
pythonfmu3 build -f %SRC%\autopilot_fmu.py %SIXDOF% %REQ%
pythonfmu3 build -f %SRC%\ego_nav_ekf_fmu.py %REQ%
pythonfmu3 build -f %SRC%\target_nav_ekf_fmu.py %REQ%

move /y guidance_fmu.fmu ..\examples\fmu\ >nul
move /y autopilot_fmu.fmu ..\examples\fmu\ >nul
move /y ego_nav_ekf_fmu.fmu ..\examples\fmu\ >nul
move /y target_nav_ekf_fmu.fmu ..\examples\fmu\ >nul

echo Built SHIFT missile controller FMUs into ..\examples\fmu\
