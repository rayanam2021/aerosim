@echo off
setlocal
cd /d "%~dp0"

set SRC=python\aerosim_scenarios\shift_missile_scenario_fmus
set REQ=%SRC%\requirements_shift_missile.txt
set SIXDOF=%SRC%\sixdof.py

if not exist "..\examples\fmu" mkdir "..\examples\fmu"

pythonfmu3 build -f %SRC%\threat_missile_fmu.py %SIXDOF% %REQ%

move /y threat_missile_fmu.fmu ..\examples\fmu\ >nul

echo Built SHIFT missile scenario FMUs into ..\examples\fmu\
