@echo off
setlocal
cd /d "%~dp0"

set SRC=python\aerosim_dynamics_models\shift_missile_dynamics_fmus
set REQ=%SRC%\requirements_shift_missile.txt
set ATM=%SRC%\atmosphere.py
set SIXDOF=%SRC%\sixdof.py
set GEOM=%SRC%\airframe_geometry.py
set MODEL=%SRC%\mlp_model.pt

if not exist "..\examples\fmu" mkdir "..\examples\fmu"

rem aerodynamics bundles atmosphere.py, sixdof.py, airframe_geometry.py and, if
rem present, the Luminary aero_sm surrogate weights (mlp_model.pt). Without the
rem weights the FMU falls back to the analytic aero tier automatically.
if exist "%MODEL%" (
    pythonfmu3 build -f %SRC%\aerodynamics_sm_fmu.py %ATM% %SIXDOF% %GEOM% %MODEL% %REQ%
) else (
    echo mlp_model.pt not found; building aerodynamics with the analytic fallback tier.
    pythonfmu3 build -f %SRC%\aerodynamics_sm_fmu.py %ATM% %SIXDOF% %GEOM% %REQ%
)
pythonfmu3 build -f %SRC%\servo_sm_fmu.py %REQ%
pythonfmu3 build -f %SRC%\structures_sm_fmu.py %REQ%
pythonfmu3 build -f %SRC%\propulsion_sm_fmu.py %REQ%
pythonfmu3 build -f %SRC%\corrector_fmu.py %REQ%

move /y aerodynamics_sm_fmu.fmu ..\examples\fmu\ >nul
move /y servo_sm_fmu.fmu ..\examples\fmu\ >nul
move /y structures_sm_fmu.fmu ..\examples\fmu\ >nul
move /y propulsion_sm_fmu.fmu ..\examples\fmu\ >nul
move /y corrector_fmu.fmu ..\examples\fmu\ >nul

echo Built SHIFT missile dynamics FMUs into ..\examples\fmu\
