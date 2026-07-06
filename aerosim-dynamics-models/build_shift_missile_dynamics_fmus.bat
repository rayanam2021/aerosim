@echo off
setlocal
cd /d "%~dp0"

set SRC=python\aerosim_dynamics_models\shift_missile_dynamics_fmus
set REQ=%SRC%\requirements_shift_missile.txt
set ATM=%SRC%\atmosphere.py
set SIXDOF=%SRC%\sixdof.py
set MODEL=%SRC%\mlp_model.pt

if not exist "%MODEL%" (
    echo mlp_model.pt not found; generating placeholder surrogate...
    python "%SRC%\create_placeholder_mlp_model.py" --out "%MODEL%"
)

if not exist "..\examples\fmu" mkdir "..\examples\fmu"

rem aerodynamics bundles atmosphere.py, sixdof.py and the surrogate weights.
if exist "%MODEL%" (
    pythonfmu3 build -f %SRC%\aerodynamics_sm_fmu.py %ATM% %SIXDOF% %MODEL% %REQ%
) else (
    pythonfmu3 build -f %SRC%\aerodynamics_sm_fmu.py %ATM% %SIXDOF% %REQ%
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
