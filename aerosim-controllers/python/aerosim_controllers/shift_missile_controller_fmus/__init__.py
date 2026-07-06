"""SHIFT missile GNC (guidance / navigation / control) FMUs for AeroSim.

    guidance_fmu        -> outer-loop intercept guidance (PropNav / MPC-ZEM)
    autopilot_fmu       -> inner-loop 3-axis autopilot (PID / LQR)
    ego_nav_ekf_fmu     -> ego 16-state quaternion INS (error-state EKF)
    target_nav_ekf_fmu  -> threat 9-state kinematic tracker (radar + IR fusion)

Bundled-as-resource shared modules: ``sixdof.py`` and ``atmosphere.py``.
"""
