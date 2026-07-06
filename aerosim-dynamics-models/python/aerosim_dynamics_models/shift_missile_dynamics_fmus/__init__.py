"""SHIFT missile dynamics/plant FMUs for AeroSim.

Ego interceptor plant and subsystems (all standalone ``Fmi3Slave`` scripts):

    aerodynamics_sm_fmu -> 6-DOF quaternion plant + partial ML aero surrogate
    propulsion_sm_fmu   -> boost/sustain rocket motor
    servo_sm_fmu        -> 3-axis fin actuator dynamics
    structures_sm_fmu   -> mass / full inertia tensor / load factor
    corrector_fmu       -> full 6-DOF EnKF reconstructing all force/moment channels

Shared, bundled-as-resource modules: ``atmosphere.py`` (ICAO ISA) and
``sixdof.py`` (quaternion rigid-body core).
"""
