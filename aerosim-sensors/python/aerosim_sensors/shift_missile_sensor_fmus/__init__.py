"""SHIFT missile sensor FMUs for AeroSim.

Ego navigation sensors (derive noisy measurements from the plant truth):
    imu_fmu             -> 100 Hz strapdown IMU (specific force + rates)
    gnss_fmu            ->  10 Hz GNSS position/velocity
    baro_fmu            ->  25 Hz barometric altimeter (ICAO ISA)

Target seekers (observe the threat relative to the ego):
    ir_seeker_fmu        -> passive IR, bearing-only LOS
    semi_active_radar_fmu -> semi-active radar, range + Doppler + bearing

Bundled-as-resource shared module: ``atmosphere.py``.
"""
