from typing import Optional, Iterable
import gymnasium as gym
from scenario_definitions import Scenario, Family
from aeb_ped_env import AEBPedEnv
from aeb_v2v_env import AEBV2VEnv

# ---------- Scenario generator ----------
def generate_fmvss127() -> Iterable[Scenario]:
    # S7.3 Stationary lead
    for v in list(range(10, 81, 10)):
        yield Scenario(Family.V2V_STATIONARY, v, 0, manual_brake=False, note="S7.3 no manual")
    for v in [70, 80, 90, 100]:
        yield Scenario(Family.V2V_STATIONARY, v, 0, manual_brake=True, note="S7.3 manual")
    # S7.4 Slower-moving lead @20
    for v in [40, 50, 60, 70, 80]:
        yield Scenario(Family.V2V_SLOWER_MOVING, v, 20, manual_brake=False, note="S7.4 no manual")
    for v in [70, 80, 90, 100]:
        yield Scenario(Family.V2V_SLOWER_MOVING, v, 20, manual_brake=True, note="S7.4 manual")
    # S7.5 Decelerating lead: both 50 or 80; 0.3–0.5g; headway 12–40
    for v in [50, 80]:
        for hw in [12, 20, 30, 40]:
            for decel_g in [0.3, 0.4, 0.5]:
                for manual in [False, True]:
                    yield Scenario(Family.V2V_DECELERATING, v, v, lead_decel_ms2=decel_g*9.80665,
                                   headway_m=hw, manual_brake=manual, note="S7.5")
    # S8.3 Crossing pedestrian (right @5, left @8), overlaps
    for v in [10, 20, 30, 40, 50, 60]:
        for ov in [0.25, 0.50]:
            yield Scenario(Family.PED_CROSS_RIGHT_5, v, overlap=ov, pedestrian_speed_kmh=5, daylight=True)
    for v in [10, 20, 30, 40, 50, 60]:
        yield Scenario(Family.PED_CROSS_LEFT_8, v, overlap=0.50, pedestrian_speed_kmh=8, daylight=True)
    # S8.3 Obstructed crossing (two parked vehicles)
    for v in [10, 20, 30, 40, 50]:
        yield Scenario(Family.PED_CROSS_OBSTRUCTED, v, overlap=0.50, pedestrian_speed_kmh=5, daylight=True)
    # S8.4 Stationary ped @25% right, day/darkness
    for v in [10, 20, 30, 40, 50, 55]:
        for day in [True, False]:
            yield Scenario(Family.PED_STATIONARY_25R, v, overlap=0.25, daylight=day)
    # S8.5 Along-the-path (parallel)
    for v in [10, 20, 30, 40, 50, 60, 65]:
        for day in [True, False]:
            yield Scenario(Family.PED_ALONG_25R, v, pedestrian_speed_kmh=5, overlap=0.25, daylight=day)

# ---------- Factory ----------
def make_env(s: Scenario, dt: float = 0.05) -> gym.Env:
    if s.family in {Family.V2V_STATIONARY, Family.V2V_SLOWER_MOVING, Family.V2V_DECELERATING}:
        return AEBV2VEnv(s, dt=dt)
    else:
        return AEBPedEnv(s, dt=dt)