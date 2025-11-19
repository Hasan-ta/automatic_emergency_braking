import enum
from dataclasses import dataclass
from typing import Optional

# ---------- Scenario taxonomy ----------
class Family(enum.Enum):
  V2V_STATIONARY = "S7.3"
  V2V_SLOWER_MOVING = "S7.4"
  V2V_DECELERATING = "S7.5"
  PED_CROSS_RIGHT_5 = "S8.3 right 5 km/h"
  PED_CROSS_LEFT_8 = "S8.3 left 8 km/h"
  PED_CROSS_OBSTRUCTED = "S8.3 obstructed"
  PED_STATIONARY_25R = "S8.4 stationary 25% right"
  PED_ALONG_25R = "S8.5 along 25% right"

@dataclass
class Scenario:
  family: Family
  subject_speed_kmh: float
  lead_speed_kmh: float = 0.0
  lead_decel_ms2: Optional[float] = None
  headway_m: Optional[float] = None
  pedestrian_speed_kmh: Optional[float] = None
  overlap: Optional[float] = None
  daylight: Optional[bool] = True
  manual_brake: bool = False
  note: str = ""

@dataclass
class Actor:
  position_x: float
  velocity: float
  length: float
  width: float

@dataclass
class SimulationParams:
  car_length_m: float = 5 
  car_width_m: float = 2
