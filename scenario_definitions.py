import enum
from dataclasses import dataclass
from typing import Optional
from enum import Enum

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
  lead_speed_kmh: float
  lead_decel_ms2: float
  headway_m: float
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

class Action(Enum):
  Nothing = 0
  Warning = 1
  SoftBrake = 2
  StrongBrake = 3


EGO_ACTION_DECEL = {
  Action.SoftBrake: -6.0,
  Action.StrongBrake: -9.0,
  Action.Warning: 0.0,
  Action.Nothing: 0.0,
}

@dataclass
class Vector2D:
  x: float
  y: float


@dataclass
class Scene:
  """Scene defines the origin frame of all the other components."""

  # The length of the drivable scene in meters
  length: float

  # The width of the drivable scene in meters
  width: float

  @classmethod
  def from_scenario_definition(cls, scenario: Scenario) -> "Scene":
    return cls(length=scenario.headway_m + 40.0, width=5.0)


@dataclass
class Rectangle:
  # The position of the rectangle center with respect to the scene frame.
  # The rectangle is axis aligned to the scene frame.
  center: Vector2D

  # Rectangle width in meters (lateral, y)
  width: float

  # Rectangle height in meters (longitudinal, x)
  height: float


@dataclass
class Vehicle:
  # The vehicle box in the scene frame.
  box: Rectangle

  # The vehicle velocity in the scene frame.
  velocity: Vector2D

  # The acceleration of the vehicle in the scene frame.
  acceleration: Vector2D

  def update(self, dt: float) -> None:
    vx0 = self.velocity.x
    x0 = self.box.center.x
    ax = self.acceleration.x

    x_new = x0 + vx0 * dt + 0.5 * ax * dt * dt
    vx_new = max(0.0, vx0 + ax * dt)

    # Things end when vx is zero
    if vx_new <= 1e-5:
        self.acceleration.x = 0.0

    self.box.center.x = x_new
    self.velocity.x = vx_new

    # y stays at 0
        

