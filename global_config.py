from dataclasses import dataclass

@dataclass
class DiscretizerConfig:
  gap_min: float = 0.0
  gap_max: float = 100.0
  v_min:float = 0.0
  v_max:float = 20.0
  a_min:float = -9.0
  a_max:float = 0.0
  n_gap: int = 50
  n_v_ego: int = 21
  n_a_ego: int = 10
  n_v_npc: int = 21


@dataclass
class SimulationConfig:
  dt: float = 0.1
  total_time: float = 12.0