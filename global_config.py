from dataclasses import dataclass

@dataclass
class DiscretizerConfig:
  gap_min: float = 0.0
  gap_max: float = 100.0
  v_min:float = -1.0
  v_max:float = 20.0
  n_gap: int = 1000
  n_v_ego: int = 82
  n_v_npc: int = 82


@dataclass
class SimulationConfig:
  dt: float = 0.1
  total_time: float = 12.0