from dataclasses import dataclass

@dataclass
class DiscretizerConfig:
  gap_min: float = 0.0
  gap_max: float = 100.0
  v_min:float = 0.0
  v_max:float = 20.0
  n_gap: int = 100
  n_v_ego: int = 41
  n_v_npc: int = 41


@dataclass
class SimulationConfig:
  dt: float = 0.1
  total_time: float = 12.0