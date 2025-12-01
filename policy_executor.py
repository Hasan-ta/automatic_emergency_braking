import numpy as np
from discretizer import Discretizer
from scenario_definitions import Action

class PolicyExecutor:
  def __init__(self, disc: Discretizer, q_path: str) -> None:
    self.Q = np.load(q_path)
    # self.disc = Discretizer(
    #     gap_bins=np.linspace(0.0, 60.0, 31),      # 30 bins (2m)
    #     rel_bins=np.linspace(-30.0, 30.0, 31),    # 30 bins (2 m/s)
    #     ego_bins=np.linspace(0.0, 40.0, 21),      # 20 bins (2 m/s)
    # )

    self.disc = disc

  def __call__(self, obs) -> Action:
    s = self.disc.obs_to_state(obs)
    return Action(int(np.argmax(self.Q[s])))