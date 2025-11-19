import numpy as np
from mle_mdp_planner import Discretizer

class PolicyExecutor:
  def __init__(self, q_path):
    self.Q = np.load(q_path)
    self.disc = Discretizer(
        gap_bins=np.linspace(0.0, 60.0, 31),      # 30 bins (2m)
        rel_bins=np.linspace(-30.0, 30.0, 31),    # 30 bins (2 m/s)
        ego_bins=np.linspace(0.0, 40.0, 21),      # 20 bins (2 m/s)
    )

  def __call__(self, obs):
    s = self.disc.to_state(obs)
    return int(np.argmax(self.Q[s]))