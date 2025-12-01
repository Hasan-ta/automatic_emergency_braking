from scenario_definitions import Family, Scenario
from scenario_factory import make_env
from dataclasses import dataclass
from deterministic_model import DeterministicModel, DeterministicModelConfig, load_model, build_deterministic_model, RewardsModel, plot_reward_slice_gap_ve, plot_U_slice_gap_ve
import numpy as np
from discretizer import Discretizer
from global_config import DiscretizerConfig

@dataclass
class PlannerConfig:
  gamma: float = 0.995
  tol: float = 1e-6
  max_iter: int = 10_000

class ValueIterationPlanner:
  def __init__(self, model: DeterministicModel, p_cfg: PlannerConfig):
    self.model = model
    self.cfg = p_cfg
    S, A = model.S, model.A
    self.U = np.zeros(S, dtype=np.float64)
    self.Q = np.zeros((S, A), dtype=np.float64)

  def plan(self):
    S, A = self.model.S, self.model.A
    gamma, tol, max_iter = self.cfg.gamma, self.cfg.tol, self.cfg.max_iter

    for it in range(max_iter):
      U_old = self.U.copy()
      # Bellman backup
      for a in range(A):
          p = self.model.transition[a]
          r = self.model.reward[:, a]
          self.Q[:, a] = r + gamma * (p @ U_old)

      # print(self.Q)
      self.U = self.Q.max(axis=1)
      bellman_residual = np.max(np.abs(self.U - U_old))
      if it%50 == 0:
          print(f"Value Iteration [{it}], residual: {bellman_residual}")
      if bellman_residual < tol:
          break

  def act(self, s: int) -> int:
    return int(np.argmax(self.Q[s]))
  
  def save_plan(self, path: str) -> None:
    np.save(path, self.Q)
  

# =============== Small demo ===============
if __name__ == "__main__":
  # signal.signal(signal.SIGINT, signal_handler)

  scenarios = []
  # sc = next(s for s in generate_fmvss127()
  #           if s.family == Family.V2V_DECELERATING and s.subject_speed_kmh == 80 and s.headway_m == 20)
  # scenarios.append(sc)
  # scenarios.append(Scenario(Family.V2V_STATIONARY, 50, 0, manual_brake=False, headway_m=40, note="S7.3 no manual"))

  # for v in list(range(10, 81, 10)):
  #     scenarios.append(Scenario(Family.V2V_STATIONARY, v, 0, manual_brake=False, headway_m=6*0.277778*v, note="S7.3 no manual"))
  # for v in [40, 50, 60, 70, 80]:
  #     scenarios.append(Scenario(Family.V2V_SLOWER_MOVING, v, 20, manual_brake=False, headway_m=6*(v-20)*277778, note="S7.4 no manual"))
  # for v in [50, 80]:
  #     for hw in [12, 20, 30, 40]:
  #         for decel_g in [0.3, 0.4, 0.5]:
  #             for manual in [False]:
  #                 scenarios.append(Scenario(Family.V2V_DECELERATING, v, v, lead_decel_ms2=decel_g*9.80665,
  #                                headway_m=hw, manual_brake=manual, note="S7.5"))

  scenarios.append(Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=10, lead_speed_kmh=0, lead_decel_ms2=None, headway_m=16.66668, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual'))
  scenarios.append(Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=50, lead_speed_kmh=0, lead_decel_ms2=0.0, headway_m=84, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual'))
  scenarios.append(Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=40, lead_speed_kmh=20, lead_decel_ms2=None, headway_m=16.66668, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual'))
  scenarios.append(Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=600, lead_speed_kmh=20, lead_decel_ms2=0.0, headway_m=84, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual'))

  # Discretization grids (tune to your dynamics):
  # Gap up to 60m, rel_speed -30..30 m/s, ego 0..40 m/s
  disc_config = DiscretizerConfig()
  disc = Discretizer.from_ranges(
    gap_min=disc_config.gap_min,
    gap_max=disc_config.gap_max,
    v_min=disc_config.v_min,
    v_max=disc_config.v_max,      # m/s
    n_gap=disc_config.n_gap,
    n_v_ego=disc_config.n_v_ego,
    n_v_npc=disc_config.n_v_npc,
  ) 

  model_cfg = DeterministicModelConfig()
  next_state, reward, terminal = build_deterministic_model(disc, dt=0.1, config=model_cfg)
  np.save("model_next_state.npy", next_state)
  np.save("model_reward.npy", reward)
  np.save("model_terminal.npy", terminal)
  model = load_model('model_next_state.npy', 'model_reward.npy', 'model_terminal.npy')
  plot_reward_slice_gap_ve(disc, model.reward, 2, 0.0, 1.0)

  plan_cfg = PlannerConfig(gamma=0.99, tol=1e-6, max_iter=10_000)
  planner = ValueIterationPlanner(model, plan_cfg)
  planner.plan()
  planner.save_plan("q_deterministic_planner.npy")
  plot_U_slice_gap_ve(disc, planner.U, 0.0, 1.0)

  # Evaluate the learned policy without exploration
  def evaluate_policy(env, disc: Discretizer, planner: ValueIterationPlanner, episodes=10, seed=123):
    ret = []
    for ep in range(episodes):
      obs, _ = env.reset(seed=seed+ep)
      s = disc.obs_to_state(obs)
      done = False
      R = 0.0
      while not done:
        a = planner.act(s)
        obs, r, done, trunc, _ = env.step(a)
        R += r
        s = disc.obs_to_state(obs)
        if trunc:
          break
      ret.append(R)
    return np.mean(ret), np.std(ret)
  
  for sc in scenarios:
    print(f"scenario: {sc}:")
    rewards_model = RewardsModel(model_cfg)
    env = make_env(sc, rewards_model, dt=0.1)
    mean_ret, std_ret = evaluate_policy(env, disc, planner, episodes=10)
    print(f"Evaluation: mean return={mean_ret:.2f} ± {std_ret:.2f}")
    env.close()
