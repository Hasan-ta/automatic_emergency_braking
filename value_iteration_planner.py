from scenario_definitions import Family, Scenario
from scenario_factory import make_env, generate_training_scenarios
from dataclasses import dataclass
from deterministic_model import DeterministicModel, DeterministicModelConfig, load_model, build_deterministic_model, RewardsModel, plot_reward_slice_gap_ve, plot_U_slice_gap_ve, build_transition_model, build_rewards_model, build_stochastic_model_sparse, StochasticModelSparse
import numpy as np
from discretizer import Discretizer
from global_config import DiscretizerConfig, SimulationConfig
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from q_learning import DQNInference

@dataclass
class PlannerConfig:
  gamma: float = 0.995
  tol: float = 1e-8
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
  
  def get_policy(self) -> np.ndarray:
    policy = np.empty((self.model.S,), np.int32)

    for s in range(self.model.S):
      policy[s] = self.act(s)

    return policy

  def save_plan(self, path: str) -> None:
    np.save(path, self.Q)

def plot_greedy_actions(policy: np.ndarray, disc: Discretizer, v_npc_fixed: float, action_labels=None, title: str | None = None,) -> None:
  """
  Visualize greedy action over (gap, v_ego) for a fixed v_npc.

  disc   : GapDiscretizer with shape() -> (Ng, Nv_e, Nv_n)
  policy : (S,) array of action indices from value iteration
  v_npc_fixed : desired lead speed (m/s) to slice at
  """

  def get_bin_centers(bins: np.ndarray) -> np.ndarray:
    return 0.5 * (bins[:-1] + bins[1:])

  def find_vnpc_index_for_value(disc, v_npc_fixed: float) -> int:
    """Pick the v_npc bin index whose center is closest to v_npc_fixed."""
    v_n_centers = get_bin_centers(disc.v_npc_bins)
    i_vn = int(np.argmin(np.abs(v_n_centers - v_npc_fixed)))
    return i_vn

  Ng, Nv_e, Nv_n = disc.shape()
  S = disc.num_states()
  assert policy.shape[0] == S

  # 1. choose v_npc slice
  i_vn = find_vnpc_index_for_value(disc, v_npc_fixed)
  v_n_centers = get_bin_centers(disc.v_npc_bins)
  v_npc_actual = v_n_centers[i_vn]

  # 2. build 2D grid of greedy actions over (i_g, i_ve)
  action_grid = np.zeros((Ng, Nv_e), dtype=int)

  for i_g in range(Ng):
    for i_ve in range(Nv_e):
      s = disc.indices_to_state(i_g, i_ve, i_vn)
      action_grid[i_g, i_ve] = policy[s]

  # 3. coordinates for plotting
  gap_centers = get_bin_centers(disc.gap_bins)
  ve_centers  = get_bin_centers(disc.v_ego_bins)

  extent = [gap_centers[0], gap_centers[-1], ve_centers[0], ve_centers[-1]]

  # 4. colormap for actions (tweak as you like)
  # Number of actions inferred from max index
  n_actions = int(action_grid.max()) + 1
  # simple distinct colors; you can customize
  base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
  cmap = ListedColormap(base_colors[:n_actions])

  plt.figure(figsize=(7, 4))
  im = plt.imshow(
    action_grid.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    cmap=cmap,
    interpolation="nearest",
  )
  plt.xlabel("gap [m]")
  plt.ylabel("v_ego [m/s]")

  if title is None:
    title = f"Greedy action over (gap, v_ego) at v_npc ≈ {v_npc_actual:.2f} m/s"
  plt.title(title)

  # 5. custom legend for actions
  if action_labels is None:
    # default labels for your AEB actions
    action_labels = ["Nothing", "Warning", "SoftBrake", "StrongBrake"][:n_actions]

  # build fake handles for legend
  from matplotlib.patches import Patch
  handles = [
    Patch(color=cmap(i), label=f"{i}: {action_labels[i]}")
    for i in range(n_actions)
  ]
  plt.legend(handles=handles, title="Action", loc="upper right")

  plt.tight_layout()
  plt.show()
  

# =============== Small demo ===============
if __name__ == "__main__":
  # signal.signal(signal.SIGINT, signal_handler)

  scenarios = generate_training_scenarios()

  sim_config = SimulationConfig()

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
  # next_state = build_deterministic_model(disc, dt=0.1, config=model_cfg)
  
  # np.save("model_next_state.npy", next_state)
  # np.save("model_reward.npy", reward)
  # np.save("model_terminal.npy", terminal)
  # model = load_model('model_next_state.npy', 'model_reward.npy', 'model_terminal.npy')
  # transitions = build_transition_model(disc, dt=sim_config.dt)

  # model = build_stochastic_model_sparse(disc, sim_config.dt, model_cfg)
  # model.save()
  model = StochasticModelSparse.load(disc.num_states(), 4)
    
  # plot_reward_slice_gap_ve(disc, model.reward, 2, 0.0, 1.0)

  plan_cfg = PlannerConfig(gamma=0.96, tol=1e-8, max_iter=10_000)
  planner = ValueIterationPlanner(model, plan_cfg)
  planner.plan()
  policy = planner.get_policy()
  # executor = DQNInference("aeb_dqn_qnet.pt")
  # policy = executor.get_policy(disc)
  plot_greedy_actions(policy, disc, 0.0)
  plot_greedy_actions(policy, disc, 4.0)
  plot_greedy_actions(policy, disc, 8.0)
  plot_greedy_actions(policy, disc, 12.0)
  plot_greedy_actions(policy, disc, 16.0)
  # planner.save_plan("q_deterministic_planner.npy")
  # plot_U_slice_gap_ve(disc, planner.U, 4.0, 1.0)

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
    env = make_env(sc, rewards_model, dt=sim_config.dt, max_time=sim_config.total_time)
    mean_ret, std_ret = evaluate_policy(env, disc, planner, episodes=10)
    print(f"Evaluation: mean return={mean_ret:.2f} ± {std_ret:.2f}")
    env.close()
