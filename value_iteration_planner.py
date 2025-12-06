from scenario_definitions import Family, Scenario
from scenario_factory import make_env, generate_training_scenarios
from dataclasses import dataclass
from deterministic_model import DeterministicModel, DeterministicModelConfig, RewardsModel, plot_reward_slice_gap_ve, plot_U_slice_gap_ve, build_stochastic_model_sparse, StochasticModelSparse, build_rewards_model
import numpy as np
from discretizer import Discretizer, GapEgoAccelDiscretizer
from global_config import DiscretizerConfig, SimulationConfig
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
# from q_learning import DQNInference

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

def plot_greedy_actions_gap_ve(
    policy: np.ndarray,
    disc,
    v_npc_fixed: float,
    a_ego_fixed: float = 0.0,
    action_labels=None,
    title: str | None = None,
):
    """
    Visualize greedy actions over (gap, v_ego) for a fixed (a_ego, v_npc).

    disc   : GapEgoAccelDiscretizer
    policy : (S,) array of greedy action indices (e.g. 0..3)
    v_npc_fixed : desired lead speed (m/s) for the slice
    a_ego_fixed : desired ego acceleration (m/s^2) for the slice
    """

    def _bin_centers(bins: np.ndarray) -> np.ndarray:
      return 0.5 * (bins[:-1] + bins[1:])


    def _find_bin_index_for_value(bins: np.ndarray, value: float) -> int:
        """Pick the bin whose center is closest to 'value'."""
        centers = _bin_centers(bins)
        idx = int(np.argmin(np.abs(centers - value)))
        return idx

    Ng, Nv_e, Na_e, Nv_n = disc.shape()
    S = disc.num_states()
    assert policy.shape[0] == S

    # --- 1. choose slice indices for v_npc and a_ego ---

    i_vn = _find_bin_index_for_value(disc.v_npc_bins, v_npc_fixed)
    i_ae = _find_bin_index_for_value(disc.a_ego_bins, a_ego_fixed)

    v_npc_centers = _bin_centers(disc.v_npc_bins)
    a_ego_centers = _bin_centers(disc.a_ego_bins)
    v_npc_actual = float(v_npc_centers[i_vn])
    a_ego_actual = float(a_ego_centers[i_ae])

    # --- 2. build 2D grid of actions over (i_gap, i_ve) ---

    action_grid = np.zeros((Ng, Nv_e), dtype=int)

    for i_g in range(Ng):
        for i_ve in range(Nv_e):
            s = disc.indices_to_state(i_g, i_ve, i_ae, i_vn)
            action_grid[i_g, i_ve] = int(policy[s])

    # --- 3. coordinates for plotting ---

    gap_centers = _bin_centers(disc.gap_bins)
    ve_centers  = _bin_centers(disc.v_ego_bins)

    extent = [gap_centers[0], gap_centers[-1],
              ve_centers[0],  ve_centers[-1]]

    # --- 4. colormap + legend ---

    n_actions = int(action_grid.max()) + 1
    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    cmap = ListedColormap(base_colors[:n_actions])

    if action_labels is None:
        # default for your AEB actions
        action_labels = ["Nothing", "Warning", "SoftBrake", "StrongBrake"][:n_actions]

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
        title = (f"Greedy action over (gap, v_ego)\n"
                 f"slice at v_npc ≈ {v_npc_actual:.2f} m/s, "
                 f"a_ego ≈ {a_ego_actual:.2f} m/s²")
    plt.title(title)

    # legend
    handles = [
        Patch(color=cmap(i), label=f"{i}: {action_labels[i]}")
        for i in range(n_actions)
    ]
    plt.legend(handles=handles, title="Action", loc="upper right")

    plt.tight_layout()
    plt.show()
  

# =============== Small demo ===============
if __name__ == "__main__":

  scenarios = generate_training_scenarios()

  sim_config = SimulationConfig()

  disc_config = DiscretizerConfig()
  disc = GapEgoAccelDiscretizer.from_ranges(
    gap_min=disc_config.gap_min,
    gap_max=disc_config.gap_max,
    v_min=disc_config.v_min,
    v_max=disc_config.v_max,      # m/s
    a_min=disc_config.a_min,
    a_max=disc_config.a_max,      # m/s
    n_gap=disc_config.n_gap,
    n_v_ego=disc_config.n_v_ego,
    n_a_ego=disc_config.n_a_ego,
    n_v_npc=disc_config.n_v_npc,
  ) 

  model_cfg = DeterministicModelConfig()

  # model = build_stochastic_model_sparse(disc, sim_config.dt, model_cfg)
  # model.save()
  model = StochasticModelSparse.load(disc.num_states(), 4)
  model.reward = build_rewards_model(disc, dt=sim_config.dt, config=model_cfg)
    
  plot_reward_slice_gap_ve(disc, model.reward, 0, 0.0, 1.0)
  plot_reward_slice_gap_ve(disc, model.reward, 2, 0.0, 1.0)
  plot_reward_slice_gap_ve(disc, model.reward, 3, 0.0, 1.0)

  plan_cfg = PlannerConfig(gamma=0.96, tol=1e-8, max_iter=10_000)
  planner = ValueIterationPlanner(model, plan_cfg)
  planner.plan()
  policy = planner.get_policy()
  # executor = DQNInference("aeb_dqn_qnet.pt")
  # policy = executor.get_policy(disc)
  plot_greedy_actions_gap_ve(policy, disc, 0.0, 0.0)
  plot_greedy_actions_gap_ve(policy, disc, 5.6, 0.0)
  plot_greedy_actions_gap_ve(policy, disc, 8.0, 0.0)
  plot_greedy_actions_gap_ve(policy, disc, 12.0, 0.0)
  plot_greedy_actions_gap_ve(policy, disc, 16.0, 0.0)
  planner.save_plan("q_deterministic_planner.npy")
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
