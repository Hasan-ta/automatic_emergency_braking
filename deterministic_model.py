from discretizer import Discretizer
from scenario_definitions import Action, EGO_ACTION_DECEL
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import math

@dataclass
class DeterministicModelConfig:
  stopped_threshold: float = 0.0
  vehicle_length: float = 5.0
  vehicle_width: float = 2.0
  step_penalty: float = -0.1
  collision_penalty: float = -6000.0
  soft_brake_penalty: float = -10.0
  strong_brake_penalty: float = -20.0
  stop_gap_penalty_multiplier: float = 5.0
  num_mc_samples: int = 512

@dataclass
class State:
  x_npc_in_ego: float
  v_ego: float
  a_ego: float
  v_npc: float

@dataclass
class DeterministicModel:
  S: int
  A: int
  transition: List
  reward: np.ndarray
  terminal: np.ndarray

@dataclass
class StochasticModelSparse:
  S: int
  A: int
  transition: List
  reward: np.ndarray

  def save(self):
    for a in range(self.A):
      save_npz(f"sparse_transitions_action_{a}.npz", self.transition[a])
    np.save("stochastic_dense_rewards.npy", self.reward)

  @classmethod
  def load(cls, S: int, A: int) -> 'StochasticModelSparse':
    transition = [load_npz(f"sparse_transitions_action_{a}.npz") for a in range(A)]
    reward = np.load("stochastic_dense_rewards.npy")

    return cls(S, A, transition, reward)

# Small helper: 1D overlap check for rectangles centered at x with lengths L
def _overlap_1d(len1: float, center2: float, len2: float) -> bool:
  min1 = -0.5 * len1
  max1 = 0.5 * len1
  min2 = center2 - 0.5 * len2
  max2 = center2 + 0.5 * len2
  return (min1 <= max2) and (min2 <= max1)

def _stopped(state: State, config: DeterministicModelConfig) ->bool:
  return state.v_ego <= config.stopped_threshold

def _collided(state: State, config: DeterministicModelConfig) -> bool:
  return _overlap_1d(config.vehicle_length, state.x_npc_in_ego, config.vehicle_length) and (not _stopped(state, config))

def _gap(state: State, config: DeterministicModelConfig) -> float:
  return max(0.0, state.x_npc_in_ego - config.vehicle_length)

def compute_ttc(state: State, config: DeterministicModelConfig) -> float:
  v_rel = state.v_npc - state.v_ego
  if(v_rel >= 0.0):
    return 1e6
  
  return _gap(state, config) / -v_rel
  

def terminal_state(state: State, config: DeterministicModelConfig) -> bool:
  if state.x_npc_in_ego < 0.0 or _stopped(state, config) or _collided(state, config):
    return True
  
  return False

import numpy as np

import numpy as np

def _sample_from_cell_4d(disc, s: int, n_samples: int) -> np.ndarray:
  """
  Sample continuous states uniformly from inside cell s for the
  GapEgoAccelDiscretizer (gap, v_ego, a_ego, v_npc).

  disc     : GapEgoAccelDiscretizer
  s        : discrete state index
  n_samples: number of samples to draw

  Returns:
      samples: (n_samples, 4) array with columns
                [gap, v_ego, a_ego, v_npc]
  """
  # Decode the 4D indices for this state
  i_g, i_ve, i_ae, i_vn = disc.state_to_indices(s)

  # Get bin bounds in each dimension
  g_low,  g_high  = disc.gap_bins[i_g],     disc.gap_bins[i_g + 1]
  ve_low, ve_high = disc.v_ego_bins[i_ve],  disc.v_ego_bins[i_ve + 1]
  ae_low, ae_high = disc.a_ego_bins[i_ae],  disc.a_ego_bins[i_ae + 1]
  vn_low, vn_high = disc.v_npc_bins[i_vn],  disc.v_npc_bins[i_vn + 1]

  # Sample uniformly in each dimension
  gaps   = np.random.uniform(g_low,  g_high,  size=n_samples)
  v_ego  = np.random.uniform(ve_low, ve_high, size=n_samples)
  a_ego  = np.random.uniform(ae_low, ae_high, size=n_samples)
  v_npc  = np.random.uniform(vn_low, vn_high, size=n_samples)

  samples = np.stack([gaps, v_ego, a_ego, v_npc], axis=1).astype(np.float64)
  return samples



class RewardsModel:
  def __init__(self, config: DeterministicModelConfig) -> None:
    self.config = config

  def __call__(self, state: State, action: int | Action, next_state: State) -> float:
    if not isinstance(action, Action):
      action = Action(action)

    # --- reward shaping (like env.step) ---
    r = self.config.step_penalty  # time penalty per step
    
    if _collided(next_state, self.config):
      r += self.config.collision_penalty
      return r

    acc_diff = abs(state.a_ego - next_state.a_ego)
    r -= acc_diff*2

    ttc = compute_ttc(state, self.config)
    if (ttc > 6.0) and (action == Action.SoftBrake or action == Action.StrongBrake):
      r += -2000
      return r
    
    if action == Action.SoftBrake:
      r += self.config.soft_brake_penalty * ttc
    elif action == Action.StrongBrake:
      r += self.config.strong_brake_penalty * ttc

    return r
  
def _step_continuous_gap_model(
    states: np.ndarray,  # shape (N,3) = [gap, v_ego, a_ego, v_npc]
    action: Action,
    dt: float,
    process_noise_std: np.ndarray | None = None,
) -> np.ndarray:
  """
  Continuous dynamics for one step, possibly with additive Gaussian process noise.
  """
  gap = states[:, 0]
  v_e = states[:, 1]
  a_e = states[:, 2]
  v_n = states[:, 3]

  a_e_action = EGO_ACTION_DECEL[action]

  # simple Euler-ish update; you can use your exact model
  # gap_{t+1} = gap_t + (v_n - v_e) * dt
  gap_next = gap + (v_n - v_e) * dt - 0.5*a_e*dt*dt
  a_e_next = np.full_like(a_e, a_e_action)
  v_e_next = np.maximum(0.0, v_e + a_e_next * dt)
  v_n_next = np.maximum(0.0, v_n)

  x_next = np.stack([gap_next, v_e_next, a_e_next, v_n_next], axis=1)

  if process_noise_std is not None:
      noise = np.random.randn(*x_next.shape) * process_noise_std[None, :]
      x_next = x_next + noise

  return x_next.astype(np.float64)

# def mc_transition_row_sparse(
#     disc: Discretizer,
#     s: int,
#     action: Action,
#     dt: float,
#     n_samples: int = 512,
#     process_noise_std: np.ndarray | None = None,
# ):
#   """
#   Returns the nonzero entries of the Monte Carlo transition row:

#   - next_states: 1D array of column indices
#   - probs      : 1D array of probabilities, same length
#   """
#   S = disc.num_states()
#   samples = _sample_from_cell_3d(disc, s, n_samples)
#   x_next = _step_continuous_gap_model(
#     samples,
#     action=action,
#     dt=dt,
#     process_noise_std=process_noise_std,
#   )

#   next_states = np.array([disc.obs_to_state(x_next[i]) for i in range(n_samples)], dtype=np.int32)
#   counts = np.bincount(next_states, minlength=S).astype(np.float64)
#   nz = np.nonzero(counts)[0]

#   if len(nz) == 0:
#     return np.array([s], dtype=np.int32), np.array([1.0], dtype=np.float64)

#   probs = counts[nz] / counts.sum()
#   return nz.astype(np.int32), probs.astype(np.float64)

# def build_transition_csr_for_action(disc:Discretizer, dt: float, action: Action, S: int, n_samples=512):
#   row_idx = []
#   col_idx = []
#   data = []

#   print(f"building T(s' | s, {action}) for {S} states")
#   for s in range(S):
#     if (s+1) % 1000 == 0:
#       print(f"{s+1} states done!")
#     cols, probs = mc_transition_row_sparse(
#       disc,
#       s=s,
#       action=action,
#       dt=dt,
#       n_samples=n_samples,
#     )
#     row_idx.append(np.full(len(cols), s, dtype=np.int32))
#     col_idx.append(cols)
#     data.append(probs)

#   row_idx = np.concatenate(row_idx)
#   col_idx = np.concatenate(col_idx)
#   data = np.concatenate(data)

#   P_a = csr_matrix((data, (row_idx, col_idx)), shape=(S, S))
#   return P_a

# def build_transition_model(disc:Discretizer, dt: float) -> List:
#   S = disc.num_states()
#   transition_matrices = []
#   for a_enum in Action:
#     P_a = build_transition_csr_for_action(
#       disc=disc,
#       dt=dt,
#       action=a_enum,
#       S=S,
#       n_samples=512,
#     )
#     transition_matrices.append(P_a)

#   return transition_matrices

def build_rewards_model(disc: Discretizer, dt: float, config: DeterministicModelConfig):

  S = disc.num_states()
  A = len(Action)
  rewards_model = RewardsModel(config)

  reward = np.zeros((S, A), dtype=np.float64)

  # Precompute centers for all states: shape (S,4)
  centers = disc.state_centers()

  print(f"building rewards model for {S} states")
  for s in range(S):
    if (s+1) % 10000 == 0:
      print(f"{s+1} states done!")
    # Continuous center for this discrete state
    x_npc, v_ego, a_ego, v_npc = centers[s]
    current_state = State(x_npc, v_ego, a_ego, v_npc)

    for a_idx, a_enum in enumerate(Action):
      # --- dynamics: one step of constant-acceleration integration ---

      # Ego
      ax_e = EGO_ACTION_DECEL[a_enum]
      v_e_next = max(0.0, v_ego + ax_e * dt)
      a_e_next = ax_e

      v_rel = v_npc - v_ego
      x_n_next = x_npc + v_rel * dt - 0.5 * ax_e * dt * dt
      v_n_next = v_npc


      # --- map next continuous state back to discrete ---
      state_prime = State(x_n_next, v_e_next, a_e_next, v_n_next)

      reward[s, a_idx] = rewards_model(current_state, a_enum, state_prime)

  return reward


def mc_transition_row_sa_sparse(
    disc:Discretizer,
    config: DeterministicModelConfig,
    s: int,
    action: Action,
    dt: float,
    n_samples: int = 512,
    process_noise_std: np.ndarray | None = None,
):
    """
    Monte Carlo estimate for a single (s, a):

      - T(s'|s,a) -> sparse row (cols, probs)
      - r(s,a)    -> expected immediate reward (scalar)

    Returns:
      cols   : 1D array of next-state indices s'
      probs  : 1D array of probabilities, same length
      r_mean : scalar, mean reward over samples
    """
    S = disc.num_states()
    rewards_model = RewardsModel(config)

    # 1) sample continuous states in cell s
    samples = _sample_from_cell_4d(disc, s, n_samples)  # (N,3) = [gap, v_ego, v_npc]

    # 2) propagate them
    x_next = _step_continuous_gap_model(
        samples,
        action=action,
        dt=dt,
        process_noise_std=process_noise_std,
    )

    counts = np.zeros(S, dtype=np.int64)
    rewards = np.zeros(n_samples, dtype=np.float64)

    # 3) discretize next states + compute reward per sample
    for i in range(n_samples):
        x = samples[i]
        xn = x_next[i]

        sp = disc.obs_to_state(xn)

        r = rewards_model(State(x[0], x[1], x[2], x[3]), action, State(xn[0], xn[1], xn[2], xn[3]))

        counts[sp] += 1
        rewards[i] = r

    total = counts.sum()
    if total == 0:
        # degenerate: no valid transitions, fallback to self-loop with 0 reward
        cols = np.array([s], dtype=np.int32)
        probs = np.array([1.0], dtype=np.float64)
        r_mean = 0.0
        return cols, probs, r_mean

    nz = np.nonzero(counts)[0]
    probs = counts[nz].astype(np.float64) / total
    cols = nz.astype(np.int32)

    # expected reward r(s,a) is just the mean over all samples
    r_mean = float(rewards.mean())

    return cols, probs, r_mean


def build_sparse_P_and_R_for_action(
    disc: Discretizer,
    config: DeterministicModelConfig,
    dt: float,
    action: Action,
    S: int,
    n_samples: int = 512,
):
    """
    For a single action, build:

      P_a: csr_matrix (S,S) with T(s'|s,a)
      R_a: np.ndarray (S,)   with r(s,a)
    """
    row_idx = []
    col_idx = []
    prob_data = []
    R_a = np.zeros(S, dtype=np.float64)

    print(f"building T(s' | s, {action}) for {S} states")
    for s in range(S):
        if (s+1) % 1000 == 0:
          print(f"{s+1} states done!")
        cols, probs, r_mean = mc_transition_row_sa_sparse(
            disc=disc,
            config=config,
            s=s,
            action=action,
            dt=dt,
            n_samples=n_samples,
        )

        row_idx.append(np.full(len(cols), s, dtype=np.int32))
        col_idx.append(cols)
        prob_data.append(probs)
        R_a[s] = r_mean

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    prob_data = np.concatenate(prob_data)

    P_a = csr_matrix((prob_data, (row_idx, col_idx)), shape=(S, S))
    return P_a, R_a

def build_stochastic_model_sparse(
  disc: Discretizer,
  dt: float,
  config: DeterministicModelConfig) -> StochasticModelSparse:

  S = disc.num_states()
  A = 4
  transition_mats = []
  R_sa = np.empty((S, A), dtype=np.float64)

  for a_enum in Action:
    P_a, R_a = build_sparse_P_and_R_for_action(
      disc=disc,
      config=config,
      dt=dt,
      action=a_enum,
      S=S,
      n_samples=512,
    )
    transition_mats.append(P_a)
    # print(R_sa.shape)
    R_sa[:, a_enum.value] = R_a

  return StochasticModelSparse(S, A, transition_mats, R_sa)


# def build_deterministic_model(
#   disc: Discretizer,
#   dt: float,
#   config: DeterministicModelConfig
# ):
#   """
#   Precompute a deterministic, stationary MDP model for the V2V AEB problem.

#   States are discretized via `disc` for obs = [x_npc, v_ego, v_npc].
#   Dynamics are:
#     - Ego acceleration determined by Action (EGO_ACTION_DECEL)
#     - NPC acceleration = lead_decel_ms2 (from scenario)
#     - Constant-acceleration integration per time step dt

#   Returns:
#     next_state: (S, A) int array, next_state[s, a] = s'
#     reward:     (S, A) float array, reward[s, a]
#     terminal:   (S,) bool array, True if state is terminal
#   """
#   S = disc.num_states()
#   A = len(Action)
#   rewards_model = RewardsModel(config)

#   next_state = np.zeros((S, A), dtype=np.int32)
#   reward = np.zeros((S, A), dtype=np.float64)
#   terminal = np.zeros(S, dtype=bool)

#   # Precompute centers for all states: shape (S,4)
#   centers = disc.state_centers()

#   print(f"building model for {S} states")
#   for s in range(S):
#     if (s+1) % 10000 == 0:
#       print(f"{s+1} states done!")
#     # Continuous center for this discrete state
#     x_npc, v_ego, v_npc = centers[s]
#     current_state = State(x_npc, v_ego, v_npc)

#     # Optionally mark impossible states as terminal (e.g. ego behind 0, lead behind 0)
#     # You can tighten this logic as you like:
#     # if terminal_state(current_state, config):
#     #   print(f"current_state: {current_state} is terminal")
#     #   terminal[s] = True
#     #   for a_idx in range(A):
#     #     next_state[s, a_idx] = s
#     #     reward[s, a_idx] = 0.0
#     #   continue

#     for a_idx, a_enum in enumerate(Action):
#       # --- dynamics: one step of constant-acceleration integration ---

#       # Ego
#       ax_e = EGO_ACTION_DECEL[a_enum]
#       v_e_next = max(0.0, v_ego + ax_e * dt)

#       v_rel = v_npc - v_ego
#       x_n_next = x_npc + v_rel * dt - 0.5 * ax_e * dt * dt
#       v_n_next = v_npc

#       # --- map next continuous state back to discrete ---
#       sp = disc.values_to_state(x_n_next, v_e_next, v_n_next)
#       next_state[s, a_idx] = sp
#       state_prime = State(x_n_next, v_e_next, v_n_next)

#       reward[s, a_idx] = rewards_model(current_state, a_enum, state_prime)

#   return next_state, reward, terminal


# def load_model(next_state_path, reward_path, terminal_path) -> DeterministicModel:
#   next_state = np.load(next_state_path)
#   reward = np.load(reward_path)
#   terminal = np.load(terminal_path)
#   S = next_state.shape[0]
#   A = next_state.shape[1]

#   transition_matrices = []
#   rows = np.arange(S, dtype=np.int32)  # same for all actions
#   data = np.ones(S, dtype=np.float64)

#   for a in range(A):
#     cols = next_state[:, a]
#     P_a = csr_matrix((data, (rows, cols)), shape=(S, S))
#     transition_matrices.append(P_a)

#   return DeterministicModel(S, A, transition_matrices, reward, terminal)


def plot_reward_slice_gap_ve(
    disc: Discretizer,
    reward: np.ndarray,
    action_idx: int,
    v_npc_fixed: float,
    tol_v=0.5,
):
  centers = disc.state_centers()  # (S,4): [gap, v_ego, v_npc, a_npc]
  gaps = centers[:, 0]
  v_e = centers[:, 1]
  v_n = centers[:, 2]

  # Select states with v_npc ≈ v_npc_fixed and a_npc ≈ a_npc_fixed
  mask = (np.abs(v_n - v_npc_fixed) <= tol_v)
  if not np.any(mask):
    print("No states in slice; relax tolerances.")
    return

  g_slice = gaps[mask]
  v_e_slice = v_e[mask]

  r_slice = reward[mask, action_idx]

  # Build grid for plotting
  unique_g = np.unique(g_slice)
  unique_ve = np.unique(v_e_slice)
  G, VE = np.meshgrid(unique_g, unique_ve, indexing="ij")
  R = np.full_like(G, np.nan, dtype=np.float64)

  # Fill R
  for g_val, ve_val, r_val in zip(g_slice, v_e_slice, r_slice):
    i = np.where(unique_g == g_val)[0][0]
    j = np.where(unique_ve == ve_val)[0][0]
    R[i, j] = r_val

  plt.figure(figsize=(6, 4))
  im = plt.imshow(
    R.T,
    origin="lower",
    aspect="auto",
    extent=[unique_g[0], unique_g[-1], unique_ve[0], unique_ve[-1]],
  )
  plt.colorbar(im, label="reward")
  plt.xlabel("gap [m]")
  plt.ylabel("v_ego [m/s]")
  plt.xticks(np.linspace(0, 100, 20))
  plt.title(f"Reward slice, action={action_idx}, v_npc≈{v_npc_fixed}")
  plt.tight_layout()
  plt.show()

def plot_U_slice_gap_ve(
    disc,
    U: np.ndarray,
    v_npc_fixed: float,
    tol_v: float = 0.5,
    title: str | None = None,
):
    """
    Visualize a slice of the utility function U over (gap, v_ego),
    for states with v_npc ≈ v_npc_fixed and a_npc ≈ a_npc_fixed.

    disc: GapAccelDiscretizer
    U:    (S,) utility array from value iteration
    """
    centers = disc.state_centers()  # (S,4): [gap, v_ego, v_npc, a_npc]
    gaps = centers[:, 0]
    v_e  = centers[:, 1]
    v_n  = centers[:, 2]

    # Select states near the desired (v_npc, a_npc)
    mask = (np.abs(v_n - v_npc_fixed) <= tol_v)
    if not np.any(mask):
        print("No states in slice; relax tol_v/tol_a.")
        return

    g_slice = gaps[mask]
    v_e_slice = v_e[mask]
    U_slice = U[mask]

    # Unique sorted coordinates for grid
    unique_g = np.unique(g_slice)
    unique_ve = np.unique(v_e_slice)

    G, VE = np.meshgrid(unique_g, unique_ve, indexing="ij")
    U_grid = np.full_like(G, np.nan, dtype=np.float64)

    # Fill grid
    for gg, vv, u in zip(g_slice, v_e_slice, U_slice):
        i = np.where(unique_g == gg)[0][0]
        j = np.where(unique_ve == vv)[0][0]
        U_grid[i, j] = u

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        U_grid.T,
        origin="lower",
        aspect="auto",
        extent=[unique_g[0], unique_g[-1], unique_ve[0], unique_ve[-1]],
    )
    plt.colorbar(im, label="U(gap, v_ego | v_npc, a_npc)")
    plt.xlabel("gap [m]")
    plt.ylabel("v_ego [m/s]")
    if title is None:
        title = f"Utility slice, v_npc≈{v_npc_fixed}"
    plt.title(title)
    plt.tight_layout()
    plt.show()
