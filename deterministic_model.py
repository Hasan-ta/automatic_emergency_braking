from discretizer import Discretizer
from scenario_definitions import Action, EGO_ACTION_DECEL
import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import math

@dataclass
class DeterministicModelConfig:
  stopped_threshold: float = 0.0
  vehicle_length: float = 5.0
  vehicle_width: float = 2.0
  step_penalty: float = -0.05
  collision_penalty: float = -6000.0
  soft_brake_penalty: float = -10.0
  strong_brake_penalty: float = -20.0
  stop_gap_penalty_multiplier: float = 5.0

@dataclass
class State:
  x_npc_in_ego: float
  v_ego: float
  v_npc: float

@dataclass
class DeterministicModel:
  S: int
  A: int
  transition: List
  reward: np.ndarray
  terminal: np.ndarray

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

class RewardsModel:
  def __init__(self, config: DeterministicModelConfig) -> None:
    self.config = config

  def __call__(self, state: State, action: int | Action, next_state: State) -> float:
    # if(terminal_state(state, self.config)):
    #   return 0.0
    if not isinstance(action, Action):
      action = Action(action)

    # --- reward shaping (like env.step) ---
    r = self.config.step_penalty  # time penalty per step

    
    if _collided(next_state, self.config):
      r += self.config.collision_penalty
      return r

    # elif _stopped(next_state, self.config):
    #   stop_gap_penalty = -1.0 * self.config.stop_gap_penalty_multiplier * _gap(next_state, self.config)
    #   r += stop_gap_penalty

    ttc = compute_ttc(state, self.config)
    if (ttc > 6.0) and (action == Action.SoftBrake or action == Action.StrongBrake):
      r += -2000
      return r
    
    # print(f"action: {action}")
    if action == Action.SoftBrake:
      r += self.config.soft_brake_penalty * ttc
    elif action == Action.StrongBrake:
      # print(f"added strong braking penalty")
      r += self.config.strong_brake_penalty * ttc

    return r

def build_deterministic_model(
  disc: Discretizer,
  dt: float,
  config: DeterministicModelConfig
):
  """
  Precompute a deterministic, stationary MDP model for the V2V AEB problem.

  States are discretized via `disc` for obs = [x_npc, v_ego, v_npc].
  Dynamics are:
    - Ego acceleration determined by Action (EGO_ACTION_DECEL)
    - NPC acceleration = lead_decel_ms2 (from scenario)
    - Constant-acceleration integration per time step dt

  Returns:
    next_state: (S, A) int array, next_state[s, a] = s'
    reward:     (S, A) float array, reward[s, a]
    terminal:   (S,) bool array, True if state is terminal
  """
  S = disc.num_states()
  A = len(Action)
  rewards_model = RewardsModel(config)

  next_state = np.zeros((S, A), dtype=np.int32)
  reward = np.zeros((S, A), dtype=np.float64)
  terminal = np.zeros(S, dtype=bool)

  # Precompute centers for all states: shape (S,4)
  centers = disc.state_centers()

  print(f"building model for {S} states")
  for s in range(S):
    if (s+1) % 10000 == 0:
      print(f"{s+1} states done!")
    # Continuous center for this discrete state
    x_npc, v_ego, v_npc = centers[s]
    current_state = State(x_npc, v_ego, v_npc)

    # Optionally mark impossible states as terminal (e.g. ego behind 0, lead behind 0)
    # You can tighten this logic as you like:
    # if terminal_state(current_state, config):
    #   print(f"current_state: {current_state} is terminal")
    #   terminal[s] = True
    #   for a_idx in range(A):
    #     next_state[s, a_idx] = s
    #     reward[s, a_idx] = 0.0
    #   continue

    for a_idx, a_enum in enumerate(Action):
      # --- dynamics: one step of constant-acceleration integration ---

      # Ego
      ax_e = EGO_ACTION_DECEL[a_enum]
      v_e_next = max(0.0, v_ego + ax_e * dt)

      v_rel = v_npc - v_ego
      x_n_next = x_npc + v_rel * dt - 0.5 * ax_e * dt * dt
      v_n_next = v_npc

      # --- map next continuous state back to discrete ---
      sp = disc.values_to_state(x_n_next, v_e_next, v_n_next)
      next_state[s, a_idx] = sp
      state_prime = State(x_n_next, v_e_next, v_n_next)

      reward[s, a_idx] = rewards_model(current_state, a_enum, state_prime)

  return next_state, reward, terminal


def load_model(next_state_path, reward_path, terminal_path) -> DeterministicModel:
  next_state = np.load(next_state_path)
  reward = np.load(reward_path)
  terminal = np.load(terminal_path)
  S = next_state.shape[0]
  A = next_state.shape[1]

  transition_matrices = []
  rows = np.arange(S, dtype=np.int32)  # same for all actions
  data = np.ones(S, dtype=np.float64)

  for a in range(A):
    cols = next_state[:, a]
    P_a = csr_matrix((data, (rows, cols)), shape=(S, S))
    transition_matrices.append(P_a)

  return DeterministicModel(S, A, transition_matrices, reward, terminal)


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
