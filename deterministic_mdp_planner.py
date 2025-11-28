from discretizer import Discretizer
from scenario_definitions import Action, EGO_ACTION_DECEL
import numpy as np
from scipy.sparse import csr_matrix

def build_deterministic_transition_model(
  disc: Discretizer,
  scene_length: float,
  lead_decel_ms2: float,
  dt: float,
  ego_length: float = 5.0,
  npc_length: float = 5.0,
  lane_width: float = 3.5,
  collision_penalty: float = -1000.0,
  step_penalty: float = -0.05,
  soft_brake_penalty: float = -0.1,
  strong_brake_penalty: float = -0.2,
):
  """
  Precompute a deterministic, stationary MDP model for the V2V AEB problem.

  States are discretized via `disc` for obs = [x_ego, x_npc, v_ego, v_npc].
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

  next_state = np.zeros((S, A), dtype=np.int32)
  # reward = np.zeros((S, A), dtype=np.float32)
  # terminal = np.zeros(S, dtype=bool)

  # Precompute centers for all states: shape (S,4)
  centers = disc.state_centers()

  # Small helper: 1D overlap check for rectangles centered at x with lengths L
  # def _overlap_1d(center1: float, len1: float, center2: float, len2: float) -> bool:
  #   min1 = center1 - 0.5 * len1
  #   max1 = center1 + 0.5 * len1
  #   min2 = center2 - 0.5 * len2
  #   max2 = center2 + 0.5 * len2
  #   return (min1 <= max2) and (min2 <= max1)

  for s in range(S):
    # Continuous center for this discrete state
    x_ego, x_npc, v_ego, v_npc = centers[s]

    # Optionally mark impossible states as terminal (e.g. ego behind 0, lead behind 0)
    # You can tighten this logic as you like:
    if x_ego < 0.0 or x_ego > scene_length or x_npc < 0.0 or x_npc > scene_length:
      # terminal[s] = True
      # If terminal, any action keeps you in a dummy self-loop with zero reward
      for a_idx in range(A):
        next_state[s, a_idx] = s
        # reward[s, a_idx] = 0.0
      continue

    for a_idx, a_enum in enumerate(Action):
      # --- dynamics: one step of constant-acceleration integration ---

      # Ego
      ax_e = EGO_ACTION_DECEL[a_enum]
      x_e_next = x_ego + v_ego * dt + 0.5 * ax_e * dt * dt
      v_e_next = max(0.0, v_ego + ax_e * dt)

      # NPC (lead)
      ax_n = lead_decel_ms2
      x_n_next = x_npc + v_npc * dt + 0.5 * ax_n * dt * dt
      v_n_next = max(0.0, v_npc + ax_n * dt)

      # --- map next continuous state back to discrete ---
      sp = disc.values_to_state(x_e_next, x_n_next, v_e_next, v_n_next)
      next_state[s, a_idx] = sp

      # # --- collision and out-of-scene checks at NEXT state ---
      # collided = _overlap_1d(x_e_next, ego_length, x_n_next, npc_length)
      # out_of_scene = (
      #   (x_e_next < 0.0) or (x_e_next > scene_length)
      #   or (x_n_next < 0.0) or (x_n_next > scene_length)
      # )

      # # --- reward shaping (like env.step) ---
      # r = step_penalty  # time penalty per step

      # if collided:
      #   r += collision_penalty

      # if a_enum == Action.SoftBrake:
      #   r += soft_brake_penalty
      # elif a_enum == Action.StrongBrake:
      #   r += strong_brake_penalty

      # reward[s, a_idx] = r

      # # If we want to treat collision/out-of-scene as terminal, we can
      # if collided or out_of_scene:
      #   # Mark NEXT state as terminal; this is a choice:
      #   terminal[sp] = True

  # return next_state, reward, terminal
  transition_matrices = []
  rows = np.arange(S, dtype=np.int32)  # same for all actions
  data = np.ones(S, dtype=np.float64)

  for a in range(A):
    cols = next_state[:, a]
    P_a = csr_matrix((data, (rows, cols)), shape=(S, S))
    transition_matrices.append(P_a)

  return next_state, transition_matrices



def build_deterministic_rewards_model()