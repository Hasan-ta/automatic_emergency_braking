from scenario_factory import make_env, generate_training_scenarios
from discretizer import Discretizer, GapEgoAccelDiscretizer
from global_config import DiscretizerConfig, SimulationConfig
from policy_executor import PolicyExecutor, NaivePolicyExecutor
from deterministic_model import DeterministicModelConfig, RewardsModel
from q_learning import DQNInference
from noisy_obs_wrapper import NoisyObsWrapper
from metrics import DiscomfortTracker
import numpy as np
import math

from dataclasses import dataclass

def evaluate_policy(env, disc: Discretizer, policy: PolicyExecutor, episodes=1, seed=123):
    ret = []
    collided =0
    discomfort = []
    collision_speed = []
    for ep in range(episodes):
      discomfort_tracker = DiscomfortTracker(dt=0.1, init_action_int=0, init_a_ego=0.0, w_warning=0.0)
      obs, _ = env.reset(seed=seed+ep)
      done = False
      R = 0.0
      base = env.unwrapped
      num_steps = 0
      while not done:
        num_steps += 1
        a = policy(obs)
        obs, r, done, trunc, info = env.step(a)
        discomfort_tracker.step(action_int=a.value, a_ego=obs[2])
        R += r
        if info.get("collided"):
          collided += 1
          v_ego = getattr(base, "_v_s", np.nan)   # adjust names
          v_lead = getattr(base, "_v_l", np.nan)
          collision_speed.append(abs(v_lead - v_ego))
          break
        if trunc:
          break
      ret.append(R)
      discomfort.append(discomfort_tracker.cumulative / num_steps)
    return np.mean(ret), np.std(ret), collided, np.mean(discomfort), np.std(discomfort), np.mean(collision_speed)

if __name__ == "__main__":

  use_noisy_obs: bool = True

  scenarios = generate_training_scenarios()

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
  rewards_model = RewardsModel(model_cfg)

  executor = PolicyExecutor(disc, '/Users/htafish/projects/aa228/final_project/q_deterministic_planner.npy')
  # executor = NaivePolicyExecutor()

  # executor = DQNInference("aeb_dqn_qnet.pt")
  executor = DQNInference("belief_aeb_dqn_qnet.pt")
  
  @dataclass
  class Metrics:
    discomfort_mean: float
    return_mean: float
    return_std: float
    num_collisions: int
    mean_collision_speed: float
    
  results_table = {}
  
  sim_config = SimulationConfig()
  for sc_idx, sc in enumerate(scenarios):
    print(f"scenario: {sc}:")
    env = make_env(sc, rewards_model, dt=sim_config.dt, max_time=sim_config.total_time)
    if use_noisy_obs:
      # Per-dimension noise std for [gap, v_ego, a_ego, v_npc]
      sigma = np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float64)
      env = NoisyObsWrapper(env, sigma=sigma, clip=True, seed=123)
    mean_ret, std_ret, collided, discomfrotm_mean, discomfort_std, mean_collision_speed = evaluate_policy(env, disc, executor, episodes=10)
    results_table[sc_idx] = Metrics(discomfrotm_mean, mean_ret, std_ret, collided, mean_collision_speed)
    print(f"Evaluation: mean return={mean_ret:.2f} ± {std_ret:.2f}, collided: {collided}/10, discomfort_mean: {discomfrotm_mean:.2f}, discomfort_std: {discomfort_std:.2f}, mean collision speed: {mean_collision_speed}")
    if(collided > 0):
      print("COLLISION!!!!!!!!!!!!!")
    env.close()

  print("-------------------------------------------------------------------------------------")
  for k, v in results_table.items():
    print(f"| Scenario: {k}, discomfort_mean: {v.discomfort_mean:.2f}, return mean: {v.return_mean:.2f}, return std: {v.return_std:.2f}, num_collisions: {v.num_collisions}, mean collision speed: {v.mean_collision_speed}|")

  print("-------------------------------------------------------------------------------------")

  average_discomfort = 0.0
  total_collisions = 0
  average_collision_speed = 0.0
  collision_speeds = []
  for k, v in results_table.items():
    average_discomfort += v.discomfort_mean
    total_collisions += v.num_collisions
    if not math.isnan(v.mean_collision_speed):
      collision_speeds.append(v.mean_collision_speed)

  average_discomfort /= len(results_table)
  normalized_collision_score = total_collisions/(len(results_table)*10)
  normalized_discomfort_score = average_discomfort
  weighted_score = 0.7 * normalized_collision_score + 0.3 * normalized_discomfort_score
  print(f"Average discomfort: {average_discomfort}") 
  print(f"Total collisions: {total_collisions}") 
  print(f"normalized discomfort score: {normalized_discomfort_score}") 
  print(f"normalized collision score: {normalized_collision_score}") 
  print(f"weighted cost: {weighted_score}")
  print(f"Average collision speed: {np.mean(collision_speeds)}") 