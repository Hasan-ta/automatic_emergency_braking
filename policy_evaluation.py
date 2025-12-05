from scenario_factory import make_env, generate_training_scenarios
from discretizer import Discretizer, GapEgoAccelDiscretizer
from global_config import DiscretizerConfig, SimulationConfig
from policy_executor import PolicyExecutor
from deterministic_model import DeterministicModelConfig, RewardsModel
from q_learning import DQNInference
from noisy_obs_wrapper import NoisyObsWrapper
import numpy as np

def evaluate_policy(env, disc: Discretizer, policy: PolicyExecutor, episodes=1, seed=123):
    ret = []
    collided =0
    for ep in range(episodes):
      obs, _ = env.reset(seed=seed+ep)
      done = False
      R = 0.0
      while not done:
        a = policy(obs)
        obs, r, done, trunc, info = env.step(a)
        R += r
        if info.get("collided"):
          collided += 1
        if trunc:
          break
      ret.append(R)
    return np.mean(ret), np.std(ret), collided

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

  # executor = DQNInference("aeb_dqn_qnet.pt")
  # executor = DQNInference("belief_aeb_dqn_qnet.pt")
  
  
  sim_config = SimulationConfig()
  for sc in scenarios:
    print(f"scenario: {sc}:")
    env = make_env(sc, rewards_model, dt=sim_config.dt, max_time=sim_config.total_time)
    if use_noisy_obs:
      # Per-dimension noise std for [gap, v_ego, a_ego, v_npc]
      sigma = np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float64)
      env = NoisyObsWrapper(env, sigma=sigma, clip=True, seed=123)
    mean_ret, std_ret, collided = evaluate_policy(env, disc, executor, episodes=10)
    print(f"Evaluation: mean return={mean_ret:.2f} ± {std_ret:.2f}, collided: {collided}/10")
    if(collided > 0):
      print("COLLISION!!!!!!!!!!!!!")
    env.close()