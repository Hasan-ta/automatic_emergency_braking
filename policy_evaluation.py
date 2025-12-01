from scenario_factory import make_env, generate_training_scenarios
from discretizer import Discretizer
from global_config import DiscretizerConfig, SimulationConfig
from policy_executor import PolicyExecutor
from deterministic_model import DeterministicModelConfig, RewardsModel
from q_learning import DQNInference
import numpy as np

def evaluate_policy(env, disc: Discretizer, policy: PolicyExecutor, episodes=10, seed=123):
    ret = []
    for ep in range(episodes):
      obs, _ = env.reset(seed=seed+ep)
      done = False
      R = 0.0
      while not done:
        a = policy(obs)
        obs, r, done, trunc, _ = env.step(a)
        R += r
        if trunc:
          break
      ret.append(R)
    return np.mean(ret), np.std(ret)

if __name__ == "__main__":

  scenarios = generate_training_scenarios()

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
  rewards_model = RewardsModel(model_cfg)

  # executor = PolicyExecutor(disc, '/Users/htafish/projects/aa228/final_project/q_deterministic_planner.npy')

  executor = DQNInference("aeb_dqn_qnet.pt")
  
  sim_config = SimulationConfig()
  for sc in scenarios:
    print(f"scenario: {sc}:")
    env = make_env(sc, rewards_model, dt=sim_config.dt, max_time=sim_config.total_time)
    mean_ret, std_ret = evaluate_policy(env, disc, executor, episodes=10)
    print(f"Evaluation: mean return={mean_ret:.2f} ± {std_ret:.2f}")
    env.close()