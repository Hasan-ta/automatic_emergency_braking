import argparse
import numpy as np
from fmvss_simulation import make_env, Family, Scenario
from policy_executor import PolicyExecutor


# Evaluate the learned policy without exploration
def evaluate_policy(executor, episodes=10, seed=123):
  scenarios = []
  # for v in list(range(10, 81, 10)):
  #       scenarios.append(Scenario(Family.V2V_STATIONARY, v, 0, manual_brake=False, headway_m=6*0.277778*v, note="S7.3 no manual"))
  # for v in [40, 50, 60, 70, 80]:
  #     scenarios.append(Scenario(Family.V2V_SLOWER_MOVING, v, 20, manual_brake=False, headway_m=6*(v-20)*277778, note="S7.4 no manual"))
  # for v in [50, 80]:
  #     for hw in [12, 20, 30, 40]:
  #         for decel_g in [0.3, 0.4, 0.5]:
  #             for manual in [False]:
  #                 scenarios.append(Scenario(Family.V2V_DECELERATING, v, v, lead_decel_ms2=decel_g*9.80665,
  #                                 headway_m=hw, manual_brake=manual, note="S7.5"))

  scenarios.append(Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=10, lead_speed_kmh=0, lead_decel_ms2=None, headway_m=16.66668, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual'))

  for sc in scenarios:
    print(f"scenario: {sc}:")
    env = make_env(sc, dt=0.05)
    ret = []

    for ep in range(episodes):
      obs, _ = env.reset(seed=seed+ep)
      done = False
      R = 0.0
      while not done:
          a = executor(obs)
          obs, r, done, trunc, _ = env.step(a)
          R += r
          if trunc:
              break
      ret.append(R)

    env.close()
    mean_ret, std_ret = np.mean(ret), np.std(ret)
    
    print(f"Evaluation: mean return={mean_ret:.2f} ± {std_ret:.2f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate policy on FMVSS policies.")
  parser.add_argument("--policy", required=True, help="The policy file")
  args = parser.parse_args()

  executor = PolicyExecutor(args.policy)

  evaluate_policy(executor)