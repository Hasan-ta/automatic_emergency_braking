import numpy as np
import matplotlib.pyplot as plt
from policy_executor import NaivePolicyExecutor, PolicyExecutor
from policy_evaluation import evaluate_policy
from discretizer import GapEgoAccelDiscretizer
from global_config import DiscretizerConfig, SimulationConfig
from deterministic_model import DeterministicModelConfig, RewardsModel
from scenario_factory import make_env
from scenario_definitions import Scenario, Family
from noisy_obs_wrapper import NoisyObsWrapper

def run_evaluation():
  sim_config = SimulationConfig()
  model_cfg = DeterministicModelConfig()
  rewards_model = RewardsModel(model_cfg)
  disc_config= DiscretizerConfig()
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
  executor1 = NaivePolicyExecutor()
  executor2 = PolicyExecutor(disc, '/Users/htafish/projects/aa228/final_project/q_deterministic_planner.npy')

  sc= Scenario(family=Family.V2V_DECELERATING, subject_speed_kmh=80, lead_speed_kmh=80, lead_decel_ms2=3.92266, headway_m=12, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.5')
  env = make_env(sc, rewards_model, dt=sim_config.dt, max_time=sim_config.total_time)
  sigma = np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float64)
  env = NoisyObsWrapper(env, sigma=sigma, clip=True, seed=123)

  _, axes = plt.subplots(nrows=3, ncols=1, figsize=(2.2,8), dpi=300)
  for i, executor in enumerate((executor1, executor2)):
    mean_ret, std_ret, collided, discomfrotm_mean, discomfort_std, mean_collision_speed = evaluate_policy(env, disc, executor, episodes=1)
    print("------")
    print(executor)
    print(f"collision: {collided}")

  obs1 = np.array(executor1.obs)
  obs2 = np.array(executor2.obs)
  # est = [np.array([[0], [0], [0], [0]])] + executor1.est
  est = executor1.est
  est = np.array(est)

  t = np.arange(obs1.shape[0])*0.1
  print(obs1.shape)
  print(est.shape)
  axes[0].plot(t[:], obs1[:, 3] - obs1[:, 1])
  axes[0].plot(t[:], est[:, 3] - est[:, 1])
  v_mean = (est[:, 3] - est[:, 1]).squeeze(-1)
  v_std = np.array(executor1.v_std)
  axes[0].fill_between(
    t[:],
    v_mean - v_std,
    v_mean + v_std,
    alpha=0.2,
    label="±1σ"
  )
  axes[0].legend(["Observed relative velocity", "Estimated relative velocity", "Relative velocity uncertainty"])
  # axes[0].set_xlabel("Simutlaion time [s]")
  axes[0].set_ylabel("v_t - v_e [m/s]")
  axes[0].set_xticklabels([])

  t2 = np.arange(obs2.shape[0])*0.1
  axes[1].plot(t, obs1[:,2])
  axes[1].grid()
  axes[1].set_title("Baseline deceleration profile")
  # axes[1].set_xlabel("Simutlaion time [s]")
  axes[1].set_ylabel("a_e [m/s^2]")
  axes[1].set_xticklabels([])
  axes[2].plot(t2, obs2[:,2])
  axes[2].grid()
  axes[2].set_title("QMDP deceleration profile")
  axes[2].set_xlabel("Simutlaion time [s]")
  axes[2].set_ylabel("a_e [m/s^2]")

  plt.savefig(f"fig10.png", bbox_inches="tight")
  
  plt.show()




def main():
    # Data (float64)
    metrics = [
        ("p(collision)", np.float64(0.19), np.float64(0.34)),
        ("Average collision speed",            np.float64(3.0), np.float64(2.0)),
        ("Average Discomfort",               np.float64(15.33), np.float64(14.52)),
    ]
    labels = ["QMDP", "Baseline"]

    # IEEE-friendly compact styling
    plt.rcParams.update({
        "font.size": 4,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # Single-column IEEE-ish figure size
    # (Each category gets its own separate figure)
    fig_size = (3.5, 1.6)  # width ~ single-column; short height

    i = 3
    for name, qmdp_val, base_val in metrics:
        values = np.array([qmdp_val, base_val], dtype=np.float64)

        plt.figure(figsize=fig_size, dpi=300)
        bars = plt.bar(labels, values, label=None)

        # Y label per plot
        plt.ylabel(name)

        # Light grid
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        # Headroom for text labels
        ymax = float(values.max())
        plt.ylim(0.0, ymax * 1.25 if ymax > 0 else 1.0)

        # Write bar heights on top of bars
        for bar in bars:
            h = bar.get_height()
            text = f"{h:.3f}".rstrip("0").rstrip(".")
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                text,
                ha="center",
                va="bottom",
                fontsize=8
            )

        # Optional title (comment out if you want ultra-minimal IEEE style)
        plt.title(name)

        plt.tight_layout()

        # Optional save (one file per metric)
        # safe_name = name.lower().replace(" ", "_")
        # plt.savefig(f"{safe_name}_qmdp_vs_baseline.pdf", bbox_inches="tight")
        plt.savefig(f"fig_{i}.png", bbox_inches="tight")
        i += 1

        plt.show()

    run_evaluation()

if __name__ == "__main__":
    main()
