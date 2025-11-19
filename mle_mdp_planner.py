# mle_mdp_planner.py
# Maximum-Likelihood MDP (model-based RL) + exploration for continuous Gym envs (e.g., AEB)
# - Discretize observations -> states
# - Online MLE of transitions/rewards with Dirichlet smoothing
# - Value Iteration over learned model
# - Exploration: epsilon-greedy AND/OR bonus in planning (MBIE-EB style)

from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix
from scenario_factory import make_env
from scenario_definitions import Family, Scenario

import signal

terminate_signal = False
def signal_handler(sig, frame):
    global terminate_signal
    print('Terminating!')
    terminate_signal = True

# =============== Discretizer ===============
@dataclass
class Discretizer:
    gap_bins: np.ndarray          # e.g., np.linspace(0, 60, 31)
    rel_bins: np.ndarray          # e.g., np.linspace(-30, 30, 31)
    ego_bins: np.ndarray          # e.g., np.linspace(0, 40, 21)

    def state_dim(self) -> Tuple[int, int, int]:
        return (len(self.gap_bins)-1, len(self.rel_bins)-1, len(self.ego_bins)-1)

    def to_state(self, obs: np.ndarray) -> int:
        """Map continuous obs to a single int state id by 3D binning."""
        # Try V2V first: [gap, v_rel, v_ego]; else Ped: [dx, dy, v_ego] -> map |dy| as 'gap-ish'
        try:
            gap, v_rel, v_ego = float(obs[0]), float(obs[1]), float(obs[2])
        except Exception:
            # Ped mapping: use dx as 'gap' and -|dy| as 'relative speed proxy' to favor near-crossing
            gap = float(obs[0])
            v_rel = -abs(float(obs[1]))
            v_ego = float(obs[2])

        i = np.clip(np.digitize(gap, self.gap_bins)-1, 0, len(self.gap_bins)-2)
        j = np.clip(np.digitize(v_rel, self.rel_bins)-1, 0, len(self.rel_bins)-2)
        k = np.clip(np.digitize(v_ego, self.ego_bins)-1, 0, len(self.ego_bins)-2)

        dims = self.state_dim()
        sid = (i * dims[1] + j) * dims[2] + k
        return int(sid)

    def num_states(self) -> int:
        d = self.state_dim()
        return d[0]*d[1]*d[2]
    
# class TransitionModel:
#     def __init__(self, discretizer: Discretizer, n_actions: int):
#         self.T = []
#         for


# =============== Maximum-Likelihood MDP model ===============
@dataclass
class MLEConfig:
    n_states: int
    n_actions: int
    alpha: float = 0.1          # Dirichlet prior (smoothing) for transitions
    reward_clip: Optional[Tuple[float,float]] = (-1000.0, 5.0)

class MLEModel:
    def __init__(self, cfg: MLEConfig):
        self.cfg = cfg
        S, A = cfg.n_states, cfg.n_actions
        # Transition counts and reward stats
        self.counts = np.zeros((S, A), dtype=np.int32)
        self.counts_next = np.zeros((S, A, S), dtype=np.int32)
        self.R_sum = np.zeros((S, A), dtype=np.float64)
        self.R_mean = np.zeros((S, A), dtype=np.float64)

    def update(self, s: int, a: int, r: float, sp: int):
        self.counts[s, a] += 1
        self.counts_next[s, a, sp] += 1
        # running mean for rewards
        n = self.counts[s, a]
        if self.cfg.reward_clip is not None:
            r = float(np.clip(r, self.cfg.reward_clip[0], self.cfg.reward_clip[1]))
        self.R_sum[s, a] += r
        self.R_mean[s, a] = self.R_sum[s, a] / max(1, n)

    def transition_matrix(self, a):
        S, A = self.cfg.n_states, self.cfg.n_actions
        T = np.empty((S, S), dtype=np.float64)
        for s in range(S):
            T[s,:] = self.transition_row(s, a)
        return T

    def transition_row(self, s: int, a: int) -> np.ndarray:
        # Dirichlet posterior mean with symmetric prior alpha
        alpha = self.cfg.alpha
        row = self.counts_next[s, a].astype(np.float64) + alpha
        row_sum = row.sum()
        if row_sum <= 0.0:
            row_sum = 1.0
        row /= row_sum
        return row

    def reward_sa(self, s: int, a: int) -> float:
        return float(self.R_mean[s, a])
    
    def reward_a(self, a: int):
        return self.R_mean[:, a]


# =============== Value Iteration with optional exploration bonus ===============
@dataclass
class PlannerConfig:
    gamma: float = 0.995
    tol: float = 1e-6
    max_iter: int = 10_000
    # Exploration bonus coefficients (MBIE-EB/UCB style): bonus = beta / sqrt(N(s,a) + 1)
    beta: float = 2.0
    use_bonus: bool = True

class ValueIterationPlanner:
    def __init__(self, model: MLEModel, p_cfg: PlannerConfig):
        self.model = model
        self.cfg = p_cfg
        S, A = model.cfg.n_states, model.cfg.n_actions
        self.U = np.zeros(S, dtype=np.float64)
        self.Q = np.zeros((S, A), dtype=np.float64)

    def plan(self):
        S, A = self.model.cfg.n_states, self.model.cfg.n_actions
        gamma, tol, max_iter = self.cfg.gamma, self.cfg.tol, self.cfg.max_iter
        # beta = self.cfg.beta
        # counts = self.model.counts

        # U = self.U
        # for it in range(max_iter):
        #     U_old = U.copy()
        #     # Bellman backup with exploration bonus in reward
        #     for s in range(S):
        #         best = -1e30
        #         for a in range(A):
        #             p = self.model.transition_row(s, a)        # S'
        #             r = self.model.reward_sa(s, a)
        #             bonus = (beta / math.sqrt(counts[s, a] + 1.0)) if self.cfg.use_bonus else 0.0
        #             q = r + bonus + gamma * (p @ U_old)
        #             if q > best:
        #                 best = q
        #         U[s] = best
        #     bellman_residual = np.max(np.abs(U - U_old))
        #     if it%50 == 0:
        #         print(f"Value Iteration [{it}], residual: {bellman_residual}")
        #     if bellman_residual < tol:
        #         break

        P = [csr_matrix(self.model.transition_matrix(a)) for a in range(A)]
        for it in range(max_iter):
            U_old = self.U.copy()
            # Bellman backup with exploration bonus in reward
            for a in range(A):
                p = P[a]
                r = self.model.reward_a(a)
                # bonus = (beta / math.sqrt(counts[s, a] + 1.0)) if self.cfg.use_bonus else 0.0
                self.Q[:, a] = r + gamma * (p @ U_old)
            self.U = self.Q.max(axis=1)
            bellman_residual = np.max(np.abs(self.U - U_old))
            if it%50 == 0:
                print(f"Value Iteration [{it}], residual: {bellman_residual}")
            if bellman_residual < tol:
                break

        # # Optional Q reconstruction
        # for s in range(S):
        #     for a in range(A):
        #         p = self.model.transition_row(s, a)
        #         r = self.model.reward_sa(s, a)
        #         bonus = (beta / math.sqrt(counts[s, a] + 1.0)) if self.cfg.use_bonus else 0.0
        #         self.Q[s, a] = r + bonus + gamma * (p @ self.U)

    def act(self, s: int) -> int:
        return int(np.argmax(self.Q[s]))


# =============== ε-greedy controller ===============
@dataclass
class ControlConfig:
    epsilon_start: float = 0.2
    epsilon_final: float = 0.02
    epsilon_decay_steps: int = 50_000

class EpsGreedyController:
    def __init__(self, planner: ValueIterationPlanner, ctrl: ControlConfig):
        self.planner = planner
        self.ctrl = ctrl
        self.steps = 0

    def epsilon(self) -> float:
        t = min(self.steps, self.ctrl.epsilon_decay_steps)
        frac = 1.0 - t / self.ctrl.epsilon_decay_steps
        return self.ctrl.epsilon_final + (self.ctrl.epsilon_start - self.ctrl.epsilon_final) * frac

    def act(self, s: int, n_actions: int) -> int:
        self.steps += 1
        if np.random.rand() < self.epsilon():
            return np.random.randint(n_actions)
        return self.planner.act(s)


# =============== Training loop (online) ===============
@dataclass
class TrainConfig:
    episodes: int = 200
    steps_per_ep: int = 400
    plan_every_k_steps: int = 200          # re-run VI periodically
    print_every: int = 1

def train_mle_mdp(scenarios,
                  disc: Discretizer,
                  plan_cfg: PlannerConfig,
                  eps_cfg: ControlConfig,
                  train_cfg: TrainConfig,
                  seed: int = 0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    today = datetime.now()
    formatted_date = today.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = "q_function_" + formatted_date + ".npy"

    # State/action sizes
    S = disc.num_states()
    A = 4

    model = MLEModel(MLEConfig(n_states=S, n_actions=A, alpha=0.0))
    planner = ValueIterationPlanner(model, plan_cfg)

    for sc_i, sc in enumerate(scenarios):
        print(f"Optimizing scenario:{sc}")
        controller = EpsGreedyController(planner, eps_cfg)
        env = make_env(sc, dt=0.05)
        # A = env.action_space.n

        global_step = 0
        returns = []

        for ep in range(train_cfg.episodes):
            if terminate_signal:
                break
            obs, _ = env.reset(seed=seed + ep)
            s = disc.to_state(obs)
            ep_ret = 0.0

            for t in range(train_cfg.steps_per_ep):
                # Plan occasionally (expensive); first time forces Q init
                if global_step % train_cfg.plan_every_k_steps == 0:
                    planner.plan()

                a = controller.act(s, A)
                obs2, r, done, trunc, info = env.step(a)
                sp = disc.to_state(obs2)

                # Update model with this new sample
                model.update(s, a, r, sp)

                ep_ret += r
                global_step += 1
                s = sp
                if done or trunc:
                    break

            returns.append(ep_ret)
            if (ep+1) % train_cfg.print_every == 0:
                avg = np.mean(returns[-train_cfg.print_every:])
                print(f"[sc {sc_i}][ep {ep+1:4d}] avg return (last {train_cfg.print_every}): {avg:.2f} | eps={controller.epsilon():.3f}")

            if(ep+1) % 20 == 0:
                np.save(file_name, planner.Q)
        env.close()

    # Final plan for a clean policy
    planner.plan()
    return model, planner, returns


# =============== Small demo ===============
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    scenarios = []
    # sc = next(s for s in generate_fmvss127()
    #           if s.family == Family.V2V_DECELERATING and s.subject_speed_kmh == 80 and s.headway_m == 20)
    # scenarios.append(sc)
    # scenarios.append(Scenario(Family.V2V_STATIONARY, 50, 0, manual_brake=False, headway_m=40, note="S7.3 no manual"))

    # for v in list(range(10, 81, 10)):
    #     scenarios.append(Scenario(Family.V2V_STATIONARY, v, 0, manual_brake=False, headway_m=6*0.277778*v, note="S7.3 no manual"))
    # for v in [40, 50, 60, 70, 80]:
    #     scenarios.append(Scenario(Family.V2V_SLOWER_MOVING, v, 20, manual_brake=False, headway_m=6*(v-20)*277778, note="S7.4 no manual"))
    # for v in [50, 80]:
    #     for hw in [12, 20, 30, 40]:
    #         for decel_g in [0.3, 0.4, 0.5]:
    #             for manual in [False]:
    #                 scenarios.append(Scenario(Family.V2V_DECELERATING, v, v, lead_decel_ms2=decel_g*9.80665,
    #                                headway_m=hw, manual_brake=manual, note="S7.5"))

    scenarios.append(Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=10, lead_speed_kmh=0, lead_decel_ms2=None, headway_m=16.66668, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual'))

    # Discretization grids (tune to your dynamics):
    # Gap up to 60m, rel_speed -30..30 m/s, ego 0..40 m/s
    disc = Discretizer(
        gap_bins=np.linspace(0.0, 60.0, 31),      # 30 bins (2m)
        rel_bins=np.linspace(-30.0, 30.0, 31),    # 30 bins (2 m/s)
        ego_bins=np.linspace(0.0, 40.0, 21),      # 20 bins (2 m/s)
    )

    plan_cfg = PlannerConfig(gamma=0.99, tol=1e-6, max_iter=10_000, beta=2.0, use_bonus=False)
    eps_cfg = ControlConfig(epsilon_start=0.5, epsilon_final=0.02, epsilon_decay_steps=50_000)
    train_cfg = TrainConfig(episodes=2000, steps_per_ep=250, plan_every_k_steps=2000, print_every=5)

    model, planner, returns = train_mle_mdp(scenarios, disc, plan_cfg, eps_cfg, train_cfg, seed=0)

    # Evaluate the learned policy without exploration
    def evaluate_policy(env, disc, planner, episodes=10, seed=123):
        ret = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed+ep)
            s = disc.to_state(obs)
            done = False
            R = 0.0
            while not done:
                a = planner.act(s)
                obs, r, done, trunc, _ = env.step(a)
                R += r
                s = disc.to_state(obs)
                if trunc:
                    break
            ret.append(R)
        return np.mean(ret), np.std(ret)
    
    for sc in scenarios:
        print(f"scenario: {sc}:")
        env = make_env(sc, dt=0.05)
        mean_ret, std_ret = evaluate_policy(env, disc, planner, episodes=10)
        print(f"Evaluation: mean return={mean_ret:.2f} ± {std_ret:.2f}")
        env.close()
