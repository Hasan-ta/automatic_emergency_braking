import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deterministic_model import RewardsModel, DeterministicModelConfig, compute_ttc, State
from scenario_factory import make_env, generate_training_scenarios
from scenario_definitions import Family, Scenario, Action
from global_config import SimulationConfig
from aeb_v2v_env import AEBV2VEnv
from discretizer import Discretizer
from kalman_filter import GapEgoAccelKF
from noisy_obs_wrapper import NoisyObsWrapper

import random
import gymnasium as gym
import numpy as np

import signal

torch.set_default_dtype(torch.float64)

terminate_signal = False
def signal_handler(sig, frame):
    global terminate_signal
    print('Terminating!')
    terminate_signal = True

class MultiScenarioAEBEnv(gym.Env):
    """
    Wraps multiple Scenario instances and, on each reset(), picks one at random
    and creates an underlying AEBV2VEnv. From the outside it looks like a single
    env with obs = [gap, v_ego, v_npc] and Discrete(4) actions.
    """

    metadata = {}

    def __init__(self, scenarios, rewards_model: RewardsModel, dt: float = 0.05, max_time: float = 6.0):
        super().__init__()
        assert len(scenarios) > 0
        self.rewards_model = rewards_model
        self.scenarios = list(scenarios)
        self.dt = dt
        self.max_time = max_time

        self._inner_env: AEBV2VEnv | None = None
        self.current_scenario: Scenario | None = None

        # Build a dummy env just to copy spaces
        dummy_env = make_env(self.scenarios[0], rewards_model, dt=self.dt, max_time=self.max_time)
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

    def _make_inner_env(self, scenario: Scenario):
        return AEBV2VEnv(scenario, self.rewards_model, dt=self.dt, max_time=self.max_time)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # sample a scenario (uniform; you can weight this)
        self.current_scenario = random.choice(self.scenarios)
        self._inner_env = self._make_inner_env(self.current_scenario)

        obs, info = self._inner_env.reset(seed=seed, options=options)
        return obs, info | {"scenario": self.current_scenario}

    def step(self, action):
        if self._inner_env is None:
            raise RuntimeError("step() called before reset()")

        obs, r, terminated, truncated, info = self._inner_env.step(action)
        info = info | {"scenario": self.current_scenario}
        return obs, r, terminated, truncated, info

    def render(self):
        # If you ever add a render_mode to AEBV2VEnv, just forward it
        # if self._inner_env is not None:
        #     return self._inner_env.render()
        return None

    def close(self):
        if self._inner_env is not None:
            self._inner_env.close()
            self._inner_env = None



class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Simple fixed scaling for [gap, v_ego, v_npc]."""
    gap, v_e, a_e, v_n = obs
    # Example scales – tune to your scenario:
    gap_scale = 100.0    # m
    v_scale   = 60.0     # m/s (~216 km/h)
    a_scale = 10.0
    return np.array([gap / gap_scale, v_e / v_scale, a_e/a_scale, v_n / v_scale], dtype=np.float64)


def select_action(q_net, obs, eps: float, act_dim: int, device):
    if random.random() < eps:
        return random.randrange(act_dim)
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float64, device=device).unsqueeze(0)
        q_vals = q_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

def train_dqn(
    env,
    num_episodes: int = 500,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 50_000,
    target_update_freq: int = 1000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # obs_dim = env.observation_space.shape[0]  # 3
    obs_dim = 8
    act_dim = env.action_space.n              # 4

    q_net = QNetwork(obs_dim, act_dim).to(device)
    target_net = QNetwork(obs_dim, act_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=200_000)

    global_step = 0
    rewards_per_episode = []

    def linear_epsilon(step):
        if step >= eps_decay_steps:
            return eps_end
        return eps_start - (eps_start - eps_end) * (step / eps_decay_steps)

    for episode in range(num_episodes):
        global terminate_signal
        if terminate_signal:
            break
        obs_noisy, info = env.reset()
        # obs = normalize_obs(obs)
        done = False
        ep_reward = 0.0

        # KF init
        kf = GapEgoAccelKF(dt=0.1)
        kf.reset(obs_noisy[:3])
        b = kf.belief_features(include_var=True)


        while not done:
            eps = linear_epsilon(global_step)
            a = select_action(q_net, b, eps, act_dim, device)

            next_obs, r, terminated, truncated, info = env.step(a)
            # print(f"r: {r}, a: {a}, gap: {info["gap_m"]}")
            done = terminated or truncated
            kf.predict(0.0)
            kf.update(next_obs[:3])

            b_next = kf.belief_features(include_var=True)

            replay.push(b, a, r, b_next, done)
            b = b_next
            ep_reward += r
            global_step += 1

            # DQN update
            if len(replay) >= batch_size:
                s, a_b, r_b, s2, d_b = replay.sample(batch_size)

                s_t  = torch.tensor(s, dtype=torch.float64, device=device)
                a_t  = torch.tensor(a_b, dtype=torch.int64, device=device).unsqueeze(1)
                r_t  = torch.tensor(r_b, dtype=torch.float64, device=device).unsqueeze(1)
                s2_t = torch.tensor(s2, dtype=torch.float64, device=device)
                d_t  = torch.tensor(d_b, dtype=torch.float64, device=device).unsqueeze(1)

                # Q(s,a)
                q_vals = q_net(s_t).gather(1, a_t)  # (B,1)

                # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                with torch.no_grad():
                    q_next = target_net(s2_t).max(dim=1, keepdim=True)[0]
                    target = r_t + gamma * q_next * (1.0 - d_t)

                loss = nn.functional.mse_loss(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()

            # target network update
            if global_step % target_update_freq == 0 and global_step > 0:
                target_net.load_state_dict(q_net.state_dict())

        rewards_per_episode.append(ep_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, return={ep_reward:.2f}, eps={eps:.3f}")

    return q_net, target_net, rewards_per_episode


if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal_handler)

  scenarios = generate_training_scenarios()
  model_config = DeterministicModelConfig()
  rewards_model = RewardsModel(model_config)
  sim_config = SimulationConfig()
  # env = make_env(sc, rewards_model, dt=sim_config.dt)
  multi_env = MultiScenarioAEBEnv(scenarios, rewards_model, dt=sim_config.dt, max_time=sim_config.total_time)

  # Optional: add observation noise to [gap, v_ego, a_ego]
  sigma = np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float64)
  env = NoisyObsWrapper(multi_env, sigma=sigma, clip=True, seed=123)
  q_net, target_net, returns = train_dqn(env, batch_size=128, num_episodes=25000, eps_decay_steps=200_000)
  # q_net, target_net, returns = train_dqn(env, num_episodes=10000)
  torch.save(q_net.state_dict(), "belief_aeb_dqn_qnet.pt")


class DQNInference:
    def __init__(self, weights_path: str) -> None:
        self.q_net_loaded = QNetwork(obs_dim=8, act_dim=4)   # same architecture!
        self.q_net_loaded.load_state_dict(torch.load(weights_path))
        self.q_net_loaded.eval()  # set to eval mode for inference
        self.model_config = DeterministicModelConfig()
        self.kf = GapEgoAccelKF(dt=0.1)
        self.kf_initialized = False

    def __call__(self, obs) -> Action:
        if not self.kf_initialized:
            self.kf.reset(obs[:3])
            self.kf_initialized = True
            return Action.Nothing
        
        self.kf.predict(0.0)
        self.kf.update(obs[:3])

        mu = self.kf.mu

        ttc = compute_ttc(State(mu[0], mu[1], mu[2], mu[3]), self.model_config)
        if(ttc > 8.0):
            return Action.Nothing
        
        with torch.no_grad():
            x = torch.tensor(self.kf.belief_features(), dtype=torch.float64).unsqueeze(0)
            q = self.q_net_loaded(x)
            return Action(int(q.argmax(dim=1).item()))
        
    def get_policy(self, disc: Discretizer) -> np.ndarray:
        S = disc.num_states()
        centers = disc.state_centers()

        policy = np.empty((S,), np.int32)

        for s in range(S):
            policy[s] = self.__call__(centers[s]).value

        return policy
      

