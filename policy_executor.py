import numpy as np
from discretizer import Discretizer
from scenario_definitions import Action, Actor
from deterministic_model import State, compute_ttc, DeterministicModelConfig
from kalman_filter import GapEgoAccelKF


class PolicyExecutor:
  def __init__(self, disc: Discretizer, q_path: str) -> None:
    self.Q = np.load(q_path)
    # self.disc = Discretizer(
    #     gap_bins=np.linspace(0.0, 60.0, 31),      # 30 bins (2m)
    #     rel_bins=np.linspace(-30.0, 30.0, 31),    # 30 bins (2 m/s)
    #     ego_bins=np.linspace(0.0, 40.0, 21),      # 20 bins (2 m/s)
    # )

    self.disc = disc
    self.model_config = DeterministicModelConfig()
    self.kf = GapEgoAccelKF(dt=0.1)
    self.kf_initialized = False
    self.obs = []
    self.est = []

  def gaussian_belief_over_states(self, mu: np.ndarray,
                                Sigma: np.ndarray,
                                state_centers: np.ndarray,
                                max_states: int = 256) -> np.ndarray:
    """
    Build a discrete belief over states from a Gaussian belief over continuous x.
    - mu:  (d,) mean of x
    - Sigma: (d,d) covariance of x
    - state_centers: (S,d) continuous coordinates of each discrete state
    - max_states: limit to top-K most likely states for efficiency
    Returns: b (S,) with sum(b) = 1
    """
    S, d = state_centers.shape
    # Mahalanobis distances
    # For robustness, use pseudo-inverse
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)

    diff = state_centers - mu.reshape(1, 4)       # (S,d)
    m2 = np.einsum("sd,dd,sd->s", diff, Sigma_inv, diff)  # (S,)

    # Unnormalized Gaussian weights (ignore constant (2π)^{-d/2} det^{-1/2})
    w = np.exp(-0.5 * m2)

    # Optional: restrict to top-K states
    if max_states is not None and max_states < S:
        idx = np.argpartition(-w, max_states)[:max_states]
        w_small = w[idx]
        if w_small.sum() <= 1e-15:
            # fallback: uniform over these K
            b = np.zeros(S, dtype=np.float64)
            b[idx] = 1.0 / len(idx)
            return b
        w_small /= w_small.sum()
        b = np.zeros(S, dtype=np.float64)
        b[idx] = w_small
        return b

    # Otherwise use all states
    if w.sum() <= 1e-15:
        return np.ones(S, dtype=np.float64) / S
    return w / w.sum()

  def qmdp_action(self,
                  mu: np.ndarray,
                  Sigma: np.ndarray,
                  disc: 'Discretizer',
                  max_states: int = 256) -> int:
        """
        QMDP: use belief (Gaussian over continuous state) to form a
        discrete belief over states, then compute Q(b,a) = sum_s b_s Q[s,a].
        """
        centers = disc.state_centers()     # (S,3) or (S,d)
        b = self.gaussian_belief_over_states(mu, Sigma, centers, max_states=max_states)  # (S,)
        # Q_b(a) = sum_s b(s) Q[s,a] = b^T Q
        Q_b = b @ self.Q                   # (A,)
        return int(np.argmax(Q_b))

  def __call__(self, obs) -> Action:
    self.obs.append(obs)
    if not self.kf_initialized:
      self.kf.reset(obs[:3])
      self.kf_initialized = True
      return Action.Nothing
    
    self.kf.predict(0.0)
    self.kf.update(obs[:3])

    mu = self.kf.mu
    self.est.append(mu)

    # ttc = compute_ttc(State(mu[0], mu[1], mu[2], mu[3]), self.model_config)
    # if(ttc > 4.0):
    #     return Action.Nothing
    # s = self.disc.obs_to_state(obs)
    # return Action(int(np.argmax(self.Q[s])))

    return Action(self.qmdp_action(self.kf.mu, self.kf.Sigma, self.disc))
  

class NaivePolicyExecutor:
  def __init__(self) -> None:
    self.kf = GapEgoAccelKF(dt=0.1)
    self.kf_initialized = False
    self.model_config = DeterministicModelConfig()
    self.obs = []
    self.est = []
    self.v_std = []

  def ttc_policy(self, ttc):
    if ttc < 2.0:
        return Action.StrongBrake
    elif ttc < 4.0:
        return Action.SoftBrake
    else:
        return Action.Nothing

  def __call__(self, obs) -> Action:
    self.obs.append(obs)
    if not self.kf_initialized:
      self.kf.reset(obs[:3])
      self.kf_initialized = True
      self.est.append(self.kf.mu)
      self.v_std.append(np.sqrt(self.kf.Sigma[1, 1] + self.kf.Sigma[3, 3]))
      print(np.sqrt(self.kf.Sigma[3, 3]))
      return Action.Nothing
    
    self.kf.predict(0.0)
    self.kf.update(obs[:3])

    mu = self.kf.mu
    self.est.append(mu)
    self.v_std.append(np.sqrt(self.kf.Sigma[1, 1] + self.kf.Sigma[3, 3]))
    print(np.sqrt(self.kf.Sigma[3, 3]))

    ttc = compute_ttc(State(mu[0], mu[1], mu[2], mu[3]), self.model_config)
    # print(f"ttc from executor: {ttc}")

    return self.ttc_policy(ttc)