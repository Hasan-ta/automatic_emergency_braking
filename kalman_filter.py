import numpy as np

class GapEgoAccelKF:
    """
    KF belief over x = [gap, v_ego, a_ego, v_npc]^T
    """
    def __init__(
        self,
        dt: float,
        q_diag=(0.5, 0.5, 1.0, 1.0),   # process noise stds
        r_diag=(0.5, 0.5, 0.5),   # observation noise stds
    ):
        self.dt = float(dt)
        self.n = 4

        self.mu = np.zeros((4, 1), dtype=np.float64)
        self.Sigma = np.eye(4, dtype=np.float64) * 2.0

        q = np.asarray(q_diag, dtype=np.float64)
        r = np.asarray(r_diag, dtype=np.float64)
        self.Q = np.diag(q**2)
        self.R = np.diag(r**2)

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    def reset(self, init_obs):
        self.mu[:3] = np.asarray(init_obs, dtype=np.float64).reshape(3, 1)
        self.mu[-1, 0] = 0.0
        self.Sigma = np.eye(4, dtype=np.float64)

    def _F_and_u(self, a_npc: float):
        dt = self.dt

        # State order: [gap, v_ego, a_ego, v_npc]
        F = np.array([
            [1.0,   -dt,  -0.5 * dt**2,   dt],  # gap update depends on v_ego, a_ego, v_npc
            [0.0,   1.0,       dt,        0.0], # v_ego
            [0.0,   0.0,      1.0,        0.0], # a_ego random walk
            [0.0,   0.0,      0.0,        1.0], # v_npc
        ], dtype=np.float64)

        # affine term from known npc acceleration
        u = np.array([
            0.5 * a_npc * dt**2,
            0.0,
            0.0,
            a_npc * dt,
        ], dtype=np.float64).reshape(4, 1)

        return F, u

    def predict(self, a_npc: float):
        F, u = self._F_and_u(a_npc)
        self.mu = F @ self.mu + u
        self.Sigma = F @ self.Sigma @ F.T + self.Q

    def update(self, obs):
        z = np.asarray(obs, dtype=np.float64).reshape(3, 1)

        S = self.H @ self.Sigma @ self.H.T + self.R
        K = self.Sigma @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ self.mu
        self.mu = self.mu + K @ y
        self.Sigma = (np.eye(self.n) - K @ self.H) @ self.Sigma

    def belief_features(self, include_var=True):
        mu_flat = self.mu.flatten().astype(np.float64)
        if not include_var:
            return mu_flat  # 4 dims

        var = np.diag(self.Sigma).astype(np.float64)
        return np.concatenate([mu_flat, var], axis=0)  # 8 dims
