from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class Discretizer:
    """
    Discretizer for observation: [x_ego, x_npc, v_ego, v_npc]

    Each dimension has its own 1D bin edges. We use np.digitize to map a
    continuous value to a bin index, then pack (i_xe, i_xn, i_ve, i_vn)
    into a single flat state index.
    """
    x_ego_bins: np.ndarray   # shape (Nx+1,)
    x_npc_bins: np.ndarray   # shape (Nx+1,)
    v_ego_bins: np.ndarray   # shape (Nv+1,)
    v_npc_bins: np.ndarray   # shape (Nv+1,)

    # ---------- basic shape info ----------

    def shape(self) -> Tuple[int, int, int, int]:
        """
        Number of discrete bins in each dimension.
        """
        return (
            len(self.x_ego_bins) - 1,
            len(self.x_npc_bins) - 1,
            len(self.v_ego_bins) - 1,
            len(self.v_npc_bins) - 1,
        )

    def num_states(self) -> int:
        nx_e, nx_n, nv_e, nv_n = self.shape()
        return nx_e * nx_n * nv_e * nv_n

    # ---------- mapping continuous -> bin indices ----------

    def _bin_index(self, value: float, bins: np.ndarray) -> int:
        """
        Map value to a bin index in [0, len(bins)-2] using np.digitize,
        clamped to the valid range.
        """
        idx = np.digitize(value, bins) - 1
        idx = int(np.clip(idx, 0, len(bins) - 2))
        return idx

    def obs_to_indices(self, obs: np.ndarray) -> Tuple[int, int, int, int]:
        """
        obs: array-like (4,) = [x_ego, x_npc, v_ego, v_npc]
        Returns the 4D bin indices.
        """
        x_ego, x_npc, v_ego, v_npc = [float(o) for o in obs]
        i_xe = self._bin_index(x_ego, self.x_ego_bins)
        i_xn = self._bin_index(x_npc, self.x_npc_bins)
        i_ve = self._bin_index(v_ego, self.v_ego_bins)
        i_vn = self._bin_index(v_npc, self.v_npc_bins)
        return i_xe, i_xn, i_ve, i_vn

    def values_to_indices(
        self, x_ego: float, x_npc: float, v_ego: float, v_npc: float
    ) -> Tuple[int, int, int, int]:
        return self.obs_to_indices([x_ego, x_npc, v_ego, v_npc])

    # ---------- pack/unpack flat state index ----------

    def indices_to_state(self, i_xe: int, i_xn: int, i_ve: int, i_vn: int) -> int:
        """
        Pack 4D indices -> flat state id.
        """
        nx_e, nx_n, nv_e, nv_n = self.shape()
        # ((i_xe * Nx_n + i_xn) * Nv_e + i_ve) * Nv_n + i_vn
        s = ((i_xe * nx_n + i_xn) * nv_e + i_ve) * nv_n + i_vn
        return int(s)

    def state_to_indices(self, s: int) -> Tuple[int, int, int, int]:
        """
        Unpack flat state id -> 4D indices.
        """
        nx_e, nx_n, nv_e, nv_n = self.shape()
        s = int(s)
        i_xe = s // (nx_n * nv_e * nv_n)
        r = s % (nx_n * nv_e * nv_n)

        i_xn = r // (nv_e * nv_n)
        r = r % (nv_e * nv_n)

        i_ve = r // nv_n
        i_vn = r % nv_n

        return i_xe, i_xn, i_ve, i_vn

    # ---------- main interface: continuous obs <-> flat state ----------

    def obs_to_state(self, obs: np.ndarray) -> int:
        """
        Continuous observation -> flat discrete state id.
        """
        i_xe, i_xn, i_ve, i_vn = self.obs_to_indices(obs)
        return self.indices_to_state(i_xe, i_xn, i_ve, i_vn)

    def values_to_state(self, x_ego: float, x_npc: float, v_ego: float, v_npc: float) -> int:
        i_xe, i_xn, i_ve, i_vn = self.values_to_indices(x_ego, x_npc, v_ego, v_npc)
        return self.indices_to_state(i_xe, i_xn, i_ve, i_vn)

    # ---------- centers & iteration helpers (handy for planning) ----------

    def _bin_centers(self, bins: np.ndarray) -> np.ndarray:
        return 0.5 * (bins[:-1] + bins[1:])

    def state_centers(self) -> np.ndarray:
        """
        Returns array of shape (S, 4) with the continuous center coordinates
        [x_ego, x_npc, v_ego, v_npc] for each discrete state.
        """
        x_e_centers = self._bin_centers(self.x_ego_bins)
        x_n_centers = self._bin_centers(self.x_npc_bins)
        v_e_centers = self._bin_centers(self.v_ego_bins)
        v_n_centers = self._bin_centers(self.v_npc_bins)

        Xe, Xn, Ve, Vn = np.meshgrid(
            x_e_centers, x_n_centers, v_e_centers, v_n_centers, indexing="ij"
        )
        centers = np.stack([Xe, Xn, Ve, Vn], axis=-1)  # (Nx_e, Nx_n, Nv_e, Nv_n, 4)
        return centers.reshape(-1, 4)                  # (S, 4)

    def index_tuples(self) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of all (i_xe, i_xn, i_ve, i_vn) index tuples for all states.
        Order matches the flat indexing used in indices_to_state/state_to_indices.
        """
        nx_e, nx_n, nv_e, nv_n = self.shape()
        tuples = []
        for i_xe in range(nx_e):
            for i_xn in range(nx_n):
                for i_ve in range(nv_e):
                    for i_vn in range(nv_n):
                        tuples.append((i_xe, i_xn, i_ve, i_vn))
        return tuples

    # ---------- convenience constructor ----------

    @classmethod
    def from_ranges(
        cls,
        x_min: float,
        x_max: float,
        v_min: float,
        v_max: float,
        n_x_ego: int = 40,
        n_x_npc: int = 40,
        n_v_ego: int = 20,
        n_v_npc: int = 20,
    ) -> "Discretizer":
        """
        Build a uniform-grid discretizer from simple ranges.
        """
        x_ego_bins = np.linspace(x_min, x_max, n_x_ego + 1, dtype=np.float64)
        x_npc_bins = np.linspace(x_min, x_max, n_x_npc + 1, dtype=np.float64)
        v_ego_bins = np.linspace(v_min, v_max, n_v_ego + 1, dtype=np.float64)
        v_npc_bins = np.linspace(v_min, v_max, n_v_npc + 1, dtype=np.float64)
        return cls(x_ego_bins, x_npc_bins, v_ego_bins, v_npc_bins)
