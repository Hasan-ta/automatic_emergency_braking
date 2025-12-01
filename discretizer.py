from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class Discretizer:
    """
    Discretizer for observation: [gap, v_ego, v_npc]

    Each dimension has its own 1D bin edges. We map continuous values to
    bin indices via np.digitize, then pack (i_gap, i_ve, i_vn) into a
    single flat state index s ∈ {0, ..., S-1}.
    """

    gap_bins: np.ndarray    # shape (Ng+1,)
    v_ego_bins: np.ndarray  # shape (Nv_e+1,)
    v_npc_bins: np.ndarray  # shape (Nv_n+1,)

    # ---------- basic shape info ----------

    def shape(self) -> Tuple[int, int, int]:
        """
        Number of discrete bins in each dimension.
        """
        return (
            len(self.gap_bins) - 1,
            len(self.v_ego_bins) - 1,
            len(self.v_npc_bins) - 1,
        )

    def num_states(self) -> int:
        Ng, Nv_e, Nv_n = self.shape()
        return Ng * Nv_e * Nv_n

    # ---------- mapping continuous -> bin indices ----------

    def _bin_index(self, value: float, bins: np.ndarray) -> int:
        """
        Map value to a bin index in [0, len(bins)-2] using np.digitize,
        clamped to the valid range.
        """
        idx = np.digitize(value, bins) - 1
        idx = int(np.clip(idx, 0, len(bins) - 2))
        return idx

    def obs_to_indices(self, obs: np.ndarray) -> Tuple[int, int, int]:
        """
        obs: array-like (3,) = [gap, v_ego, v_npc]
        Returns the 3D bin indices (i_gap, i_ve, i_vn).
        """
        gap, v_ego, v_npc = [float(o) for o in obs]
        i_g = self._bin_index(gap, self.gap_bins)
        i_ve = self._bin_index(v_ego, self.v_ego_bins)
        i_vn = self._bin_index(v_npc, self.v_npc_bins)
        return i_g, i_ve, i_vn

    def values_to_indices(
        self, gap: float, v_ego: float, v_npc: float
    ) -> Tuple[int, int, int]:
        return self.obs_to_indices([gap, v_ego, v_npc])

    # ---------- pack/unpack flat state index ----------

    def indices_to_state(self, i_g: int, i_ve: int, i_vn: int) -> int:
        """
        Pack 3D indices -> flat state id.
        """
        Ng, Nv_e, Nv_n = self.shape()
        # (i_g * Nv_e + i_ve) * Nv_n + i_vn
        s = (i_g * Nv_e + i_ve) * Nv_n + i_vn
        return int(s)

    def state_to_indices(self, s: int) -> Tuple[int, int, int]:
        """
        Unpack flat state id -> 3D indices (i_gap, i_ve, i_vn).
        """
        Ng, Nv_e, Nv_n = self.shape()
        s = int(s)
        i_g = s // (Nv_e * Nv_n)
        r = s % (Nv_e * Nv_n)

        i_ve = r // Nv_n
        i_vn = r % Nv_n

        return i_g, i_ve, i_vn

    # ---------- main interface: continuous obs <-> flat state ----------

    def obs_to_state(self, obs: np.ndarray) -> int:
        """
        Continuous observation -> flat discrete state id.
        """
        i_g, i_ve, i_vn = self.obs_to_indices(obs)
        return self.indices_to_state(i_g, i_ve, i_vn)

    def values_to_state(self, gap: float, v_ego: float, v_npc: float) -> int:
        i_g, i_ve, i_vn = self.values_to_indices(gap, v_ego, v_npc)
        return self.indices_to_state(i_g, i_ve, i_vn)

    # ---------- centers & iteration helpers (for planning/precompute) ----------

    def _bin_centers(self, bins: np.ndarray) -> np.ndarray:
        return 0.5 * (bins[:-1] + bins[1:])

    def state_centers(self) -> np.ndarray:
        """
        Returns array of shape (S, 3) with the continuous center coordinates
        [gap, v_ego, v_npc] for each discrete state.
        """
        gap_centers = self._bin_centers(self.gap_bins)
        v_e_centers = self._bin_centers(self.v_ego_bins)
        v_n_centers = self._bin_centers(self.v_npc_bins)

        G, Ve, Vn = np.meshgrid(
            gap_centers, v_e_centers, v_n_centers, indexing="ij"
        )
        centers = np.stack([G, Ve, Vn], axis=-1)  # (Ng, Nv_e, Nv_n, 3)
        return centers.reshape(-1, 3)             # (S, 3)

    def index_tuples(self) -> List[Tuple[int, int, int]]:
        """
        Returns list of all (i_gap, i_ve, i_vn) index tuples for all states.
        Order matches the flat indexing.
        """
        Ng, Nv_e, Nv_n = self.shape()
        tuples = []
        for i_g in range(Ng):
            for i_ve in range(Nv_e):
                for i_vn in range(Nv_n):
                    tuples.append((i_g, i_ve, i_vn))
        return tuples

    # ---------- convenience constructor ----------

    @classmethod
    def from_ranges(
        cls,
        gap_min: float,
        gap_max: float,
        v_min: float,
        v_max: float,
        n_gap: int = 40,
        n_v_ego: int = 20,
        n_v_npc: int = 20,
    ) -> "Discretizer":
        """
        Build a uniform-grid discretizer for [gap, v_ego, v_npc] from simple ranges.
        """
        gap_bins = np.linspace(gap_min, gap_max, n_gap + 1, dtype=np.float64)
        v_ego_bins = np.linspace(v_min, v_max, n_v_ego + 1, dtype=np.float64)
        v_npc_bins = np.linspace(v_min, v_max, n_v_npc + 1, dtype=np.float64)
        return cls(gap_bins, v_ego_bins, v_npc_bins)
