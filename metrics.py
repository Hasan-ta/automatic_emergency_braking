import numpy as np

def compute_discomfort(
    action_int: int,
    a_ego: float,
    a_ego_prev: float,
    dt: float,
    prev_action_int: int | None = None,
    # action weights
    w_warning: float = 0.2,
    w_soft: float = 1.0,
    w_strong: float = 2.0,
    # accel/jerk weights
    k_a: float = 0.02,
    k_j: float = 0.005,
    # accel exponent
    accel_power: float = 2.0,
    # switch penalty
    w_switch: float = 0.15,
) -> np.float64:
    """
    Combined passenger discomfort per step.

    Depends on:
      1) action annoyance      (warning/soft/strong)
      2) physical decel cost   from a_ego
      3) jerk cost             from delta a_ego / dt
      4) number of switches    (penalty if prev_action != action)

    Assumes action encoding:
      0 = Nothing
      1 = Warning
      2 = SoftBrake
      3 = StrongBrake

    Returns:
      per-step discomfort as np.float64
    """
    # Ensure float64 scalars
    a_ego = np.float64(a_ego)
    a_ego_prev = np.float64(a_ego_prev)
    dt = np.float64(dt)

    # ---------- 1) action annoyance ----------
    d_action = np.float64(0.0)
    if action_int == 1:
        d_action = np.float64(w_warning)
    elif action_int == 2:
        d_action = np.float64(w_soft)
    elif action_int == 3:
        d_action = np.float64(w_strong)

    # ---------- 2) decel discomfort ----------
    # only penalize negative accel (braking)
    decel_mag = np.maximum(np.float64(0.0), -a_ego)
    d_accel = np.float64(k_a) * np.power(decel_mag, np.float64(accel_power))

    # ---------- 3) jerk discomfort ----------
    # guard dt
    if dt > 0:
        jerk = (a_ego - a_ego_prev) / dt
    else:
        jerk = np.float64(0.0)
    d_jerk = np.float64(k_j) * np.abs(jerk)

    # ---------- 4) switch discomfort ----------
    d_switch = np.float64(0.0)
    if prev_action_int is not None and int(prev_action_int) != int(action_int):
        d_switch = np.float64(w_switch)

    return np.float64(d_action + d_accel + d_jerk + d_switch)


class DiscomfortTracker:
    """
    Convenience stateful tracker.
    Keeps cumulative discomfort and switch count.

    You can store one instance in the env and update every step.
    """

    def __init__(
        self,
        dt: float,
        init_action_int: int = 0,
        init_a_ego: float = 0.0,
        **kwargs
    ):
        self.dt = np.float64(dt)
        self.prev_action_int = int(init_action_int)
        self.prev_a_ego = np.float64(init_a_ego)
        self.cumulative = np.float64(0.0)
        self.switches = 0
        self.kwargs = kwargs

    def reset(self, action_int: int = 0, a_ego: float = 0.0):
        self.prev_action_int = int(action_int)
        self.prev_a_ego = np.float64(a_ego)
        self.cumulative = np.float64(0.0)
        self.switches = 0

    def step(self, action_int: int, a_ego: float) -> np.float64:
        action_int = int(action_int)
        a_ego = np.float64(a_ego)

        d = compute_discomfort(
            action_int=action_int,
            a_ego=a_ego,
            a_ego_prev=self.prev_a_ego,
            dt=self.dt,
            prev_action_int=self.prev_action_int,
            **self.kwargs
        )

        if action_int != self.prev_action_int:
            self.switches += 1

        self.cumulative = np.float64(self.cumulative + d)

        self.prev_action_int = action_int
        self.prev_a_ego = a_ego

        return d
