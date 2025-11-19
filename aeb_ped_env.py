from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scenario_definitions import Scenario, Family

# ---------- Pedestrian Env (with visualization) ----------
class AEBPedEnv(gym.Env):
    """
    2D ego + pedestrian mannequin. Ego moves along x; ped crosses or parallels per scenario.
    “No-contact” termination.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, scenario: Scenario, dt: float = 0.1, render_mode: Optional[str] = None):
        super().__init__()
        self.sc = scenario
        assert scenario.family in {
            Family.PED_CROSS_RIGHT_5, Family.PED_CROSS_LEFT_8,
            Family.PED_CROSS_OBSTRUCTED, Family.PED_STATIONARY_25R, Family.PED_ALONG_25R
        }
        self.dt = dt
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        # obs = [ego_x_to_ped, ego_y_to_ped, ego_vx]
        self.observation_space = spaces.Box(
            low=np.array([-200.0, -10.0, 0.0], np.float32),
            high=np.array([ 200.0,  10.0, 80.0], np.float32),
            dtype=np.float32,
        )
        # Ego starts at (0,0) heading +x
        self._xe = 0.0; self._ye = 0.0; self._vxe = kph(scenario.subject_speed_kmh)
        self._xp, self._yp = self._spawn_ped()
        self._vxp, self._vyp = self._ped_velocity()
        self._t = 0.0
        self._last_frame = None

    def _spawn_ped(self) -> Tuple[float,float]:
        if self.sc.family in {Family.PED_CROSS_RIGHT_5, Family.PED_CROSS_OBSTRUCTED}:
            return (25.0, +4.0)
        if self.sc.family == Family.PED_CROSS_LEFT_8:
            return (25.0, -6.0)
        if self.sc.family == Family.PED_STATIONARY_25R:
            return (30.0, +0.5)
        if self.sc.family == Family.PED_ALONG_25R:
            return (15.0, +0.5)
        return (25.0, +4.0)

    def _ped_velocity(self) -> Tuple[float,float]:
        if self.sc.family == Family.PED_CROSS_RIGHT_5:
            return (0.0, -kph(self.sc.pedestrian_speed_kmh or 5.0))
        if self.sc.family == Family.PED_CROSS_LEFT_8:
            return (0.0, +kph(self.sc.pedestrian_speed_kmh or 8.0))
        if self.sc.family == Family.PED_CROSS_OBSTRUCTED:
            return (0.0, -kph(self.sc.pedestrian_speed_kmh or 5.0))
        if self.sc.family == Family.PED_STATIONARY_25R:
            return (0.0, 0.0)
        if self.sc.family == Family.PED_ALONG_25R:
            return (kph(self.sc.pedestrian_speed_kmh or 5.0), 0.0)
        return (0.0, 0.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._xe = 0.0; self._ye = 0.0; self._vxe = kph(self.sc.subject_speed_kmh)
        self._xp, self._yp = self._spawn_ped()
        self._vxp, self._vyp = self._ped_velocity()
        self._t = 0.0
        self._last_frame = None
        return self._obs(), {}

    def step(self, a: int):
        dt = self.dt
        a_ego = {0: -6.0, 1: -9.0, 2: 0.0, 3: 0.0}.get(int(a), 0.0)

        # Integrate ego
        self._vxe = max(0.0, self._vxe + a_ego*dt)
        self._xe += self._vxe*dt

        # Integrate pedestrian
        self._xp += self._vxp*dt
        self._yp += self._vyp*dt
        self._t += dt

        # Contact model: car rectangle (L=4.5m, W=2.0m) vs ped point
        half_w = 1.0
        collided = (abs(self._yp - self._ye) <= half_w) and (0.0 <= (self._xp - self._xe) <= 4.5)
        done = collided or self._t > 12.0 or (self._xp - self._xe) < -5.0 or abs(self._yp) > 8.0
        r = -1000.0 if collided else (1.0 - 0.05*abs(a_ego))
        info = {"collided": collided, "dx": self._xp - self._xe, "dy": self._yp - self._ye, "time_s": self._t}

        if self.render_mode == "rgb_array":
            self._last_frame = self._draw_frame()

        return self._obs(), float(r), done, False, info

    def _obs(self):
        return np.array([self._xp - self._xe, self._yp - self._ye, self._vxe], np.float32)

    # ---- Visualization for Ped ----
    def render(self):
        if self.render_mode == "human":
            print(f"[{self.sc.family.value}] t={self._t:4.1f}s | dx={self._xp - self._xe:6.2f} | dy={self._yp - self._ye:5.2f} | vE={self._vxe:5.2f}")
            return
        elif self.render_mode == "rgb_array":
            if self._last_frame is None:
                self._last_frame = self._draw_frame()
            return self._last_frame
        else:
            return

    def _draw_frame(self, width=720, height=300):
        frame = np.ones((height, width, 3), dtype=np.uint8)*242
        # Ground/lane
        lane_y1, lane_y2 = int(height*0.35), int(height*0.65)
        frame[lane_y1:lane_y2, :, :] = 220
        # World->pixel mapping (keep ego near 1/3)
        px_per_m = 10.0
        x0 = int(width*0.30)
        y0 = height//2
        # Ego rectangle
        xe_px = x0
        ye_px = y0
        ped_dx = self._xp - self._xe
        ped_dy = self._yp - self._ye
        xp_px = x0 + int(ped_dx*px_per_m)
        yp_px = y0 - int(ped_dy*px_per_m)

        # Draw ego car (4.5m x 2.0m)
        car_len_px, car_w_px = int(4.5*px_per_m), int(2.0*px_per_m)
        _rect(frame, xe_px, ye_px, car_len_px, car_w_px, color=(60,120,255))
        # Pedestrian (circle)
        _disk(frame, xp_px, yp_px, r=int(0.3*px_per_m), color=(255,80,80))

        # Occluders for obstructed crossing: two parked rectangles to the right shoulder
        if self.sc.family == Family.PED_CROSS_OBSTRUCTED:
            occ_w, occ_h = int(4.5*px_per_m), int(2.0*px_per_m)
            # parked cars at +x ahead of ego, off to the right side
            _rect(frame, x0 + int(8*px_per_m), y0 - int(2.5*px_per_m), occ_w, occ_h, color=(180,180,180))
            _rect(frame, x0 + int(14*px_per_m), y0 - int(2.5*px_per_m), occ_w, occ_h, color=(180,180,180))

        txt = f"{self.sc.family.value}  t={self._t:.1f}s  dx={ped_dx:.1f}m dy={ped_dy:.1f}m vE={self._vxe:.1f}"
        _put_text(frame, txt, (8, 22))
        return frame