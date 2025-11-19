from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scenario_definitions import Scenario, Family, Actor, SimulationParams
from simulation_utils import kph, compute_gap, check_collision, compute_ttc


# ---------- V2V Env (with visualization) ----------
class AEBV2VEnv(gym.Env):
    """
    Longitudinal V2V env for S7.3–S7.5. State: [gap, v_rel, v_ego].
    Action: 0=strong,1=full,2=warn,3=idle. “No-contact” termination.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, scenario: Scenario, dt: float = 0.1, render_mode: Optional[str] = None):
        super().__init__()
        assert scenario.family in {Family.V2V_STATIONARY, Family.V2V_SLOWER_MOVING, Family.V2V_DECELERATING}
        self.sc = scenario
        self.dt = dt
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -100.0, 0.0], np.float32),
            high=np.array([500.0, 100.0, 80.0], np.float32),
            dtype=np.float32,
        )
        self._x_s = 0.0
        self._x_l = scenario.headway_m if scenario.headway_m is not None else 30.0
        self._v_s = kph(scenario.subject_speed_kmh)
        self._v_l = kph(scenario.lead_speed_kmh)
        self._t = 0.0
        self._done = False
        self._last_frame = None
        self._last_a = None

        self._simulation_params = SimulationParams()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._x_s = 0.0
        self._x_l = self.sc.headway_m if self.sc.headway_m is not None else 30.0
        self._v_s = kph(self.sc.subject_speed_kmh)
        self._v_l = kph(self.sc.lead_speed_kmh)
        self._t = 0.0
        self._done = False
        self._last_frame = None
        self._last_a = None
        return self._obs(), {}

    def step(self, a: int):
        dt = self.dt
        a_ego = {0: -6.0, 1: -9.0, 2: 0.0, 3: 0.0}.get(int(a), None)
        assert a_ego is not None
        # Approx manual brake after FCW (TTC<2s), kick in after 1s
        v_rel = self._v_l - self._v_s

        ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
        target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
        ttc = compute_ttc(target, ego)
        manual_extra = -4.0 if (self.sc.manual_brake and ttc < 2.0 and self._t > 1.0) else 0.0

        # Lead decel for S7.5 until stop
        a_lead = 0.0
        if self.sc.family == Family.V2V_DECELERATING:
            a_lead = -abs(self.sc.lead_decel_ms2 or g(0.3))
            if self._v_l + a_lead*dt < 0:
                a_lead = -self._v_l/dt

        self._v_s = max(0.0, self._v_s + (a_ego + manual_extra)*dt)
        self._v_l = max(0.0, self._v_l + a_lead*dt)
        self._x_s += self._v_s*dt
        self._x_l += self._v_l*dt
        self._t += dt

        ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
        target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
        gap = compute_gap(target, ego)
        collided = check_collision(target, ego)
        self._done = collided or self._t > 20.0 or self._v_s <= 0.0

        def smooth_l1_ttc(ttc, a):
            assert a == 1 or a == 0
            multiplier = 1 if a == 0 else 4
            if ttc > 1.5:
                return multiplier*(ttc**2)
            else:
                return multiplier * ttc
            
        def smooth_l1_gap_penalty(gap):
            multiplier = 200
            if gap < 0.3:
                return -100
            if gap < 1.5:
                return (gap+2)**2 * multiplier
            # if gap > 1.5:
            #     return gap**2 * multiplier
            return gap * multiplier

        r = 1.0
        if self._done:
            if collided:
                r = -6000.0
            elif self._v_s <= 0.0:
                r = -smooth_l1_gap_penalty(gap)
        else:
            if a == 2:
                r = -10
            else:
                if a == 1 or a == 0:
                    r -= smooth_l1_ttc(ttc, a)
        # if self._last_a is not None and a != self._last_a:
        #     r -= 10.0

        self._last_a = a
        # r = -1000.0 if collided else (1.0 - 0.05*abs(a_ego))
        info = {"gap_m": gap, "v_s": self._v_s, "v_l": self._v_l, "collided": collided, "time_s": self._t, "ttc_s": ttc}

        if self.render_mode == "rgb_array":
            self._last_frame = self._draw_frame()

        return self._obs(), float(r), self._done, False, info

    def _obs(self):
        ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
        target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
        gap = compute_gap(target, ego)
        v_rel = self._v_l - self._v_s
        return np.array([gap, v_rel, self._v_s], np.float32)

    # ---- Visualization for V2V ----
    # def render(self):
    #     if self.render_mode == "human":
    #         ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
    #         target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
    #         gap = compute_gap(target, ego)
    #         print(f"[{self.sc.family.value}] t={self._t:4.1f}s | gap={gap:6.2f} m | v_s={self._v_s:5.2f} | v_l={self._v_l:5.2f}")
    #         return
    #     elif self.render_mode == "rgb_array":
    #         if self._last_frame is None:
    #             self._last_frame = self._draw_frame()
    #         return self._last_frame
    #     else:
    #         return

    # def _draw_frame(self, width=720, height=220):
    #     frame = np.ones((height, width, 3), dtype=np.uint8)*240
    #     # Lane band
    #     lane_y1, lane_y2 = int(height*0.35), int(height*0.65)
    #     frame[lane_y1:lane_y2, :, :] = 220
    #     # Map meters to pixels with ego near 1/3 width
    #     px_per_m = 6.0
    #     x0 = int(width*0.33)
    #     ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
    #     target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
    #     gap = compute_gap(target, ego)
    #     ttc = compute_ttc(target, ego)
    #     x_ego_px = x0
    #     x_lead_px = x0 + int(gap+target.length * px_per_m)
    #     car_len, car_w = int(ego.length * px_per_m), int(ego.width*px_per_m)
    #     cy = height//2

    #     def draw_car(xc, color):
    #         x1 = max(0, min(width-1, xc - car_len//2))
    #         x2 = max(0, min(width-1, xc + car_len//2))
    #         y1 = max(0, cy - car_w//2)
    #         y2 = min(height-1, cy + car_w//2)
    #         frame[y1:y2, x1:x2, :] = color

    #     draw_car(x_ego_px, np.array([60,120,255], np.uint8))   # ego
    #     draw_car(x_lead_px, np.array([255,160,60], np.uint8))  # lead

    #     # Optional HUD text (no dependency on cv2)
    #     txt = f"{self.sc.family.value}  t={self._t:.1f}s  gap={gap:.1f}m  vE={self._v_s:.1f}  vL={self._v_l:.1f}, ttc={ttc:.2f}"
    #     put_text(frame, txt, (8, 22))
    #     return frame