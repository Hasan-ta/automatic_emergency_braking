# fmvss127_scenarios.py
# FMVSS 127 AEB scenarios + Gymnasium envs + visualization (rgb_array + Matplotlib)

from mle_mdp_planner import PolicyExecutor
import enum
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------- Helpers ----------
def kph(kmh: float) -> float: return kmh/3.6
def g(ms2: float = 1.0) -> float: return 9.80665 * ms2

@dataclass
class VehicleParams:
    v0_mps: float
    x0_m: float

# ---------- Scenario taxonomy ----------
class Family(enum.Enum):
    V2V_STATIONARY = "S7.3"
    V2V_SLOWER_MOVING = "S7.4"
    V2V_DECELERATING = "S7.5"
    PED_CROSS_RIGHT_5 = "S8.3 right 5 km/h"
    PED_CROSS_LEFT_8 = "S8.3 left 8 km/h"
    PED_CROSS_OBSTRUCTED = "S8.3 obstructed"
    PED_STATIONARY_25R = "S8.4 stationary 25% right"
    PED_ALONG_25R = "S8.5 along 25% right"

@dataclass
class Scenario:
    family: Family
    subject_speed_kmh: float
    lead_speed_kmh: float = 0.0
    lead_decel_ms2: Optional[float] = None
    headway_m: Optional[float] = None
    pedestrian_speed_kmh: Optional[float] = None
    overlap: Optional[float] = None
    daylight: Optional[bool] = True
    manual_brake: bool = False
    note: str = ""

# ---------- Scenario generator ----------
def generate_fmvss127() -> Iterable[Scenario]:
    # S7.3 Stationary lead
    for v in list(range(10, 81, 10)):
        yield Scenario(Family.V2V_STATIONARY, v, 0, manual_brake=False, note="S7.3 no manual")
    for v in [70, 80, 90, 100]:
        yield Scenario(Family.V2V_STATIONARY, v, 0, manual_brake=True, note="S7.3 manual")
    # S7.4 Slower-moving lead @20
    for v in [40, 50, 60, 70, 80]:
        yield Scenario(Family.V2V_SLOWER_MOVING, v, 20, manual_brake=False, note="S7.4 no manual")
    for v in [70, 80, 90, 100]:
        yield Scenario(Family.V2V_SLOWER_MOVING, v, 20, manual_brake=True, note="S7.4 manual")
    # S7.5 Decelerating lead: both 50 or 80; 0.3–0.5g; headway 12–40
    for v in [50, 80]:
        for hw in [12, 20, 30, 40]:
            for decel_g in [0.3, 0.4, 0.5]:
                for manual in [False, True]:
                    yield Scenario(Family.V2V_DECELERATING, v, v, lead_decel_ms2=decel_g*9.80665,
                                   headway_m=hw, manual_brake=manual, note="S7.5")
    # S8.3 Crossing pedestrian (right @5, left @8), overlaps
    for v in [10, 20, 30, 40, 50, 60]:
        for ov in [0.25, 0.50]:
            yield Scenario(Family.PED_CROSS_RIGHT_5, v, overlap=ov, pedestrian_speed_kmh=5, daylight=True)
    for v in [10, 20, 30, 40, 50, 60]:
        yield Scenario(Family.PED_CROSS_LEFT_8, v, overlap=0.50, pedestrian_speed_kmh=8, daylight=True)
    # S8.3 Obstructed crossing (two parked vehicles)
    for v in [10, 20, 30, 40, 50]:
        yield Scenario(Family.PED_CROSS_OBSTRUCTED, v, overlap=0.50, pedestrian_speed_kmh=5, daylight=True)
    # S8.4 Stationary ped @25% right, day/darkness
    for v in [10, 20, 30, 40, 50, 55]:
        for day in [True, False]:
            yield Scenario(Family.PED_STATIONARY_25R, v, overlap=0.25, daylight=day)
    # S8.5 Along-the-path (parallel)
    for v in [10, 20, 30, 40, 50, 60, 65]:
        for day in [True, False]:
            yield Scenario(Family.PED_ALONG_25R, v, pedestrian_speed_kmh=5, overlap=0.25, daylight=day)

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
        gap = max(0.0, self._x_l - self._x_s)
        ttc = gap/(-v_rel) if v_rel < 0 else np.inf
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

        gap = self._x_l - self._x_s
        collided = gap <= 5.0
        self._done = collided or self._t > 20.0 or self._v_s <= 0.0

        def smooth_l1(ttc, a):
            assert a == 1 or a == 0
            multiplier = 2 if a == 0 else 4
            if ttc > 10.0:
                return 100
            if ttc > 1.5:
                return multiplier*(ttc**2)
            else:
                return multiplier * ttc


        r = 1.0
        if collided:
            r = -6000.0
        elif a == 2:
            r = -1
        else:
            if a == 1 or a == 0:
                r -= smooth_l1(ttc, a)
        if self._last_a is not None and a != self._last_a:
            r -= 10.0

        self._last_a = a
        # r = -1000.0 if collided else (1.0 - 0.05*abs(a_ego))
        info = {"gap_m": max(0.0, gap), "v_s": self._v_s, "v_l": self._v_l, "collided": collided, "time_s": self._t}

        if self.render_mode == "rgb_array":
            self._last_frame = self._draw_frame()

        return self._obs(), float(r), self._done, False, info

    def _obs(self):
        gap = max(0.0, self._x_l - self._x_s)
        v_rel = self._v_l - self._v_s
        return np.array([gap, v_rel, self._v_s], np.float32)

    # ---- Visualization for V2V ----
    def render(self):
        if self.render_mode == "human":
            gap = max(0.0, self._x_l - self._x_s)
            print(f"[{self.sc.family.value}] t={self._t:4.1f}s | gap={gap:6.2f} m | v_s={self._v_s:5.2f} | v_l={self._v_l:5.2f}")
            return
        elif self.render_mode == "rgb_array":
            if self._last_frame is None:
                self._last_frame = self._draw_frame()
            return self._last_frame
        else:
            return

    def _draw_frame(self, width=720, height=220):
        frame = np.ones((height, width, 3), dtype=np.uint8)*240
        # Lane band
        lane_y1, lane_y2 = int(height*0.35), int(height*0.65)
        frame[lane_y1:lane_y2, :, :] = 220
        # Map meters to pixels with ego near 1/3 width
        px_per_m = 6.0
        x0 = int(width*0.33)
        gap = max(0.0, self._x_l - self._x_s)
        x_ego_px = x0
        x_lead_px = x0 + int(gap * px_per_m)
        car_len, car_w = 28, 20
        cy = height//2

        def draw_car(xc, color):
            x1 = max(0, min(width-1, xc - car_len//2))
            x2 = max(0, min(width-1, xc + car_len//2))
            y1 = max(0, cy - car_w//2)
            y2 = min(height-1, cy + car_w//2)
            frame[y1:y2, x1:x2, :] = color

        draw_car(x_ego_px, np.array([60,120,255], np.uint8))   # ego
        draw_car(x_lead_px, np.array([255,160,60], np.uint8))  # lead

        # Optional HUD text (no dependency on cv2)
        txt = f"{self.sc.family.value}  t={self._t:.1f}s  gap={gap:.1f}m  vE={self._v_s:.1f}  vL={self._v_l:.1f}"
        _put_text(frame, txt, (8, 22))
        return frame

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

# ---------- Minimal text drawing utils (no cv2 dependency) ----------
def _put_text(img, text, org, color=(25,25,25)):
    # super-lightweight text: draw as tiny blocks (monospaced-ish)
    # to avoid extra dependencies; for nicer text, install opencv-python and use cv2.putText
    x, y = org
    h = 10
    for i, ch in enumerate(text[:90]):
        # draw a tiny vertical tick per char to keep it simple
        x0 = x + i*6
        img[max(0, y-h):y, max(0, x0):min(img.shape[1], x0+1)] = color

def _rect(img, xc, yc, w, h, color):
    x1 = max(0, min(img.shape[1]-1, int(xc - w//2)))
    x2 = max(0, min(img.shape[1]-1, int(xc + w//2)))
    y1 = max(0, min(img.shape[0]-1, int(yc - h//2)))
    y2 = max(0, min(img.shape[0]-1, int(yc + h//2)))
    img[y1:y2, x1:x2, :] = color

def _disk(img, xc, yc, r, color):
    h, w, _ = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - xc)**2 + (y - yc)**2 <= r*r
    img[mask] = color

# ---------- Factory ----------
def make_env(s: Scenario, dt: float = 0.05, render_mode: Optional[str] = None) -> gym.Env:
    if s.family in {Family.V2V_STATIONARY, Family.V2V_SLOWER_MOVING, Family.V2V_DECELERATING}:
        return AEBV2VEnv(s, dt=dt, render_mode=render_mode)
    else:
        return AEBPedEnv(s, dt=dt, render_mode=render_mode)

# ---------- Matplotlib live viewer (works for both envs) ----------
def animate_env(env, policy=None, steps=300, realtime=True, stepping= False):
    """
    Live visualization using Matplotlib. Works with wrapped envs (OrderEnforcing).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    import time

    base = getattr(env, "unwrapped", env)
    obs, _ = env.reset()
    dt = getattr(base, "dt", 1.0 / env.metadata.get("render_fps", 10))

    # Default policy
    if policy is None:
        def policy(o):
            # V2V state: [gap, v_rel, v_ego]; Ped state: [dx, dy, v_ego]
            if hasattr(base, "_x_l"):  # V2V
                g, vr, ve = o
                ttc = g/(-vr) if vr < 0 else float("inf")
                return 1 if ttc < 2.0 else 3
            else:  # Ped
                dx, dy, v = o
                return 1 if (abs(dy) < 1.2 and dx < 12.0) else 3

    # --- Interactive setup (CRUCIAL) ---
    # Turn on interactive mode and show a non-blocking window
    if not plt.isinteractive():
        plt.ion()

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_aspect("equal")
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(-10, 70); ax.set_ylim(-8, 8)
    title = ax.set_title("")

    # Lane markings
    ax.plot([-10, 70], [-1.2, -1.2], "--", lw=1)
    ax.plot([-10, 70], [ 1.2,  1.2], "--", lw=1)

    # Artists
    ego_len, ego_w = 4.5, 2.0
    ego = Rectangle((-ego_len/2, -ego_w/2), ego_len, ego_w, ec="k", fc=(0.2,0.4,1.0), lw=1.5)
    ax.add_patch(ego)

    is_v2v = hasattr(base, "_x_l")
    lead = None
    ped = None
    if is_v2v:
        lead = Rectangle((10 - ego_len/2, -ego_w/2), ego_len, ego_w, ec="k", fc=(1.0,0.5,0.1), lw=1.5)
        ax.add_patch(lead)
    else:
        ped = Circle((10, 0), radius=0.3, ec="k", fc=(1.0,0.3,0.3), lw=1.2)
        ax.add_patch(ped)
        # Optional occluders for the obstructed test
        if getattr(base.sc, "family", None) and "obstructed" in base.sc.family.value.lower():
            from matplotlib.patches import Rectangle as R
            ax.add_patch(R((8, -2.5), 4.5, 2.0, ec="none", fc=(0.7,0.7,0.7)))
            ax.add_patch(R((14, -2.5), 4.5, 2.0, ec="none", fc=(0.7,0.7,0.7)))

    # Show the window NOW (non-blocking), so subsequent draws update the same figure
    try:
        plt.show(block=False)
    except TypeError:
        # Some backends ignore block kwarg; harmless.
        plt.show()

    t0 = time.time()
    for k in range(steps):
        a = policy(obs)
        obs, r, done, trunc, info = env.step(a)

        if is_v2v:
            gap_now = max(0.0, base._x_l - base._x_s)
            ego.set_xy((-ego_len/2, -ego_w/2))
            lead.set_xy((gap_now - ego_len/2, -ego_w/2))
            title.set_text(f"{base.sc.family.value}  t={info['time_s']:.2f}s  gap={gap_now:.2f}m  "
                           f"vE={info['v_s']:.1f} vL={info['v_l']:.1f}  a={a}")
        else:
            dx = base._xp - base._xe
            dy = base._yp - base._ye
            ego.set_xy((-ego_len/2, -ego_w/2))
            ped.center = (dx, dy)
            title.set_text(f"{base.sc.family.value}  t={info['time_s']:.2f}s  dx={dx:.2f} dy={dy:.2f}  "
                           f"vE={base._vxe:.1f}  a={a}")

        # --- Force a redraw each step (CRUCIAL) ---
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Small pause keeps UI responsive across backends
        import math as _m; plt.pause(1e-3)

        if realtime:
            # target = (k+1)*dt
            # lag = target - (time.time() - t0)
            # if lag > 0:
            #     time.sleep(min(lag, 0.05))

            time.sleep(0.5)

        if stepping:
            import keyboard
            keyboard.read_key()


        if done or trunc:
            if info.get("collided", False):
                title.set_text(title.get_text() + "   COLLISION!")
                fig.canvas.draw(); fig.canvas.flush_events()
                plt.pause(10)
            break

    # Leave the window open in non-interactive sessions
    if not plt.isinteractive():
        try:
            plt.ioff(); plt.show()
        except Exception:
            pass


# ---------- Tiny demo ----------
if __name__ == "__main__":
    cases = list(generate_fmvss127())

    # 1) RGB frames -> GIF (works for both env types)
    # try:
    #     import imageio.v2 as imageio
    #     # Pick one V2V and one Ped case
    #     for idx in [40]:  # adjust indices as you like
    #         # sc = cases[idx]
    #         sc = Scenario(Family.PED_CROSS_OBSTRUCTED, 20, overlap=0.50, pedestrian_speed_kmh=5, daylight=True)
    #         env = make_env(sc, dt=0.05, render_mode="rgb_array")
    #         obs, _ = env.reset(seed=0)
    #         frames = []
    #         done = False
    #         # simple policy
    #         def pol(o):
    #             if sc.family in {Family.V2V_STATIONARY, Family.V2V_SLOWER_MOVING, Family.V2V_DECELERATING}:
    #                 g, vr, ve = o; ttc = g/(-vr) if vr < 0 else np.inf; return 1 if ttc < 2.0 else 3
    #             else:
    #                 dx, dy, v = o; return 1 if (abs(dy) < 1.2 and dx < 12.0) else 3
    #         while not done:
    #             a = pol(obs)
    #             obs, r, done, trunc, info = env.step(a)
    #             frames.append(env.render())
    #             if trunc: break
    #         dt = getattr(getattr(env, "unwrapped", env), "dt", 1.0/env.metadata.get("render_fps", 10))
    #         out = f"fmvss127_{sc.family.value.replace(' ','_')}.gif"
    #         imageio.mimsave(out, frames, duration=dt)
    #         print("Saved", out)
    #         env.close()
    # except Exception as e:
    #     print("GIF demo skipped:", repr(e))

    # 2) Live Matplotlib animation on one scenario
    try:
        # sc = cases[50]  # e.g., S7.5 case
        sc = next(s for s in generate_fmvss127()
              if s.family == Family.V2V_DECELERATING and s.subject_speed_kmh == 80 and s.headway_m == 20)
        # sc = Scenario(Family.V2V_STATIONARY, 50, 0, manual_brake=False, headway_m=40, note="S7.3 no manual")
        print(sc)
        env = make_env(sc, dt=0.05)  # no render_mode; viewer handles drawing
        executor = PolicyExecutor('/Users/htafish/projects/aa228/final_project/q_function_2025_11_16_00_27_16.npy')
        animate_env(env, steps=400, realtime=True, policy=executor, stepping=False)
        env.close()
    except Exception as e:
        print("Matplotlib viewer skipped:", repr(e))
