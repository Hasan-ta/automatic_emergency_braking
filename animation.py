import arcade
import gymnasium as gym
import numpy as np
import time
import matplotlib.pylab as plt

from scenario_definitions import Scenario, Family, Actor, Action
from scenario_factory import make_env
from simulation_utils import compute_ttc, compute_gap
from policy_executor import PolicyExecutor, NaivePolicyExecutor
from discretizer import Discretizer, GapEgoAccelDiscretizer
from deterministic_model import RewardsModel, DeterministicModelConfig
from global_config import DiscretizerConfig, SimulationConfig
from q_learning import DQNInference
from noisy_obs_wrapper import NoisyObsWrapper

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors for each action
ACTION_COLORS = {
    Action.SoftBrake: arcade.color.RED,          # strong brake
    Action.StrongBrake: arcade.color.DARK_RED,     # full brake
    Action.Warning: arcade.color.GOLD,         # warning
    Action.Nothing: arcade.color.LIGHT_GRAY,   # idle
}

def draw_filled_rect(x, y, w, h, color):
    arcade.draw_rect_filled(arcade.XYWH(x, y, w, h), color)

def draw_outline_rect_lrtb(left, right, top, bottom, color, border_width=1):
    # If your version also changed the outline API, use LRTB similarly:
    # arcade.draw_rect_outline(arcade.LRTB(left, right, top, bottom), color, border_width)
    arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, color, border_width)


class AEBArcadeViewer(arcade.Window):
    def __init__(self, env: gym.Env, policy, scale_m_to_px=8.0, max_history=400, target_fps=60):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "AEB Simulation (Arcade)")
        arcade.set_background_color(arcade.color.ASH_GREY)

        self.env = env
        self.policy = policy
        self.m_to_px = scale_m_to_px

        self.obs, _ = self.env.reset()
        self.done = False
        self.last_action = 3
        self.tt = 0.0
        self._collided = False
        self._accumulated_reward = 0.0

        # --- histories for plotting ---
        self.max_history = max_history
        self.times = []
        self.v_ego_hist = []
        self.v_lead_hist = []
        self.a_ego_hist = []
        self.action_hist = []
        self.ttc_hist = []

        # --- real-time control ---
        base = self.env.unwrapped
        self.env_dt = getattr(base, "dt", 0.05)  # fallback if env has no dt
        self.sim_speed = 1.0                     # 1.0 = real time, 0.5 slow, 2.0 fast
        self._sim_time_accum = 0.0
        self._last_nonzero_speed = 1.0          # remember for pause toggling

        # optional: fix update rate (Arcade 3.x)
        self.set_update_rate(1.0 / target_fps)

    # ---- public control helpers ----
    def set_speed(self, factor: float):
        self.sim_speed = max(0.0, min(factor, 16.0))
        if self.sim_speed > 0:
            self._last_nonzero_speed = self.sim_speed

    def pause(self):
        self.sim_speed = 0.0

    def unpause(self):
        self.sim_speed = max(self._last_nonzero_speed, 0.25)

    def toggle_pause(self):
        if self.sim_speed == 0.0:
            self.unpause()
        else:
            self.pause()

    def reset_env(self):
        self.obs, _ = self.env.reset()
        self.done = False
        self.last_action = 3
        self.tt = 0.0
        self.times.clear()
        self.v_ego_hist.clear()
        self.a_ego_hist.clear()
        self.v_lead_hist.clear()
        self.action_hist.clear()
        self.ttc_hist.clear()
        self._sim_time_accum = 0.0
        self._collided = False
        self._accumulated_reward = 0.0

    # ---- stepping logic ----
    def _step_env_once(self):
        """Advance env by exactly one env_dt of simulation time."""
        if self.done:
            return

        a = self.policy(self.obs)
        self.last_action = a

        self.obs, r, done, trunc, info = self.env.step(a)
        self._accumulated_reward += r
        base = self.env.unwrapped

        self.done = bool(done or trunc)
        self.tt = info.get("time_s", self.tt + self.env_dt)
        self._collided = info["collided"]

        # speeds
        # v_ego = getattr(base, "_v_s", np.nan)   # adjust names
        # v_lead = getattr(base, "_v_l", np.nan)
        v_ego = self.obs[1]
        v_lead = self.obs[3]

        # histories
        self.times.append(self.tt)
        self.v_ego_hist.append(v_ego)
        self.v_lead_hist.append(v_lead)
        self.a_ego_hist.append(self.obs[2])
        self.action_hist.append(a)
        base = self.env.unwrapped
        x_ego_m = getattr(base, "_x_s", 0.0)
        # x_lead_m = getattr(base, "_x_l", 30.0)
        x_lead_m = self.obs[0] + x_ego_m
        actor2 = Actor(x_ego_m, v_ego, length=5.0, width=2.0)
        actor1 = Actor(x_lead_m, v_lead, length=5.0, width=2.0)
        self.ttc_hist.append(min(compute_ttc(actor1, actor2), 10.0))

        # trim
        if len(self.times) > self.max_history:
            self.times = self.times[-self.max_history:]
            self.v_ego_hist = self.v_ego_hist[-self.max_history:]
            self.v_lead_hist = self.v_lead_hist[-self.max_history:]
            self.action_hist = self.action_hist[-self.max_history:]

    def on_update(self, delta_time: float):
        # accumulate real-time * speed
        if self.done or self.sim_speed == 0.0:
            return

        self._sim_time_accum += delta_time * self.sim_speed

        # step env enough times to keep up with (scaled) real time
        # limit substeps to avoid spiraling if sim is very slow
        max_substeps = 10
        steps = 0
        while self._sim_time_accum >= self.env_dt and steps < max_substeps and not self.done:
            self._sim_time_accum -= self.env_dt
            self._step_env_once()
            steps += 1

    def on_draw(self):
        self.clear()

        lane_y = SCREEN_HEIGHT // 2 + 80
        lane_height = 60
        draw_filled_rect(
            SCREEN_WIDTH // 2, lane_y, SCREEN_WIDTH, lane_height,
            arcade.color.LIGHT_SLATE_GRAY,
        )

        base = self.env.unwrapped
        x_ego_m = getattr(base, "_x_s", 0.0)
        # x_lead_m = getattr(base, "_x_l", 30.0)
        x_lead_m = self.obs[0] + x_ego_m

        v_ego = self.obs[1]
        v_lead = self.obs[3]

        actor2 = Actor(x_ego_m, v_ego, length=5.0, width=2.0)
        actor1 = Actor(x_lead_m, v_lead, length=5.0, width=2.0)
        gap = compute_gap(actor1, actor2)
        ttc = compute_ttc(actor1, actor2)

        x0_px = SCREEN_WIDTH * 0.30
        ego_x_px = x0_px
        lead_x_px = x0_px + (gap + 5.0) * self.m_to_px

        car_length_px = 5.0 * self.m_to_px
        car_width_px = 2.0 * self.m_to_px

        draw_filled_rect(
            ego_x_px, lane_y,
            car_length_px, car_width_px,
            arcade.color.BLUE,
        )
        draw_filled_rect(
            lead_x_px, lane_y,
            car_length_px, car_width_px,
            arcade.color.ORANGE,
        )

        # HUD
        v_ego = self.v_ego_hist[-1] if self.v_ego_hist else 0.0
        v_lead = self.v_lead_hist[-1] if self.v_lead_hist else 0.0

        speed_label = "paused" if self.sim_speed == 0.0 else f"{self.sim_speed:.2f}x"

        info_text = (
            f"t = {self.tt:.2f}s   gap = {gap:.1f}m   "
            f"v_ego = {v_ego:.1f} m/s   v_lead = {v_lead:.1f} m/s   "
            f"action = {Action(self.last_action).name}   "
            f"sim_speed = {speed_label} "
        )

        reward_text = f"ttc: {ttc}s, Rewards= {self._accumulated_reward} "
        if self._collided:
            reward_text += " COLLISION!!"

        arcade.draw_text(info_text, 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 12)
        arcade.draw_text(reward_text, 10, SCREEN_HEIGHT - 60, arcade.color.BLACK, 12)

        if self.done:
            arcade.draw_text(
                "DONE (R=reset, SPACE=pause/unpause)",
                SCREEN_WIDTH // 2 - 180, SCREEN_HEIGHT // 2 + 80,
                arcade.color.RED, 16,
            )

        if len(self.times) > 1:
            self._draw_speed_plot()
            self._draw_action_plot()
            self._draw_ttc_plot()

    # --- speed plot (unchanged except for vertical placement) ---
    def _draw_speed_plot(self):
        margin = 40
        plot_left = margin
        plot_right = SCREEN_WIDTH - margin
        plot_bottom = 1 + 140 + 100
        plot_top = 1 + plot_bottom + 80

        draw_outline_rect_lrtb(
            plot_left, plot_right, plot_top, plot_bottom,
            arcade.color.DARK_SLATE_GRAY, border_width=1,
        )
        arcade.draw_text("Speed [m/s]", plot_left, plot_top + 5, arcade.color.BLACK, 10)

        t = np.array(self.times)
        v_e = np.array(self.v_ego_hist)
        v_l = np.array(self.v_lead_hist)

        t0, t1 = t[0], t[-1] if t[-1] > t[0] else (t[0] + 1e-6)
        tn = (t - t0) / (t1 - t0 + 1e-9)

        vmax = float(np.nanmax([np.nanmax(v_e), np.nanmax(v_l), 1.0]))
        vmin = 0.0
        vn_e = (v_e - vmin) / (vmax - vmin + 1e-9)
        vn_l = (v_l - vmin) / (vmax - vmin + 1e-9)

        def to_px(tx, vy):
            x = plot_left + tx * (plot_right - plot_left)
            y = plot_bottom + vy * (plot_top - plot_bottom)
            return x, y

        for i in range(1, len(tn)):
            x1, y1 = to_px(tn[i-1], vn_e[i-1])
            x2, y2 = to_px(tn[i],   vn_e[i])
            arcade.draw_line(x1, y1, x2, y2, arcade.color.BLUE, 2)

        for i in range(1, len(tn)):
            x1, y1 = to_px(tn[i-1], vn_l[i-1])
            x2, y2 = to_px(tn[i],   vn_l[i])
            arcade.draw_line(x1, y1, x2, y2, arcade.color.ORANGE, 2)

    def _draw_ttc_plot(self):
        margin = 40
        plot_left = margin
        plot_right = SCREEN_WIDTH - margin
        plot_bottom = 10
        plot_top = 1 + plot_bottom + 80

        draw_outline_rect_lrtb(
            plot_left, plot_right, plot_top, plot_bottom,
            arcade.color.DARK_SLATE_GRAY, border_width=1,
        )
        arcade.draw_text("ttc [s]", plot_left, plot_top + 5, arcade.color.BLACK, 10)

        t = np.array(self.times)
        ttc = np.array(self.ttc_hist)

        t0, t1 = t[0], t[-1] if t[-1] > t[0] else (t[0] + 1e-6)
        tn = (t - t0) / (t1 - t0 + 1e-9)

        ttcmax = float(max(np.nanmax(ttc), 20.0))
        ttcmin = 0.0
        ttc = (ttc - ttcmin) / (ttcmax - ttcmin + 1e-9)

        def to_px(tx, ttcy):
            x = plot_left + tx * (plot_right - plot_left)
            y = plot_bottom + ttcy * (plot_top - plot_bottom)
            return x, y

        for i in range(1, len(tn)):
            x1, y1 = to_px(tn[i-1], ttc[i-1])
            x2, y2 = to_px(tn[i],   ttc[i])
            arcade.draw_line(x1, y1, x2, y2, arcade.color.BLUE, 2)

    def _draw_action_plot(self):
        if not self.action_hist:
            return

        margin = 40
        plot_left = margin + 1
        plot_right = SCREEN_WIDTH - margin + 1
        plot_bottom = 10 + 100
        plot_top = plot_bottom + 80

        draw_outline_rect_lrtb(
            plot_left, plot_right, plot_top, plot_bottom,
            arcade.color.DARK_SLATE_GRAY, border_width=1,
        )
        arcade.draw_text("Actions", plot_left, plot_top + 5, arcade.color.BLACK, 10)

        t = np.array(self.times)
        # a = np.array(self.action_hist, dtype=int)

        t0, t1 = t[0], t[-1] if t[-1] > t[0] else (t[0] + 1e-6)
        tn = (t - t0) / (t1 - t0 + 1e-9)

        n_actions = 4
        band_height = (plot_top - plot_bottom) / n_actions

        def to_px_band(tx, action_idx):
            x = plot_left + tx * (plot_right - plot_left)
            y_center = plot_bottom + (action_idx + 0.5) * band_height
            return x, y_center

        # bar_width = max(2, (plot_right - plot_left) / len(tn))
        bar_width = 2
        for i in range(len(tn)):
            act = self.action_hist[i].value
            x_center, y_center = to_px_band(tn[i], act)
            color = ACTION_COLORS.get(self.action_hist[i], arcade.color.BLACK)
            draw_filled_rect(
                x_center, y_center,
                bar_width, band_height * 0.8,
                color,
            )

    # --- input handling: pause + speed controls + reset + single-step ---
    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.R:
            self.reset_env()
        elif symbol == arcade.key.SPACE:
            self.toggle_pause()
        elif symbol in (arcade.key.EQUAL, arcade.key.PLUS):  # '+' or '='
            # double the speed
            self.set_speed(self.sim_speed * 2.0 if self.sim_speed > 0 else self._last_nonzero_speed * 2.0)
        elif symbol in (arcade.key.MINUS, arcade.key.UNDERSCORE):
            # halve the speed (but not below 0.125x unless paused)
            if self.sim_speed == 0.0:
                self.set_speed(self._last_nonzero_speed / 2.0)
            else:
                self.set_speed(max(self.sim_speed / 2.0, 0.125))
        elif symbol == arcade.key.RIGHT:
            # single-step when paused
            if self.sim_speed == 0.0 and not self.done:
                self._step_env_once()

def main():

    v_kph = 30
    h_m = v_kph * 0.277778 * 5.0
    def get_headway(speed):
        return speed * 0.277778 * 6.0
    
    sc = Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=30, lead_speed_kmh=0.0, lead_decel_ms2=0.0, headway_m=83.33340000000001, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual')
    sim_config = SimulationConfig()
    env = make_env(sc, RewardsModel(DeterministicModelConfig()), dt=sim_config.dt, max_time=sim_config.total_time)  # your custom env

    # Per-dimension noise std for [gap, v_ego, a_ego, v_npc]
    sigma = np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float64)
    env = NoisyObsWrapper(env, sigma=sigma, clip=True, seed=123)

    disc_config = DiscretizerConfig()
    disc = GapEgoAccelDiscretizer.from_ranges(
        gap_min=disc_config.gap_min,
        gap_max=disc_config.gap_max,
        v_min=disc_config.v_min,
        v_max=disc_config.v_max,      # m/s
        a_min=disc_config.a_min,
        a_max=disc_config.a_max,      # m/s
        n_gap=disc_config.n_gap,
        n_v_ego=disc_config.n_v_ego,
        n_a_ego=disc_config.n_a_ego,
        n_v_npc=disc_config.n_v_npc,
    ) 
    executor = PolicyExecutor(disc, '/Users/htafish/projects/aa228/final_project/q_deterministic_planner.npy')
    # executor = DQNInference("aeb_dqn_qnet.pt")
    # executor = DQNInference("belief_aeb_dqn_qnet.pt")
    # executor = NaivePolicyExecutor()
    window = AEBArcadeViewer(env, policy=executor, scale_m_to_px=8.0)
    arcade.run()

    _, axes = plt.subplots(2, 1)
    axes[0].plot(window.v_ego_hist)
    axes[0].plot(window.v_lead_hist)
    axes[0].plot(window.ttc_hist)
    axes[1].plot(window.a_ego_hist)
    plt.show()

if __name__ == "__main__":
    main()
