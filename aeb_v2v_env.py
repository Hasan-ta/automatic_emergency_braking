from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scenario_definitions import Scenario, Action, EGO_ACTION_DECEL, Vehicle, Rectangle, Vector2D, Scene
from simulation_utils import kph
from dataclasses import dataclass
from typing import Optional, Tuple
from simulation_utils import check_collision
from deterministic_model import RewardsModel, State


class EgoAgent:
    """Represents the ego vehicle controllable agent."""

    def __init__(self, init_state: Vehicle):
        self.vehicle_state = init_state
        # Negative x-acceleration for braking (m/s^2)
        self.action_to_decel = EGO_ACTION_DECEL

    @classmethod
    def from_scenario_definition(cls, scenario: Scenario) -> "EgoAgent":
        # Simple ego box: 2 m wide, 5 m long, starting near x=1.0
        state = Vehicle(
            box=Rectangle(Vector2D(1.0, 0.0), width=2.0, height=5.0),
            velocity=Vector2D(kph(scenario.subject_speed_kmh), 0.0),
            acceleration=Vector2D(0.0, 0.0),
        )
        return EgoAgent(state)

    def update(self, action: Action, dt: float):
        """Updates the vehicle state using the action provided by the policy."""

        assert isinstance(action, Action)

        self.vehicle_state.update(dt)
        # Set longitudinal acceleration from action
        ax = self.action_to_decel.get(action, 0.0)
        self.vehicle_state.acceleration.x = ax


class NPCVehicle:
    def __init__(self, vehicle_state: Vehicle):
        self.vehicle_state = vehicle_state

    @classmethod
    def from_scenario_definition(cls, scenario: Scenario) -> "NPCVehicle":
        # NPC starts ahead by headway_m
        state = Vehicle(
            box=Rectangle(Vector2D(scenario.headway_m + 1.0, 0.0), width=2.0, height=5.0),
            velocity=Vector2D(kph(scenario.lead_speed_kmh), 0.0),
            acceleration=Vector2D(-scenario.lead_decel_ms2, 0.0),
        )
        return NPCVehicle(state)

    def update(self, dt: float):
        self.vehicle_state.update(dt)


@dataclass
class Lane:
    # The starting position of the lane in the scene frame.
    start_x: float
    # The ending position of the lane in the scene frame.
    end_x: float
    # The width of the lane
    width: float


# ---------- AEBV2VEnv: Gymnasium environment ----------

class AEBV2VEnv(gym.Env):
    """
    Simple V2V AEB environment:
      - Ego follows a single lead vehicle in the same lane.
      - State (observation): [gap_m, rel_speed_mps, ego_speed_mps]
      - Actions: Action enum (Nothing, Warning, SoftBrake, StrongBrake)
      - Collision when front of ego overlaps back of lead in x (same lane in y).
    """

    metadata = {
        "render_modes": [],  # using Arcade externally for viz
    }

    def __init__(self, scenario: Scenario, rewards_model: RewardsModel, dt: float = 0.05, max_time: float = 6.0):
        super().__init__()
        self.rewards_model = rewards_model
        self.scenario = scenario
        self.dt = dt
        self.max_time = max_time

        # Scene & lane
        self.scene = Scene.from_scenario_definition(scenario)
        self.lane = Lane(start_x=0.0, end_x=self.scene.length, width=3.5)

        # Agents
        self.ego: EgoAgent = EgoAgent.from_scenario_definition(scenario)
        self.npc: NPCVehicle = NPCVehicle.from_scenario_definition(scenario)

        # Gym spaces
        self.action_space = spaces.Discrete(len(Action))

        # Observations: [x_npc_in_ego, v_ego, a_ego, v_npc]
        obs_low = np.array([0.0,    0.0,  -10.0,  0.0], dtype=np.float64)
        obs_high = np.array([150.0, 60.0, 0.0, 60.0], dtype=np.float64)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        # Internal time and done flag
        self.time: float = 0.0
        self._terminated: bool = False

        # Convenience attributes (for visualization, matching your previous code)
        self._x_s = 0.0  # ego center x
        self._x_l = 0.0  # lead center x
        self._v_s = 0.0  # ego vx
        self._a_s = 0.0  # ego ax
        self._v_l = 0.0  # lead vx

    # ----- Gym API -----

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Recreate agents fresh from scenario
        self.ego = EgoAgent.from_scenario_definition(self.scenario)
        self.npc = NPCVehicle.from_scenario_definition(self.scenario)

        self.time = 0.0
        self._terminated = False

        self._sync_cache()
        obs = self._get_obs()
        info = self._get_info(collided=False)
        return obs, info

    def step(self, action: int | Action):
        if isinstance(action, int):
            try:
                action_enum = Action(action)
            except ValueError:
                raise ValueError(f"Invalid discrete action {action}")
        else:
            action_enum = action

        if self._terminated:
            raise RuntimeError("step() called on terminated environment. Call reset().")
        
        obs_prev = self._get_obs()

        # Integrate one step
        self.ego.update(action_enum, self.dt)
        self.npc.update(self.dt)
        self.time += self.dt

        self._sync_cache()

        # Compute collision and termination
        collided = check_collision(self.ego, self.npc)
        out_of_scene = (
            self.ego.vehicle_state.box.center.x > self.scene.length
            or self.npc.vehicle_state.box.center.x < 0.0
        )

        terminated = collided or self.ego.vehicle_state.velocity.x <= 1e-5
        truncated = (self.time >= self.max_time) or out_of_scene

        self._terminated = terminated or truncated

        obs = self._get_obs()
        step_reward = self.rewards_model(State(obs_prev[0], obs_prev[1], obs_prev[2], obs_prev[3]), action, State(obs[0], obs[1], obs[2], obs[3]))

        info = self._get_info(collided=collided)

        return obs, float(step_reward), bool(terminated), bool(truncated), info

    # ----- Internals -----

    def _get_obs(self) -> np.ndarray:
        """
        Observation: [gap, v_ego, v_npc, a_npc].
        """
        ego_box = self.ego.vehicle_state.box
        npc_box = self.npc.vehicle_state.box

        # gap: front of ego to rear of lead
        ego_front = ego_box.center.x 
        lead_rear = npc_box.center.x
        gap = lead_rear - ego_front

        v_ego = self.ego.vehicle_state.velocity.x
        a_ego = self.ego.vehicle_state.acceleration.x
        v_npc = self.npc.vehicle_state.velocity.x

        obs = np.array([gap, v_ego, a_ego, v_npc], dtype=np.float64)
        return obs



    def _sync_cache(self):
        """Update convenience attributes used by external visualizers."""
        self._x_s = self.ego.vehicle_state.box.center.x
        self._x_l = self.npc.vehicle_state.box.center.x
        self._v_s = self.ego.vehicle_state.velocity.x
        self._a_s = self.ego.vehicle_state.acceleration.x
        self._v_l = self.npc.vehicle_state.velocity.x

    def _get_info(self, collided: bool) -> dict:
        ego_box = self.ego.vehicle_state.box
        npc_box = self.npc.vehicle_state.box

        x_ego = ego_box.center.x
        x_npc = npc_box.center.x
        v_ego = self.ego.vehicle_state.velocity.x
        v_npc = self.npc.vehicle_state.velocity.x

        # gap (front of ego to rear of lead) still handy for logging
        ego_front = x_ego + ego_box.height * 0.5
        lead_rear = x_npc - npc_box.height * 0.5
        gap = max(0.0, lead_rear - ego_front)

        return {
            "time_s": self.time,
            "x_ego": x_ego,
            "x_npc": x_npc,
            "v_s": v_ego,
            "v_l": v_npc,
            "gap_m": gap,
            "collided": collided,
        }



# ---------- V2V Env (with visualization) ----------
# class AEBV2VEnv(gym.Env):
#     """
#     Longitudinal V2V env for S7.3–S7.5. State: [gap, v_rel, v_ego].
#     Action: 0=strong,1=full,2=warn,3=idle. “No-contact” termination.
#     """
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

#     def __init__(self, scenario: Scenario, dt: float = 0.1):
#         super().__init__()
#         self.time = 0.0
#         self.dt = dt
#         self.scene = Scene.from_scenario_definition(scenario)
#         self.ego_agent = EgoAgent.from_scenario_definition(scenario)
#         self.npcs = [NPCVehicle.from_scenario_definition(scenario)]

#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Box(
#             # The x component of the ego agent position in the scene frame.
#             low=np.array([0.0, 0.0, 0.0], np.float32),
#             # The x component of the npc vehicle position in the scene frame.
#             high=np.array([200.0, 200.0, 80.0], np.float32),
#             dtype=np.float32,
#         )


#     def __init__(self, scenario: Scenario, dt: float = 0.1, render_mode: Optional[str] = None):
#         super().__init__()
#         assert scenario.family in {Family.V2V_STATIONARY, Family.V2V_SLOWER_MOVING, Family.V2V_DECELERATING}
#         self.sc = scenario
#         self.dt = dt
#         self.render_mode = render_mode
#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Box(
#             low=np.array([0.0, -100.0, 0.0], np.float32),
#             high=np.array([500.0, 100.0, 80.0], np.float32),
#             dtype=np.float32,
#         )
#         self._x_s = 0.0
#         self._x_l = scenario.headway_m if scenario.headway_m is not None else 30.0
#         self._v_s = kph(scenario.subject_speed_kmh)
#         self._v_l = kph(scenario.lead_speed_kmh)
#         self._t = 0.0
#         self._done = False
#         self._last_frame = None
#         self._last_a = None

#         self._simulation_params = SimulationParams()

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self._x_s = 0.0
#         self._x_l = self.sc.headway_m if self.sc.headway_m is not None else 30.0
#         self._v_s = kph(self.sc.subject_speed_kmh)
#         self._v_l = kph(self.sc.lead_speed_kmh)
#         self._t = 0.0
#         self._done = False
#         self._last_frame = None
#         self._last_a = None
#         return self._obs(), {}

#     def step(self, a: int):
#         dt = self.dt
#         a_ego = {0: -6.0, 1: -9.0, 2: 0.0, 3: 0.0}.get(int(a), None)
#         assert a_ego is not None
#         # Approx manual brake after FCW (TTC<2s), kick in after 1s
#         v_rel = self._v_l - self._v_s

#         ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
#         target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
#         ttc = compute_ttc(target, ego)
#         manual_extra = -4.0 if (self.sc.manual_brake and ttc < 2.0 and self._t > 1.0) else 0.0

#         # Lead decel for S7.5 until stop
#         a_lead = 0.0
#         if self.sc.family == Family.V2V_DECELERATING:
#             a_lead = -abs(self.sc.lead_decel_ms2 or g(0.3))
#             if self._v_l + a_lead*dt < 0:
#                 a_lead = -self._v_l/dt

#         self._v_s = max(0.0, self._v_s + (a_ego + manual_extra)*dt)
#         self._v_l = max(0.0, self._v_l + a_lead*dt)
#         self._x_s += self._v_s*dt
#         self._x_l += self._v_l*dt
#         self._t += dt

#         ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
#         target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
#         gap = compute_gap(target, ego)
#         collided = check_collision(target, ego)
#         self._done = collided or self._t > 20.0 or self._v_s <= 0.0

#         def smooth_l1_ttc(ttc, a):
#             assert a == 1 or a == 0
#             multiplier = 2 if a == 0 else 4
#             if ttc > 1.5:
#                 return multiplier*((ttc+1.5)**2)
#             else:
#                 return multiplier * (ttc+1.5)
            
#         def smooth_l1_gap_penalty(gap):
#             multiplier = 200
#             if gap < 0.5:
#                 return -1000
#             else:
#                 return gap * multiplier

#         r = 1.0
#         if self._done:
#             if collided:
#                 r = -6000.0
#             elif self._v_s <= 0.0:
#                 r = -smooth_l1_gap_penalty(gap)
#         else:
#             if a == 2:
#                 r = -10
#             else:
#                 if a == 1 or a == 0:
#                     r -= smooth_l1_ttc(ttc, a)
#         if self._last_a is not None and a != self._last_a:
#             r -= 40.0

#         self._last_a = a
#         # r = -1000.0 if collided else (1.0 - 0.05*abs(a_ego))
#         info = {"gap_m": gap, "v_s": self._v_s, "v_l": self._v_l, "collided": collided, "time_s": self._t, "ttc_s": ttc}

#         if self.render_mode == "rgb_array":
#             self._last_frame = self._draw_frame()

#         return self._obs(), float(r), self._done, False, info

#     def _obs(self):
#         ego = Actor(self._x_s, self._v_s, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
#         target = Actor(self._x_l, self._v_l, self._simulation_params.car_length_m, self._simulation_params.car_width_m)
#         gap = compute_gap(target, ego)
#         v_rel = self._v_l - self._v_s
#         return np.array([gap, v_rel, self._v_s], np.float32)
