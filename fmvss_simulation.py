# fmvss127_scenarios.py
# FMVSS 127 AEB scenarios + Gymnasium envs + visualization (rgb_array + Matplotlib)

import argparse
from policy_executor import PolicyExecutor
from scenario_definitions import Scenario, Family, SimulationParams
from scenario_factory import make_env
from dataclasses import dataclass

@dataclass
class ReplayParameters:
    frame_delay: float = 0.05

# ---------- Matplotlib live viewer (works for both envs) ----------
def animate_env(env, replay_parameters, policy=None, steps=300, realtime=True, stepping= False):
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
    # ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(-10, 70); ax.set_ylim(-8, 8)
    title = ax.set_title("")

    # Lane markings
    ax.plot([-10, 70], [-1.2, -1.2], "--", lw=1)
    ax.plot([-10, 70], [ 1.2,  1.2], "--", lw=1)

    # Artists
    params = SimulationParams()
    ego_len, ego_w = params.car_length_m, params.car_width_m
    ego = Rectangle((base._x_s - ego_len/2, -ego_w/2), ego_len, ego_w, ec="k", fc=(0.2,0.4,1.0), lw=1.5)
    ax.add_patch(ego)

    is_v2v = hasattr(base, "_x_l")
    lead = None
    ped = None
    if is_v2v:
        lead = Rectangle((base._x_l - ego_len/2, -ego_w/2), ego_len, ego_w, ec="k", fc=(1.0,0.5,0.1), lw=1.5)
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
    accumulative_rewards = 0.0
    action_text = ax.text(30, 7.0, "", fontsize=12, horizontalalignment='center', verticalalignment='center')
    for k in range(steps):
        a = policy(obs)
        obs, r, done, trunc, info = env.step(a)
        accumulative_rewards += r

        if a == 2:
            action_text.set_text('Warning') 
            action_text.set_color('yellow')
        if a == 0:
            action_text.set_text('Soft Brake') 
            action_text.set_color('orange')
        if a == 1:
            action_text.set_text('Strong Brake') 
            action_text.set_color('red')
        if a == 3:
            action_text.set_text('') 

        if is_v2v:
            gap_now = info["gap_m"]
            ego.set_xy((base._x_s -ego_len/2, -ego_w/2))
            lead.set_xy((base._x_l - ego_len/2, -ego_w/2))
            ttc = info["ttc_s"]
            title.set_text(f"{base.sc.family.value}  t={info['time_s']:.2f}s  gap={gap_now:.2f}m  "
                           f"vE={info['v_s']:.1f} vL={info['v_l']:.1f}  a={a}, ttc={ttc:.2f}")
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
            target = (k+1)*dt
            lag = target - (time.time() - t0)
            if lag > 0:
                time.sleep(min(lag, 0.05))

            time.sleep(replay_parameters.frame_delay)

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
    # cases = list(generate_fmvss127())

    parser = argparse.ArgumentParser(description="Evaluate policy on FMVSS policies.")
    parser.add_argument("--policy", required=True, help="The policy file")
    parser.add_argument("--frame_delay", type=float, default=0.05, help="Delay between frames")
    args = parser.parse_args()

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
        # sc = next(s for s in generate_fmvss127()
        #       if s.family == Family.V2V_DECELERATING and s.subject_speed_kmh == 80 and s.headway_m == 20)
        # sc = Scenario(Family.V2V_STATIONARY, 50, 0, manual_brake=False, headway_m=40, note="S7.3 no manual")
        sc = Scenario(family=Family.V2V_STATIONARY, subject_speed_kmh=10, lead_speed_kmh=0, lead_decel_ms2=None, headway_m=16.66668, pedestrian_speed_kmh=None, overlap=None, daylight=True, manual_brake=False, note='S7.3 no manual')
        print(sc)
        env = make_env(sc, dt=0.05)  # no render_mode; viewer handles drawing
        executor = PolicyExecutor(args.policy)
        replay_parameters = ReplayParameters(args.frame_delay)
        animate_env(env, replay_parameters, steps=400, realtime=True, policy=executor, stepping=False)
        env.close()
    except Exception as e:
        print("Matplotlib viewer skipped:", repr(e))
