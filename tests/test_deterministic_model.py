from discretizer import Discretizer

from deterministic_model import build_deterministic_model, DeterministicModelConfig

def test_deterministic_model():
  disc = Discretizer.from_ranges(
    gap_min=0.0,
    gap_max=10.0,
    v_min=-1.0,
    v_max=10.0,      # m/s
    n_gap=11,
    n_v_ego=10,
    n_v_npc=10,
  ) 

  next_state, reward, _ = build_deterministic_model(disc, 1.0, DeterministicModelConfig())
  centers = disc.state_centers()

  s = disc.obs_to_state([6.0, 6.0, 0.0])
  print(f"s: {centers[s][0]}, {centers[s][1]}, {centers[s][2]}")

  for a in range(4):
    sp = next_state[s, a]
    print(f"a: {a}, sp: {centers[sp][0]}, {centers[sp][1]}, {centers[sp][2]}")

    print(f"a: {a}, reward: {reward[s, a]}")

if __name__ == "__main__":
  test_deterministic_model()
