"""Unit tests for models/rl_recommender.py.

Contracts tested:
- _build_obs() return type, shape, dtype, value range
- Slot 0: time_bin encoding at full shift and half shift
- Slots 1-4: per-class ambulance counts normalized by capacity
- Slots 5-8: one-hot of CNN argmax prediction
- Slot 9: confidence bin validity (one of {0.0, 0.5, 1.0})
- Slots 10-13: match scorekeeper.available_action_space()
- Slot 10 (save) is 0.0 when ambulance is at capacity
- Slot 14: humanoid_value_bin encoding at boundary and mid-range values
- recommend() return type (ActionState), action validity

All obs tests call _build_obs() directly — recommend() is only tested for
its own output contract, not used to inspect internal encoding.
"""

import numpy as np
import pytest

from gameplay.enums import ActionState
from gameplay.scorekeeper import ScoreKeeper

_VALID_ACTIONS = {"save", "squish", "skip", "scram"}


# ---------------------------------------------------------------------------
# _build_obs — return type
# ---------------------------------------------------------------------------


def test_build_obs_returns_numpy_array(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """_build_obs() returns a numpy ndarray."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    assert isinstance(obs, np.ndarray)


# ---------------------------------------------------------------------------
# _build_obs — shape
# ---------------------------------------------------------------------------


def test_build_obs_shape_is_15(rl_recommender, h_healthy, probs_healthy, sk_fresh):
    """_build_obs() returns an array of shape (15,)."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    assert obs.shape == (15,)


# ---------------------------------------------------------------------------
# _build_obs — dtype
# ---------------------------------------------------------------------------


def test_build_obs_dtype_is_float32(rl_recommender, h_healthy, probs_healthy, sk_fresh):
    """_build_obs() returns a float32 array."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# _build_obs — all values in [0, 1]
# ---------------------------------------------------------------------------


def test_build_obs_all_values_in_0_1(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """Every element of the observation vector is in [0.0, 1.0]."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    assert float(obs.min()) >= 0.0
    assert float(obs.max()) <= 1.0


# ---------------------------------------------------------------------------
# Slot 0 — time_bin
# ---------------------------------------------------------------------------


def test_build_obs_slot_0_time_bin_full_shift(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """Slot 0 is 1.0 when remaining_time equals shift_length (720/720, bin=4/4)."""
    # sk_fresh starts with remaining_time == shift_length == 720
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    assert obs[0] == pytest.approx(1.0)


def test_build_obs_slot_0_time_bin_half_shift(
    rl_recommender, h_healthy, probs_healthy, sk_half_time
):
    """Slot 0 is 0.5 when remaining_time is 360 (fraction=0.5, bin=min(4,2)=2, 2/4=0.5)."""
    # sk_half_time has remaining_time = 360, shift_length = 720
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_half_time)
    assert obs[0] == pytest.approx(0.5)


def test_build_obs_slot_0_time_bin_boundary_pins_5_bin_formula(
    rl_recommender, h_healthy, probs_healthy
):
    """Slot 0 boundary values pin the 5-bin, 4-divisor formula uniquely.

    Formula: min(4, int(fraction * 5)) / 4.0
    Probes at bin crossings to rule out 2-bin, 3-bin, or 4-bin alternatives.
    """
    from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH
    from gameplay.scorekeeper import ScoreKeeper

    def _obs_slot0(fraction: float) -> float:
        sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
        sk.remaining_time = int(fraction * SHIFT_LENGTH)
        obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk)
        return float(obs[0])

    # Bin 0: fraction < 0.2  → int(fraction*5)=0 → 0/4=0.0
    assert _obs_slot0(0.10) == pytest.approx(0.0)
    # Bin 1: 0.2 ≤ fraction < 0.4 → int(fraction*5)=1 → 1/4=0.25
    assert _obs_slot0(0.25) == pytest.approx(0.25)
    # Bin 2: 0.4 ≤ fraction < 0.6 → int(fraction*5)=2 → 2/4=0.5
    assert _obs_slot0(0.50) == pytest.approx(0.5)
    # Bin 3: 0.6 ≤ fraction < 0.8 → int(fraction*5)=3 → 3/4=0.75
    assert _obs_slot0(0.70) == pytest.approx(0.75)
    # Bin 4: fraction ≥ 0.8 → min(4, int(fraction*5)=4)=4 → 4/4=1.0
    assert _obs_slot0(0.90) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Slots 1-4 — per-class ambulance counts
# ---------------------------------------------------------------------------


def test_build_obs_slots_1_to_4_empty_ambulance(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """Slots 1-4 are all 0.0 when the ambulance is empty."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    np.testing.assert_array_equal(obs[1:5], np.zeros(4, dtype=np.float32))


def test_build_obs_slots_1_to_4_at_capacity(
    rl_recommender, h_healthy, probs_healthy, sk_at_capacity
):
    """Slots 1–4 reflect ambulance occupancy fractions at capacity (mixed: 8 healthy, 2 injured)."""
    # sk_at_capacity: ambulance["healthy"] = 8, ambulance["injured"] = 2, capacity = 10
    # Slot layout: 1=zombie, 2=healthy, 3=injured, 4=corpse
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_at_capacity)
    assert obs[1] == pytest.approx(0.0)  # zombie
    assert obs[2] == pytest.approx(0.8)  # healthy — 8/10
    assert obs[3] == pytest.approx(0.2)  # injured — 2/10
    assert obs[4] == pytest.approx(0.0)  # corpse


# ---------------------------------------------------------------------------
# Slots 5-8 — one-hot of CNN argmax
# ---------------------------------------------------------------------------


def test_build_obs_slots_5_to_8_one_hot(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """Exactly one of slots 5-8 is 1.0 and the remaining three are 0.0."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    one_hot_block = obs[5:9]
    assert float(one_hot_block.sum()) == pytest.approx(1.0)
    assert set(one_hot_block.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Slot 9 — confidence bin
# ---------------------------------------------------------------------------


def test_build_obs_slot_9_confidence_bin_is_valid(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """Slot 9 is one of {0.0, 0.5, 1.0} (confidence_bin in {0,1,2} divided by 2)."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    assert obs[9] in {0.0, 0.5, 1.0}


# ---------------------------------------------------------------------------
# Slots 10-13 — available action space
# ---------------------------------------------------------------------------


def test_build_obs_slots_10_to_13_match_available_action_space(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """Slots 10-13 exactly match the bools from scorekeeper.available_action_space()."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_fresh)
    expected = np.array(sk_fresh.available_action_space(), dtype=np.float32)
    np.testing.assert_array_equal(obs[10:14], expected)


def test_build_obs_slots_1_to_4_mixed_load_pins_slot_mapping(
    rl_recommender, h_healthy, probs_healthy
):
    """Slots 1-4 encode zombie/healthy/injured/corpse in exactly that order.

    Uses a mixed ambulance to verify the label-to-slot correspondence for
    zombie (slot 1) and injured (slot 3) — not just healthy (slot 2).
    """
    from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH
    from gameplay.scorekeeper import ScoreKeeper

    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.ambulance["zombie"] = 2
    sk.ambulance["healthy"] = 0
    sk.ambulance["injured"] = 3
    sk.ambulance["corpse"] = 0
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk)
    assert obs[1] == pytest.approx(2 / AMBULANCE_CAPACITY)  # zombie
    assert obs[2] == pytest.approx(0.0)  # healthy
    assert obs[3] == pytest.approx(3 / AMBULANCE_CAPACITY)  # injured
    assert obs[4] == pytest.approx(0.0)  # corpse


def test_build_obs_slots_10_to_13_save_false_at_capacity(
    rl_recommender, h_healthy, probs_healthy, sk_at_capacity
):
    """Slot 10 (save) is 0.0 when the ambulance is at full capacity."""
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk_at_capacity)
    assert obs[10] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recommend() — output contract
# ---------------------------------------------------------------------------


def test_recommend_returns_action_state(
    rl_recommender, h_healthy, probs_healthy, sk_fresh
):
    """recommend() returns an ActionState enum member."""
    result = rl_recommender.recommend(h_healthy, probs_healthy, sk_fresh)
    assert isinstance(result, ActionState)


def test_recommend_action_is_valid(rl_recommender, h_healthy, probs_healthy, sk_fresh):
    """The action returned by recommend() has a valid .value string."""
    action = rl_recommender.recommend(h_healthy, probs_healthy, sk_fresh)
    assert action.value in _VALID_ACTIONS


# ---------------------------------------------------------------------------
# COV-14-rl — FileNotFoundError at construction
# ---------------------------------------------------------------------------


def test_rl_recommender_raises_on_missing_weights():
    """RLRecommender raises FileNotFoundError when the RL weights file does not exist."""
    from models.RLRecommender import RLRecommender

    with pytest.raises(FileNotFoundError):
        RLRecommender(rl_model_file="no_such_file.pth")


# ---------------------------------------------------------------------------
# COV-15 — time_bin clamp at negative remaining_time
# ---------------------------------------------------------------------------


def test_build_obs_clamps_negative_remaining_time(
    rl_recommender, h_healthy, probs_healthy
):
    """Slot 0 is 0.0 when remaining_time is negative (clamp to 0 before binning)."""
    from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH

    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.remaining_time = -50
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk)
    assert obs[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# COV-16 — confidence bin boundaries (probs passed directly, no monkeypatching)
# ---------------------------------------------------------------------------


def test_build_obs_confidence_bin_low(rl_recommender, h_healthy, sk_fresh):
    """obs[9] == 0.0 when max CNN probability is below CNN_CONFIDENCE_MID (bin 0)."""
    low_probs = np.array([0.3, 0.25, 0.25, 0.20], dtype=np.float32)
    obs = rl_recommender._build_obs(h_healthy, low_probs, sk_fresh)
    assert obs[9] == pytest.approx(0.0)


def test_build_obs_confidence_bin_mid(rl_recommender, h_healthy, sk_fresh):
    """obs[9] == 0.5 when max CNN probability is in [CNN_CONFIDENCE_MID, CNN_CONFIDENCE_HIGH)."""
    mid_probs = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32)
    obs = rl_recommender._build_obs(h_healthy, mid_probs, sk_fresh)
    assert obs[9] == pytest.approx(0.5)


def test_build_obs_confidence_bin_high(rl_recommender, h_healthy, sk_fresh):
    """obs[9] == 1.0 when max CNN probability is >= CNN_CONFIDENCE_HIGH."""
    high_probs = np.array([0.85, 0.1, 0.03, 0.02], dtype=np.float32)
    obs = rl_recommender._build_obs(h_healthy, high_probs, sk_fresh)
    assert obs[9] == pytest.approx(1.0)


def test_build_obs_confidence_bin_mid_exact_boundary(
    rl_recommender, h_healthy, sk_fresh
):
    """obs[9] == 0.5 at exactly CNN_CONFIDENCE_MID (0.5) — boundary is inclusive (>=)."""
    boundary_probs = np.array([0.5, 0.2, 0.15, 0.15], dtype=np.float32)
    obs = rl_recommender._build_obs(h_healthy, boundary_probs, sk_fresh)
    assert obs[9] == pytest.approx(0.5)


def test_build_obs_confidence_bin_high_exact_boundary(
    rl_recommender, h_healthy, sk_fresh
):
    """obs[9] == 1.0 at exactly CNN_CONFIDENCE_HIGH (0.75) — boundary is inclusive (>=)."""
    boundary_probs = np.array([0.75, 0.15, 0.05, 0.05], dtype=np.float32)
    obs = rl_recommender._build_obs(h_healthy, boundary_probs, sk_fresh)
    assert obs[9] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# COV-26 — squish/skip slots blocked at time=0
# ---------------------------------------------------------------------------


def test_build_obs_squish_skip_blocked_at_time_zero(
    rl_recommender, h_healthy, probs_healthy
):
    """obs[11] (squish) and obs[12] (skip) are 0.0 when remaining_time=0."""
    from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH

    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.remaining_time = 0
    obs = rl_recommender._build_obs(h_healthy, probs_healthy, sk)
    assert obs[11] == pytest.approx(0.0)  # squish
    assert obs[12] == pytest.approx(0.0)  # skip


# ---------------------------------------------------------------------------
# COV-27 — recommend with at-capacity scorekeeper
# ---------------------------------------------------------------------------


def test_recommend_returns_valid_action_at_capacity(
    rl_recommender, h_healthy, probs_healthy, sk_at_capacity
):
    """recommend() returns a valid ActionState even when ambulance is at capacity."""
    action = rl_recommender.recommend(h_healthy, probs_healthy, sk_at_capacity)
    assert isinstance(action, ActionState)
    assert action.value in _VALID_ACTIONS


def test_recommend_never_returns_save_when_at_capacity(
    rl_recommender, h_healthy, probs_healthy, sk_at_capacity
):
    """recommend() never returns SAVE when the ambulance is at capacity.

    Calls recommend() 20 times to rule out lucky argmax coincidences — without the
    action mask, the DQN would return SAVE whenever it has the highest Q-value,
    which is the common case for a save-biased policy.
    """
    for _ in range(20):
        action = rl_recommender.recommend(h_healthy, probs_healthy, sk_at_capacity)
        assert (
            action != ActionState.SAVE
        ), "recommend() returned SAVE while ambulance is at capacity"


# ---------------------------------------------------------------------------
# Slot [14] — humanoid_value_bin
# ---------------------------------------------------------------------------


def test_build_obs_value_bin_slot(rl_recommender, probs_healthy, sk_fresh):
    """value=0.73 → bin=3 ('HIGH'), obs[14] == 3/4.0 == 0.75."""
    from gameplay.humanoid import Humanoid as _H

    h = _H(fp="consolidated_dataset/test_00000.png", state="healthy", value=0.73)
    obs = rl_recommender._build_obs(h, probs_healthy, sk_fresh)
    assert obs[14] == pytest.approx(3 / 4.0)


def test_build_obs_value_bin_very_low(rl_recommender, probs_healthy, sk_fresh):
    """value=0.1 → bin=0 ('VERY LOW'), obs[14] == 0.0."""
    from gameplay.humanoid import Humanoid as _H

    h = _H(fp="consolidated_dataset/test_00000.png", state="healthy", value=0.1)
    obs = rl_recommender._build_obs(h, probs_healthy, sk_fresh)
    assert obs[14] == pytest.approx(0.0)


def test_build_obs_value_bin_boundary(rl_recommender, probs_healthy, sk_fresh):
    """value=1.0 tests the clamp guard: int(1.0*5)=5, min(5,4)=4, obs[14]=1.0.

    value=1.0 is outside the range DataParser can produce ([0.0, 1.0)), but it
    is the only input that exercises the min(..., 4) clamp in the bin formula.
    Without the clamp, value=1.0 would produce bin index 5 — an out-of-bounds
    write. This test is intentionally out-of-range to verify that guard.
    """
    from gameplay.humanoid import Humanoid as _H

    h = _H(fp="consolidated_dataset/test_00000.png", state="healthy", value=1.0)
    obs = rl_recommender._build_obs(h, probs_healthy, sk_fresh)
    assert obs[14] == pytest.approx(1.0)
