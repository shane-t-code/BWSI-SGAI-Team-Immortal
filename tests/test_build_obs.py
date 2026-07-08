"""Parity regressions: build_obs(), TrainEnv._build_obs(), RLRecommender._build_obs().

Architecture (current):
    ``game_interfaces.obs_builder.build_obs()`` is the single source of truth for
    the 15-slot RL observation vector.  Both ``TrainEnv._build_obs()`` and
    ``RLRecommender._build_obs()`` delegate to it — there is no super() chain,
    no base_env._build_obs(), and no duplicated encoding logic.

    A mismatch between any of the three call paths is silent at runtime: the
    agent trains on one encoding but executes on a different one, producing
    wrong behavior with no error.  The load-bearing tests in this file are the
    parity regressions (``test_build_obs_matches_train_env`` and
    ``test_build_obs_matches_rl_recommender``).  If both pass, the collapse is
    intact.  If either fails, something diverged.
"""

import numpy as np
import pandas as pd
import pytest

from game_interfaces.data_parser import DataParser
from game_interfaces.obs_builder import build_obs
from game_interfaces.train_env import TrainEnv
from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper
from models.OracleClassifier import OracleClassifier
from models.RLRecommender import RLRecommender

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_TINY_DF = pd.DataFrame(
    {
        "Filename": [
            "consolidated_dataset/test_00000.png",
            "consolidated_dataset/test_00006.png",
            "consolidated_dataset/test_00007.png",
        ],
        "Class": ["Default", "Default", "Zombie"],
        "Injured": [False, True, False],
    }
)


def _make_env(
    shift_length: int = SHIFT_LENGTH, capacity: int = AMBULANCE_CAPACITY
) -> TrainEnv:
    """Build a headless TrainEnv with OracleClassifier and minimal dataset."""
    sk = ScoreKeeper(shift_length=shift_length, capacity=capacity)
    parser = DataParser(data_dir="data")
    parser.df = _TINY_DF.copy()
    parser.unvisited = set(_TINY_DF.index)
    parser.visited = []
    return TrainEnv(
        root=None,
        w=0,
        h=0,
        data_parser=parser,
        scorekeeper=sk,
        classifier=OracleClassifier(),
        display=False,
    )


# Realistic probs: predicted_class=1 (healthy), max_prob=0.8 → confidence_bin=2 (high).
# Using non-zero, non-boundary values so all encoding paths are exercised.
_PROBS = np.array([0.1, 0.8, 0.05, 0.05], dtype=np.float32)

# Humanoid with non-zero value (value=0.6 → value_bin=3 → obs[14]=0.75).
_HUMANOID = Humanoid(
    fp="consolidated_dataset/test_00000.png", state="healthy", value=0.6
)

# Varied humanoids for parametric parity tests — non-boundary values so binning is exercised.
_HUMANOIDS = [
    Humanoid(fp="consolidated_dataset/test_00000.png", state="healthy", value=0.3),
    Humanoid(fp="consolidated_dataset/test_00007.png", state="zombie", value=0.6),
    Humanoid(fp="consolidated_dataset/test_00006.png", state="injured", value=0.15),
    Humanoid(fp="consolidated_dataset/test_00004.png", state="corpse", value=0.85),
]

# ---------------------------------------------------------------------------
# Guard tests — build_obs() raises on bad inputs
# ---------------------------------------------------------------------------


def test_build_obs_raises_if_both_none() -> None:
    """build_obs(sk) raises ValueError containing 'neither humanoid nor probs'.

    Ensures that calling build_obs without a humanoid or probs produces an
    immediately actionable error message rather than a silent wrong obs.
    """
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    with pytest.raises(ValueError, match="neither humanoid nor probs"):
        build_obs(sk)


def test_build_obs_raises_if_humanoid_none() -> None:
    """build_obs(sk, probs=probs) raises ValueError containing 'probs but no humanoid'.

    Ensures that passing probs without a humanoid is caught — the value slot
    (obs[14]) requires a Humanoid instance and cannot be computed from probs alone.
    """
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    with pytest.raises(ValueError, match="probs but no humanoid"):
        build_obs(sk, probs=_PROBS)


def test_build_obs_raises_if_probs_none() -> None:
    """build_obs(sk, humanoid=h) raises ValueError containing 'humanoid but no probs'.

    Ensures that passing a humanoid without probs is caught — the classifier
    must run before _build_obs() is called.
    """
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    with pytest.raises(ValueError, match="humanoid but no probs"):
        build_obs(sk, humanoid=_HUMANOID)


# ---------------------------------------------------------------------------
# Parity regressions — build_obs() vs RLRecommender._build_obs()
# ---------------------------------------------------------------------------


def test_build_obs_matches_rl_recommender() -> None:
    """build_obs(sk, h, probs) == RLRecommender._build_obs(h, probs, sk) for identical inputs.

    This is one of the two load-bearing parity regressions.  RLRecommender._build_obs()
    delegates to build_obs(); this test confirms the delegation is correct and that
    argument reordering between the two signatures does not silently swap slots.

    Uses probs=[0.1, 0.8, 0.05, 0.05] so predicted_class=1 and confidence_bin=2
    (high), exercising the non-zero one-hot and high-confidence paths.
    Uses humanoid.value=0.6 so value_bin=3 (obs[14]=0.75), not the zero default.
    """
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    expected = build_obs(sk, _HUMANOID, _PROBS)
    actual = RLRecommender._build_obs(_HUMANOID, _PROBS, sk)
    np.testing.assert_array_equal(
        expected,
        actual,
        err_msg=(
            f"build_obs() vs RLRecommender._build_obs() mismatch.\n"
            f"build_obs:        {expected}\n"
            f"RLRecommender:    {actual}"
        ),
    )


# ---------------------------------------------------------------------------
# Parity regressions — build_obs() vs TrainEnv._build_obs()
# ---------------------------------------------------------------------------


def test_build_obs_matches_train_env() -> None:
    """build_obs(sk, h, probs) == env._build_obs() for identical inputs.

    This is the other load-bearing parity regression.  Constructs a real
    TrainEnv, overwrites env.humanoid and env.humanoid_probs with the test
    values, then calls env._build_obs() and compares element-by-element.

    Uses the same non-boundary probs and humanoid.value as the RLRecommender
    parity test so coverage is complementary, not redundant.
    """
    env = _make_env()
    env.humanoid = _HUMANOID
    env.humanoid_probs = _PROBS

    expected = build_obs(env.scorekeeper, _HUMANOID, _PROBS)
    env._build_obs()  # updates env.obs in-place
    actual = env.obs.copy()

    np.testing.assert_array_equal(
        expected,
        actual,
        err_msg=(
            f"build_obs() vs TrainEnv._build_obs() mismatch.\n"
            f"build_obs:  {expected}\n"
            f"TrainEnv:   {actual}"
        ),
    )


# ---------------------------------------------------------------------------
# Legacy parity tests — TrainEnv vs RLRecommender (parametric + edge cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("humanoid", _HUMANOIDS, ids=[h.state for h in _HUMANOIDS])
def test_build_obs_train_env_matches_rl_recommender(humanoid: Humanoid) -> None:
    """TrainEnv._build_obs() and RLRecommender._build_obs() produce identical vectors.

    Both implementations delegate to build_obs() — this parametric test checks
    that the delegation is consistent across all four humanoid states and that
    no per-class branching was accidentally introduced.

    Sets up a TrainEnv, forces it to the target humanoid, computes probs via
    OracleClassifier, then compares both obs builders element-by-element.
    """
    env = _make_env()

    env.humanoid = humanoid
    oracle = OracleClassifier()
    probs = oracle.classify(humanoid)
    env.humanoid_probs = probs

    train_env_obs = env._build_obs().copy()
    rl_recommender_obs = RLRecommender._build_obs(humanoid, probs, env.scorekeeper)

    np.testing.assert_array_almost_equal(
        train_env_obs,
        rl_recommender_obs,
        decimal=6,
        err_msg=(
            f"_build_obs() mismatch for state={humanoid.state!r}.\n"
            f"TrainEnv:      {train_env_obs}\n"
            f"RLRecommender: {rl_recommender_obs}"
        ),
    )


def test_build_obs_sync_with_partial_ambulance() -> None:
    """build_obs() agrees across all call paths when the ambulance has mixed occupancy."""
    env = _make_env()
    env.scorekeeper.ambulance["healthy"] = 2
    env.scorekeeper.ambulance["zombie"] = 1
    env.scorekeeper.remaining_time = 400

    humanoid = _HUMANOIDS[0]  # healthy
    oracle = OracleClassifier()
    probs = oracle.classify(humanoid)
    env.humanoid = humanoid
    env.humanoid_probs = probs

    train_env_obs = env._build_obs().copy()
    rl_recommender_obs = RLRecommender._build_obs(humanoid, probs, env.scorekeeper)

    np.testing.assert_array_almost_equal(train_env_obs, rl_recommender_obs, decimal=6)


def test_build_obs_sync_at_time_boundaries() -> None:
    """build_obs() agrees at shift-start and near-zero time."""
    oracle = OracleClassifier()
    humanoid = _HUMANOIDS[1]  # zombie
    probs = oracle.classify(humanoid)

    for remaining_time in [SHIFT_LENGTH, 1, 0]:
        env = _make_env()
        env.scorekeeper.remaining_time = remaining_time
        env.humanoid = humanoid
        env.humanoid_probs = probs

        train_env_obs = env._build_obs().copy()
        rl_recommender_obs = RLRecommender._build_obs(humanoid, probs, env.scorekeeper)

        np.testing.assert_array_almost_equal(
            train_env_obs,
            rl_recommender_obs,
            decimal=6,
            err_msg=f"Mismatch at remaining_time={remaining_time}",
        )


def test_build_obs_sync_at_time_boundaries_with_partial_ambulance() -> None:
    """build_obs() agrees at time boundaries when the ambulance has mixed occupancy."""
    oracle = OracleClassifier()
    humanoid = _HUMANOIDS[0]  # healthy
    probs = oracle.classify(humanoid)

    for remaining_time in [SHIFT_LENGTH, 1, 0]:
        env = _make_env()
        env.scorekeeper.remaining_time = remaining_time
        env.scorekeeper.ambulance["healthy"] = 2
        env.scorekeeper.ambulance["injured"] = 1
        env.humanoid = humanoid
        env.humanoid_probs = probs

        train_env_obs = env._build_obs().copy()
        rl_recommender_obs = RLRecommender._build_obs(humanoid, probs, env.scorekeeper)

        np.testing.assert_array_almost_equal(
            train_env_obs,
            rl_recommender_obs,
            decimal=6,
            err_msg=f"Mismatch at remaining_time={remaining_time} with partial ambulance",
        )
