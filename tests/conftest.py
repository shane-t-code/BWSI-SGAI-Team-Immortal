"""Shared pytest fixtures for bwsi-sgai tests.

All imports resolve from the project root via ``pythonpath = ["."]`` in
``pyproject.toml``.  Model fixtures are session-scoped so weights load once
per test run; everything else is function-scoped (isolated per test).
"""

import numpy as np
import pytest

from game_interfaces.data_parser import DataParser
from gameplay.constants import (
    AMBULANCE_CAPACITY,
    CLASSIFIER_MODEL_PATH,
    DATA_DIR,
    RL_MODEL_PATH,
    SHIFT_LENGTH,
)
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper
from models.CNNClassifier import CNNClassifier
from models.RLRecommender import RLRecommender

# ---------------------------------------------------------------------------
# Real image paths drawn from data/consolidated_metadata.csv.
# These are paths *as stored in the CSV* — relative to img_data_root ("data").
# Verified present at data/consolidated_dataset/test_0000{x}.png.
# ---------------------------------------------------------------------------
_FP_ZOMBIE = "consolidated_dataset/test_00007.png"
_FP_HEALTHY = "consolidated_dataset/test_00000.png"
_FP_INJURED = "consolidated_dataset/test_00006.png"
_FP_CORPSE = "consolidated_dataset/test_00004.png"


# ---------------------------------------------------------------------------
# ScoreKeeper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sk_fresh() -> ScoreKeeper:
    """Return a brand-new ScoreKeeper at full time with an empty ambulance.

    Returns:
        ScoreKeeper(shift_length=720, capacity=10) in its initial state.
    """
    return ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)


@pytest.fixture
def sk_half_time() -> ScoreKeeper:
    """Return a ScoreKeeper with exactly half the shift remaining.

    State: remaining_time=360, empty ambulance, zero score.

    Returns:
        ScoreKeeper with remaining_time forced to 360.
    """
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.remaining_time = 360
    return sk


@pytest.fixture
def sk_at_capacity() -> ScoreKeeper:
    """Return a ScoreKeeper whose ambulance is completely full.

    The ambulance dict is populated directly (10 healthy passengers) rather
    than calling ``save()`` 10 times so no time is consumed and no log
    entries are created.

    Returns:
        ScoreKeeper with ambulance["healthy"] == capacity (10/10 occupied).
    """
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.ambulance["healthy"] = sk.capacity - 2
    sk.ambulance["injured"] = 2
    return sk


@pytest.fixture
def sk_near_end() -> ScoreKeeper:
    """Return a ScoreKeeper with only 5 minutes left on the clock.

    At this point remaining_time (5) equals SQUISH cost (5 min) and is less
    than SKIP (15 min) and SAVE (30 min). Scram is always available.
    Scram is always available per design (never gated on time).  Use this
    fixture to test boundary behavior near time exhaustion.

    Returns:
        ScoreKeeper with remaining_time forced to 5.
    """
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.remaining_time = 5
    return sk


# ---------------------------------------------------------------------------
# Humanoid fixtures — one per state
# ---------------------------------------------------------------------------


@pytest.fixture
def h_zombie() -> Humanoid:
    """Return a Humanoid whose ground-truth state is 'zombie'.

    The fp points to a real image in data/consolidated_dataset/ so any
    test that opens the file will succeed.

    Returns:
        Humanoid with state="zombie" and a valid image path.
    """
    return Humanoid(fp=_FP_ZOMBIE, state="zombie")


@pytest.fixture
def h_healthy() -> Humanoid:
    """Return a Humanoid whose ground-truth state is 'healthy'.

    Returns:
        Humanoid with state="healthy" and a valid image path.
    """
    return Humanoid(fp=_FP_HEALTHY, state="healthy")


@pytest.fixture
def h_injured() -> Humanoid:
    """Return a Humanoid whose ground-truth state is 'injured'.

    Returns:
        Humanoid with state="injured" and a valid image path.
    """
    return Humanoid(fp=_FP_INJURED, state="injured")


@pytest.fixture
def h_corpse() -> Humanoid:
    """Return a Humanoid whose ground-truth state is 'corpse'.

    Returns:
        Humanoid with state="corpse" and a valid image path.
    """
    return Humanoid(fp=_FP_CORPSE, state="corpse")


@pytest.fixture
def humanoid_nonzero_value() -> Humanoid:
    """Return a healthy Humanoid with a non-zero, non-boundary value (0.7).

    Use this fixture in value-bin tests that need to verify mid-range encoding
    without hitting input bin boundaries (0.2, 0.4, 0.6, 0.8).

    Returns:
        Humanoid with state="healthy", value=0.7, and a valid image path.
    """
    return Humanoid(fp=_FP_HEALTHY, state="healthy", value=0.7)


# ---------------------------------------------------------------------------
# Probs fixture — fixed one-hot for a healthy humanoid
# ---------------------------------------------------------------------------


@pytest.fixture
def probs_healthy() -> np.ndarray:
    """Return a one-hot probability vector for a healthy humanoid.

    Slot order matches MAP_CLASS_INT_TO_STR: zombie(0), healthy(1), injured(2), corpse(3).

    Returns:
        float32 array [0.0, 1.0, 0.0, 0.0].
    """
    return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# DataParser fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def data_parser() -> DataParser:
    """Return a DataParser pointed at the real data directory.

    Function-scoped so each test gets a fresh unvisited pool — no test can
    exhaust the pool and affect another test's draw.

    Returns:
        DataParser(data_dir="data") with a fully reset unvisited pool.
    """
    return DataParser(data_dir=DATA_DIR)


# ---------------------------------------------------------------------------
# Model fixtures — session-scoped (weights load once per test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cnn_classifier() -> CNNClassifier:
    """Return a CNNClassifier loaded with the baseline CNN weights.

    Session-scoped: weights are loaded once and the same instance is reused
    across all tests in the session.  Do not mutate this object in tests.

    Returns:
        CNNClassifier with real weights and img_data_root="data".
    """
    return CNNClassifier(model_file=CLASSIFIER_MODEL_PATH, img_data_root=DATA_DIR)


@pytest.fixture(scope="session")
def rl_recommender() -> RLRecommender:
    """Return an RLRecommender loaded with the trained DQN weights.

    Session-scoped: the model file is loaded once per test run.
    Do not mutate this object in tests. Probs are passed in at the call
    site — this recommender does not own a classifier.

    Returns:
        RLRecommender with real RL weights.
    """
    return RLRecommender(rl_model_file=RL_MODEL_PATH)
