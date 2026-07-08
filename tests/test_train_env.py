"""Tests for game_interfaces/train_env.py.

COV-01: step() branch coverage (blocked action, terminal step, normal step).
COV-02: reset() contract (shape, one-hot sum, previous_cum_reward).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from game_interfaces.data_parser import DataParser
from game_interfaces.train_env import TrainEnv
from gameplay.constants import (
    AMBULANCE_CAPACITY,
    BLOCKED_ACTION_PENALTY,
    CLASSIFIER_MODEL_PATH,
    SHIFT_LENGTH,
)
from gameplay.scorekeeper import ScoreKeeper
from models.CNNClassifier import CNNClassifier

# ---------------------------------------------------------------------------
# Tiny 3-row dataset — avoids loading the full real dataset for unit tests
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
    """Build a headless TrainEnv backed by the tiny 3-row dataset."""
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
        classifier=CNNClassifier(
            model_file=CLASSIFIER_MODEL_PATH, img_data_root="data"
        ),
        display=False,
    )


# ---------------------------------------------------------------------------
# COV-01 — step() branch coverage
# ---------------------------------------------------------------------------


def test_step_blocked_action_returns_penalty_reward():
    """step(save) when ambulance is at capacity returns reward == -0.5, done=False."""
    env = _make_env(capacity=1)
    # Fill the ambulance directly so save is blocked
    env.scorekeeper.ambulance["healthy"] = 1

    obs, reward, done, _, _ = env.step(0)  # idx 0 = save

    assert reward == pytest.approx(BLOCKED_ACTION_PENALTY)
    assert done is False


def test_step_terminal_step_sets_done_true():
    """step() on the last humanoid in the pool sets episode_done=True."""
    env = _make_env()
    # Force exactly 1 item in pool, consume it, then force empty so step sees done
    env.data_parser.unvisited = set(list(_TINY_DF.index)[:1])
    env.data_parser.visited = []
    env.get_humanoid()  # consume the single remaining item
    env.data_parser.unvisited = set()  # force pool empty

    _obs, _reward, done, _trunc, _info = env.step(2)  # skip — cheap, never blocked
    assert done is True


def test_step_normal_step_obs_shape_and_range():
    """Normal step returns obs shape (15,), all values in [0, 1], done=False."""
    env = _make_env()
    obs, _reward, done, _trunc, _info = env.step(
        2
    )  # skip — cheap, never blocked on a fresh env

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (15,)
    assert float(obs.min()) >= 0.0
    assert float(obs.max()) <= 1.0
    assert done is False


# ---------------------------------------------------------------------------
# COV-02 — reset() contract
# ---------------------------------------------------------------------------


def test_reset_returns_array_shape_15():
    """reset() returns (obs, info) where obs is ndarray of shape (15,)."""
    env = _make_env()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (15,)
    assert isinstance(info, dict)


def test_reset_humanoid_class_one_hot():
    """Slots obs[5:9] sum to 1.0 after reset (one-hot humanoid class)."""
    env = _make_env()
    obs, _ = env.reset()
    assert float(obs[5:9].sum()) == pytest.approx(1.0)


def test_reset_previous_cum_reward_is_zero():
    """env.previous_cum_reward == 0 immediately after reset()."""
    env = _make_env()
    # Take a step to dirty the state
    env.step(2)
    env.reset()
    assert env.previous_cum_reward == 0


# ---------------------------------------------------------------------------
# Guard rails — step() called in the wrong state
# ---------------------------------------------------------------------------


def test_step_raises_before_reset():
    """step() raises RuntimeError when humanoid is None (reset() never called)."""
    env = _make_env()
    env.humanoid = None  # simulate pre-reset state
    with pytest.raises(RuntimeError, match="reset"):
        env.step(2)


def test_step_raises_after_episode_done():
    """step() raises RuntimeError when the episode is already over."""
    env = _make_env()
    # Exhaust the pool so the next step returns done=True
    env.data_parser.unvisited = set(list(_TINY_DF.index)[:1])
    env.data_parser.visited = []
    env.get_humanoid()
    env.data_parser.unvisited = set()
    _obs, _reward, done, _trunc, _info = env.step(2)
    assert done is True
    # Episode is done — a second step must raise
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step(2)


# ---------------------------------------------------------------------------
# BaseEnv abstract contract
# ---------------------------------------------------------------------------


def test_base_env_step_raises_not_implemented():
    """GameEnv.step() raises NotImplementedError — subclasses must override."""
    from game_interfaces.base_env import GameEnv

    env = object.__new__(GameEnv)
    with pytest.raises(NotImplementedError):
        env.step(0)
