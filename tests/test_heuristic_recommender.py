"""Tests for models/heuristic_recommender.py.

COV-07: rule coverage — at-capacity scrames, zombie prediction squishes, healthy saves.
"""

from __future__ import annotations

import numpy as np
import pytest

from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH
from gameplay.enums import ActionState
from gameplay.scorekeeper import ScoreKeeper
from models.HeuristicRecommender import HeuristicRecommender

# One-hot probs per state — slot order: zombie(0), healthy(1), injured(2), corpse(3)
_PROBS_HEALTHY = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
_PROBS_ZOMBIE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_PROBS_INJURED = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_PROBS_CORPSE = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@pytest.fixture
def sk_fresh_h() -> ScoreKeeper:
    return ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)


@pytest.fixture
def sk_at_capacity_h() -> ScoreKeeper:
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.ambulance["healthy"] = sk.capacity
    return sk


# h_healthy, h_zombie, h_injured, h_corpse are inherited from conftest.py


# ---------------------------------------------------------------------------
# COV-07 — rule coverage
# ---------------------------------------------------------------------------


def test_heuristic_recommender_scrames_when_at_capacity(sk_at_capacity_h, h_healthy):
    """recommend() returns scram when the ambulance is at capacity."""
    recommender = HeuristicRecommender()
    action = recommender.recommend(h_healthy, _PROBS_HEALTHY, sk_at_capacity_h)
    assert action == ActionState.SCRAM


def test_heuristic_recommender_squishes_zombie(sk_fresh_h, h_zombie):
    """recommend() returns squish when classifier predicts zombie."""
    recommender = HeuristicRecommender()
    action = recommender.recommend(h_zombie, _PROBS_ZOMBIE, sk_fresh_h)
    assert action == ActionState.SQUISH


def test_heuristic_recommender_squishes_corpse(sk_fresh_h, h_corpse):
    """recommend() returns squish when classifier predicts corpse."""
    recommender = HeuristicRecommender()
    action = recommender.recommend(h_corpse, _PROBS_CORPSE, sk_fresh_h)
    assert action == ActionState.SQUISH


def test_heuristic_recommender_saves_healthy(sk_fresh_h, h_healthy):
    """recommend() returns save when classifier predicts healthy."""
    recommender = HeuristicRecommender()
    action = recommender.recommend(h_healthy, _PROBS_HEALTHY, sk_fresh_h)
    assert action == ActionState.SAVE


def test_heuristic_recommender_saves_injured(sk_fresh_h, h_injured):
    """recommend() returns save when classifier predicts injured."""
    recommender = HeuristicRecommender()
    action = recommender.recommend(h_injured, _PROBS_INJURED, sk_fresh_h)
    assert action == ActionState.SAVE
