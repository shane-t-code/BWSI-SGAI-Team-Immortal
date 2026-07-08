"""Tests for models/OracleClassifier.py.

Contracts tested:
- classify() returns a correct one-hot for each state (the abstract primitive)
- probs_to_class() via module utility returns ground-truth state and confidence==1.0
"""

import numpy as np
import pytest

from gameplay.enums import State
from gameplay.humanoid import MAP_CLASS_STR_TO_INT, Humanoid
from models.HumanoidClassifier import probs_to_class
from models.OracleClassifier import OracleClassifier

_FP_ZOMBIE = "consolidated_dataset/test_00007.png"
_FP_HEALTHY = "consolidated_dataset/test_00000.png"
_FP_INJURED = "consolidated_dataset/test_00006.png"
_FP_CORPSE = "consolidated_dataset/test_00004.png"

_ALL_STATES = [
    ("zombie", _FP_ZOMBIE),
    ("healthy", _FP_HEALTHY),
    ("injured", _FP_INJURED),
    ("corpse", _FP_CORPSE),
]


# ---------------------------------------------------------------------------
# classify() — the abstract primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state,fp", _ALL_STATES)
def test_oracle_classify_is_correct_one_hot(state, fp):
    """classify() returns a one-hot with 1.0 at the correct state index."""
    h = Humanoid(fp=fp, state=state)
    classifier = OracleClassifier()
    probs = classifier.classify(h)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (4,)
    assert float(probs.sum()) == pytest.approx(1.0)
    assert set(probs.tolist()).issubset({0.0, 1.0})
    assert probs[MAP_CLASS_STR_TO_INT[state]] == pytest.approx(1.0)


@pytest.mark.parametrize("state,fp", _ALL_STATES)
def test_oracle_classify_other_indices_are_zero(state, fp):
    """classify() has 0.0 at every index except the correct state index."""
    h = Humanoid(fp=fp, state=state)
    classifier = OracleClassifier()
    probs = classifier.classify(h)

    correct_idx = MAP_CLASS_STR_TO_INT[state]
    for i, p in enumerate(probs):
        if i == correct_idx:
            assert p == pytest.approx(1.0)
        else:
            assert p == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# probs_to_class() — returns ground-truth label and confidence==1.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state,fp", _ALL_STATES)
def test_oracle_probs_to_class_returns_correct_state(state, fp):
    """probs_to_class(classify()) returns the ground-truth state."""
    h = Humanoid(fp=fp, state=state)
    classifier = OracleClassifier()
    probs = classifier.classify(h)
    label, confidence = probs_to_class(probs)
    assert label == State(state)
    assert confidence == pytest.approx(1.0)


@pytest.mark.parametrize("state,fp", _ALL_STATES)
def test_oracle_confidence_is_always_one(state, fp):
    """Oracle confidence is exactly 1.0 regardless of humanoid state."""
    h = Humanoid(fp=fp, state=state)
    classifier = OracleClassifier()
    probs = classifier.classify(h)
    _, confidence = probs_to_class(probs)
    assert confidence == pytest.approx(1.0)
