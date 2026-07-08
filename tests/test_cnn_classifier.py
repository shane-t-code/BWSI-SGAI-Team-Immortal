"""Unit tests for models/CNNClassifier.py.

Contracts tested:
- classify() return type, shape, dtype, sums-to-one
- classify() runs without error on all four humanoid states
- probs_to_class() returns valid (State, float) from classify() output
- FileNotFoundError at construction when weights are missing
"""

import numpy as np
import pytest

from gameplay.enums import State
from gameplay.humanoid import Humanoid
from models.HumanoidClassifier import probs_to_class

# Valid state labels — derived from the enum so this stays in sync automatically
_VALID_STATES = set(
    Humanoid.get_all_states()
)  # {"zombie", "healthy", "injured", "corpse"}


# ---------------------------------------------------------------------------
# classify() — the abstract primitive
# ---------------------------------------------------------------------------


def test_classify_returns_ndarray(cnn_classifier, h_healthy):
    """classify() returns a numpy ndarray."""
    probs = cnn_classifier.classify(h_healthy)
    assert isinstance(probs, np.ndarray)


def test_classify_shape_is_4(cnn_classifier, h_healthy):
    """classify() returns an array of shape (4,)."""
    probs = cnn_classifier.classify(h_healthy)
    assert probs.shape == (4,)


def test_classify_sums_to_one(cnn_classifier, h_healthy):
    """classify() output sums to approximately 1.0 (softmax output)."""
    probs = cnn_classifier.classify(h_healthy)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_classify_on_zombie_image(cnn_classifier, h_zombie):
    """classify() returns ndarray without error on a zombie image."""
    probs = cnn_classifier.classify(h_zombie)
    assert probs.shape == (4,)


def test_classify_on_healthy_image(cnn_classifier, h_healthy):
    """classify() returns ndarray without error on a healthy image."""
    probs = cnn_classifier.classify(h_healthy)
    assert probs.shape == (4,)


def test_classify_on_injured_image(cnn_classifier, h_injured):
    """classify() returns ndarray without error on an injured image."""
    probs = cnn_classifier.classify(h_injured)
    assert probs.shape == (4,)


def test_classify_on_corpse_image(cnn_classifier, h_corpse):
    """classify() returns ndarray without error on a corpse image."""
    probs = cnn_classifier.classify(h_corpse)
    assert probs.shape == (4,)


# ---------------------------------------------------------------------------
# probs_to_class() — utility
# ---------------------------------------------------------------------------


def test_probs_to_class_returns_state_and_float(cnn_classifier, h_healthy):
    """probs_to_class() returns a (State, float) tuple."""
    probs = cnn_classifier.classify(h_healthy)
    label, confidence = probs_to_class(probs)
    assert isinstance(label, State)
    assert isinstance(confidence, float)


def test_probs_to_class_label_is_valid_state(cnn_classifier, h_healthy):
    """probs_to_class() label is a valid State enum value."""
    probs = cnn_classifier.classify(h_healthy)
    label, _ = probs_to_class(probs)
    assert label.value in _VALID_STATES


def test_probs_to_class_confidence_in_range(cnn_classifier, h_healthy):
    """probs_to_class() confidence is in [0.0, 1.0]."""
    probs = cnn_classifier.classify(h_healthy)
    _, confidence = probs_to_class(probs)
    assert 0.0 <= confidence <= 1.0


def test_probs_to_class_confidence_matches_argmax(cnn_classifier, h_healthy):
    """probs_to_class() confidence equals the max probability in the array."""
    net = cnn_classifier.net
    was_training = net.training
    net.eval()
    try:
        probs = cnn_classifier.classify(h_healthy)
        label, confidence = probs_to_class(probs)
    finally:
        net.train(was_training)

    assert confidence == pytest.approx(float(np.max(probs)), abs=1e-6)
    assert label.value == Humanoid.get_state_string(int(np.argmax(probs)))


# ---------------------------------------------------------------------------
# COV-14-cnn / COV-25 — FileNotFoundError at construction
# ---------------------------------------------------------------------------


def test_cnn_classifier_raises_on_missing_weights():
    """CNNClassifier raises FileNotFoundError when the weights file does not exist."""
    from models.CNNClassifier import CNNClassifier

    with pytest.raises(FileNotFoundError):
        CNNClassifier(model_file="no_such_file.pth")
