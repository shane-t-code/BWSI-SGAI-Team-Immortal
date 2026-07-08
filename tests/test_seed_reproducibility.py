"""Seed reproducibility tests for auto mode.

Verifies that running auto mode three times with the same seed produces
byte-identical results — same score, same saved/killed counts, same
cumulative reward down to floating-point precision.

Uses object.__new__(Main) to bypass Tkinter initialisation and call
_run_auto() directly, following the same headless pattern used in
test_ui_machine_advance.py.

GUI modes (human, machine, human_cnn, human_rl, human_cnn_rl) are not
tested here — they require a Tkinter mainloop and interactive input that
cannot be driven programmatically without a display. Seed correctness for
those modes is validated at the component level: DataParser and ScoreKeeper
seeding are covered in test_data_parser.py and test_scorekeeper.py, and
_seed_all() is tested below.

Requires trained model weights to be present:
    models/weights/baselineCNN.pth
    models/weights/baselineRL.pth
Tests are skipped automatically if either file is missing.
"""

import os
import random

import numpy as np
import pytest
import torch

from game_interfaces.data_parser import DataParser
from gameplay.constants import AMBULANCE_CAPACITY, DATA_DIR, SHIFT_LENGTH
from gameplay.scorekeeper import ScoreKeeper
from main import Main, _seed_all

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CNN_WEIGHTS = "models/weights/baselineCNN.pth"
_RL_WEIGHTS = "models/weights/baselineRL.pth"

_weights_present = pytest.mark.skipif(
    not (os.path.exists(_CNN_WEIGHTS) and os.path.exists(_RL_WEIGHTS)),
    reason="Trained model weights not found — run training first.",
)


def _run_auto_seeded(seed: int) -> dict:
    """Reset all RNGs, build seeded components, run _run_auto, return score."""
    _seed_all(seed)
    data_parser = DataParser(data_dir=DATA_DIR, seed=seed)
    scorekeeper = ScoreKeeper(SHIFT_LENGTH, AMBULANCE_CAPACITY, seed=seed)

    m = object.__new__(Main)
    m.data_dir = DATA_DIR
    return m._run_auto(data_parser, scorekeeper, log=False)


# ---------------------------------------------------------------------------
# _seed_all
# ---------------------------------------------------------------------------


def test_seed_all_sets_random():
    _seed_all(0)
    a = random.random()
    _seed_all(0)
    b = random.random()
    assert a == b


def test_seed_all_sets_numpy():
    _seed_all(0)
    a = np.random.rand()
    _seed_all(0)
    b = np.random.rand()
    assert a == b


def test_seed_all_sets_torch():
    _seed_all(0)
    a = torch.rand(1).item()
    _seed_all(0)
    b = torch.rand(1).item()
    assert a == b


def test_seed_all_none_is_noop():
    """Calling _seed_all(None) should not raise and should leave RNGs alone."""
    _seed_all(42)
    _seed_all(None)  # should be a no-op
    # RNG state should continue from wherever it was — just verify no crash
    _ = random.random()


# ---------------------------------------------------------------------------
# Auto mode seed reproducibility
# ---------------------------------------------------------------------------


@_weights_present
def test_auto_seed_reproducibility_score():
    """Three auto runs with the same seed produce identical score dicts."""
    seed = 42
    scores = [_run_auto_seeded(seed) for _ in range(3)]
    assert scores[0] == scores[1], f"Run 1 vs 2 differ: {scores[0]} != {scores[1]}"
    assert scores[1] == scores[2], f"Run 2 vs 3 differ: {scores[1]} != {scores[2]}"


@_weights_present
def test_auto_different_seeds_may_differ():
    """Different seeds should (almost certainly) produce different outcomes."""
    score_42 = _run_auto_seeded(42)
    score_99 = _run_auto_seeded(99)
    # Not a hard requirement — seeds could theoretically collide — but with
    # a 200-image dataset and full-game scoring this is extremely unlikely.
    assert score_42 != score_99, (
        "Seeds 42 and 99 produced identical scores — "
        "this is possible but unlikely; investigate if it recurs."
    )
