"""Tests for pure UI helper methods and the game-over button-disable regression.

All tests here run headlessly — no Tkinter window is created.
Pure static methods are called directly on the class; the button regression
uses object.__new__(UI) to bypass __init__.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gameplay.constants import SHIFT_LENGTH
from ui.ui import UI

# ---------------------------------------------------------------------------
# UI._value_label_text — bin mapping and clamping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_category",
    [
        (0.00, "VERY LOW"),
        (0.10, "VERY LOW"),
        (0.20, "LOW"),
        (0.30, "LOW"),
        (0.40, "NEUTRAL"),
        (0.50, "NEUTRAL"),
        (0.60, "HIGH"),
        (0.70, "HIGH"),
        (0.80, "VERY HIGH"),
        (0.99, "VERY HIGH"),
        (1.00, "VERY HIGH"),
    ],
)
def test_value_label_text_bins(value, expected_category):
    """_value_label_text maps value into the correct category bin."""
    text = UI._value_label_text(value)
    assert expected_category in text
    assert f"{value:.2f}" in text


@pytest.mark.parametrize(
    "value, expected_category",
    [
        (-0.5, "VERY LOW"),  # negative clamps to 0.0
        (-0.01, "VERY LOW"),
        (1.01, "VERY HIGH"),  # over 1.0 clamps to 1.0
        (99.0, "VERY HIGH"),
    ],
)
def test_value_label_text_clamps_out_of_range(value, expected_category):
    """_value_label_text clamps values outside [0, 1] without raising."""
    text = UI._value_label_text(value)
    assert expected_category in text


# ---------------------------------------------------------------------------
# UI._time_to_clock_hands — clock hand positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "remaining_time, expected_hour, expected_minute",
    [
        (540, 3, 0),  # full shift (9 hours remaining → hour=12-9=3, minute=0)
        (480, 4, 0),  # 8 hours remaining
        (60, 11, 0),  # 1 hour remaining → floor(60/60)=1
        (30, 12, 30),  # <1 hour → floor(30/60)=0, so hour=12, minute=30
        (1, 12, 59),  # <1 hour → floor(1/60)=0, so hour=12, minute=59
        (0, 12, 0),  # time up — clamped to 0
        (-10, 12, 0),  # negative clamped to 0
    ],
)
def test_time_to_clock_hands(remaining_time, expected_hour, expected_minute):
    """_time_to_clock_hands returns correct (hour, minute) for given remaining time."""
    hour, minute = UI._time_to_clock_hands(remaining_time)
    assert hour == expected_hour
    assert minute == expected_minute


def test_time_to_clock_hands_exact_hour_boundary():
    """remaining_time that is an exact multiple of 60 returns minute=0 (not 60)."""
    _, minute = UI._time_to_clock_hands(120)
    assert minute == 0, "minute should be 0 at exact hour boundary, not 60"


# ---------------------------------------------------------------------------
# Regression: game-over buttons stay disabled even when log=True causes reset()
#
# Bug: get_next() called disable_buttons() AFTER scorekeeper.reset(), which
# restored remaining_time to SHIFT_LENGTH, fooling disable_buttons() into
# re-enabling all buttons at game end.
#
# Fix: disable_all() is now called BEFORE reset(). disable_buttons() is only
# called in the continue-play else branch.
# ---------------------------------------------------------------------------


def _make_headless_ui(log: bool = True) -> UI:
    """Build a UI instance without Tkinter, with just enough attributes for get_next()."""
    ui = object.__new__(UI)
    ui.log = log
    ui.humanoid = MagicMock()
    ui._last_probs = None
    ui._last_recommended_action = None
    ui.classifier = None
    ui.recommender = None
    ui._info_label = None
    ui.button_menu = MagicMock()
    ui.game_viewer = MagicMock()
    ui.capacity_meter = MagicMock()
    ui.value_label = MagicMock()
    return ui


def _make_game_over_scorekeeper(remaining_time: int = 0) -> MagicMock:
    """Return a mock ScoreKeeper in a game-over state.

    reset() side-effect restores remaining_time to SHIFT_LENGTH, reproducing
    the exact condition that triggered the original bug.
    """
    sk = MagicMock()
    sk.remaining_time = remaining_time
    sk.get_score.return_value = {"saved": 0, "killed": 0, "score": 0.0}
    sk.get_current_capacity.return_value = 0

    def _reset_side_effect():
        sk.remaining_time = SHIFT_LENGTH  # this is what caused the bug

    sk.reset.side_effect = _reset_side_effect
    return sk


def test_game_over_disable_all_called_before_reset():
    """disable_all() is called before scorekeeper.reset() at game end."""
    ui = _make_headless_ui(log=True)
    sk = _make_game_over_scorekeeper(remaining_time=0)
    dp = MagicMock()
    dp.unvisited = set()  # no humanoids left

    call_order = []
    ui.button_menu.disable_all.side_effect = lambda: call_order.append("disable_all")
    sk.reset.side_effect = lambda: call_order.append("reset")  # type: ignore[method-assign]

    ui.get_next(data_fp="data", data_parser=dp, scorekeeper=sk)

    assert "disable_all" in call_order, "disable_all() was never called at game end"
    assert "reset" in call_order, "reset() was never called despite log=True"
    assert call_order.index("disable_all") < call_order.index("reset"), (
        "disable_all() must be called BEFORE reset() — "
        "reset() restores remaining_time which would fool disable_buttons()"
    )


def test_game_over_disable_buttons_not_called():
    """disable_buttons() is never called in the game-over path (only in continue-play path)."""
    ui = _make_headless_ui(log=True)
    sk = _make_game_over_scorekeeper(remaining_time=0)
    dp = MagicMock()
    dp.unvisited = set()

    ui.get_next(data_fp="data", data_parser=dp, scorekeeper=sk)

    ui.button_menu.disable_buttons.assert_not_called()


def test_game_over_no_log_buttons_still_disabled():
    """disable_all() is called even when log=False (no reset() called in that case)."""
    ui = _make_headless_ui(log=False)
    sk = _make_game_over_scorekeeper(remaining_time=0)
    dp = MagicMock()
    dp.unvisited = set()

    ui.get_next(data_fp="data", data_parser=dp, scorekeeper=sk)

    ui.button_menu.disable_all.assert_called_once()
    sk.reset.assert_not_called()
