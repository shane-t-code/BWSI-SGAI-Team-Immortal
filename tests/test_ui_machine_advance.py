"""Tests for UI._machine_advance invariants.

Uses object.__new__(UI) to bypass Tkinter initialization so these tests run
headlessly without a display.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from gameplay.enums import ActionState
from ui.ui import UI


def _make_ui(classifier=None, recommender=None, humanoid=None) -> UI:
    """Build a UI instance without calling __init__ (no Tkinter required).

    Only the attributes accessed by _machine_advance are populated.
    """
    ui = object.__new__(UI)
    ui.classifier = classifier if classifier is not None else MagicMock()
    ui.recommender = recommender if recommender is not None else MagicMock()
    ui.humanoid = humanoid if humanoid is not None else MagicMock()
    return ui


def _make_scorekeeper(
    remaining_time: int = 100, blocks_action: bool = False
) -> MagicMock:
    """Return a mock ScoreKeeper.

    Args:
        remaining_time: Value returned by scorekeeper.remaining_time.
        blocks_action: If True, map_do_action returns False (blocked).
    """
    sk = MagicMock()
    sk.remaining_time = remaining_time
    sk.map_do_action.return_value = not blocks_action
    return sk


def _make_data_parser(unvisited_count: int = 5) -> MagicMock:
    dp = MagicMock()
    dp.unvisited = set(range(unvisited_count))
    return dp


# ---------------------------------------------------------------------------
# Blocked action raises RuntimeError
# ---------------------------------------------------------------------------


def test_machine_advance_raises_runtime_error_on_blocked_action() -> None:
    """_machine_advance must raise RuntimeError when map_do_action returns False.

    This guards against custom ActionRecommender implementations that ignore
    the available action mask and return a blocked action.
    """
    probs = np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float32)

    classifier = MagicMock()
    classifier.classify.return_value = probs

    recommender = MagicMock()
    recommender.recommend.return_value = ActionState.SAVE

    ui = _make_ui(classifier=classifier, recommender=recommender)
    sk = _make_scorekeeper(remaining_time=100, blocks_action=True)
    dp = _make_data_parser()

    with pytest.raises(RuntimeError, match="blocked"):
        ui._machine_advance("data/", dp, sk)


def test_machine_advance_error_message_names_the_action() -> None:
    """The RuntimeError message must name the specific action that was blocked."""
    probs = np.array([0.05, 0.05, 0.05, 0.85], dtype=np.float32)

    classifier = MagicMock()
    classifier.classify.return_value = probs

    recommender = MagicMock()
    recommender.recommend.return_value = ActionState.SCRAM

    ui = _make_ui(classifier=classifier, recommender=recommender)
    sk = _make_scorekeeper(remaining_time=100, blocks_action=True)
    dp = _make_data_parser()

    with pytest.raises(RuntimeError, match="scram"):
        ui._machine_advance("data/", dp, sk)


def test_machine_advance_does_not_raise_on_valid_action() -> None:
    """_machine_advance must not raise when map_do_action succeeds."""
    probs = np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float32)

    classifier = MagicMock()
    classifier.classify.return_value = probs

    recommender = MagicMock()
    recommender.recommend.return_value = ActionState.SKIP

    ui = _make_ui(classifier=classifier, recommender=recommender)
    ui.update_ui = MagicMock()  # type: ignore[method-assign]
    ui.get_next = MagicMock()  # type: ignore[method-assign]

    sk = _make_scorekeeper(remaining_time=100, blocks_action=False)
    sk.log_turn = MagicMock()
    dp = _make_data_parser()

    ui._machine_advance("data/", dp, sk)  # must not raise


def test_machine_advance_returns_early_when_time_expired() -> None:
    """_machine_advance must return without calling the recommender when time is up."""
    recommender = MagicMock()
    ui = _make_ui(recommender=recommender)
    sk = _make_scorekeeper(remaining_time=0)
    dp = _make_data_parser()

    ui._machine_advance("data/", dp, sk)

    recommender.recommend.assert_not_called()


def test_machine_advance_returns_early_when_pool_empty() -> None:
    """_machine_advance must return without calling the recommender when pool is empty."""
    recommender = MagicMock()
    ui = _make_ui(recommender=recommender)
    sk = _make_scorekeeper(remaining_time=100)
    dp = _make_data_parser(unvisited_count=0)

    ui._machine_advance("data/", dp, sk)

    recommender.recommend.assert_not_called()
