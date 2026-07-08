"""Contract tests for enum ordering and ActionCost values.

These tests pin the exact integer positions and string values that the RL
observation space and CNN output mapping depend on. A reordering of State or
ActionState, or a change to any ActionCost constant, will fail here loudly.
"""

import pytest

from gameplay.enums import ActionCost, ActionState, State
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper

# ---------------------------------------------------------------------------
# State enum — ordering and string values
# ---------------------------------------------------------------------------


def test_state_enum_ordering():
    """Assert that State enum members appear in the exact order: zombie(0), healthy(1), injured(2), corpse(3)."""
    members = list(State)
    assert members[0] == State.ZOMBIE
    assert members[1] == State.HEALTHY
    assert members[2] == State.INJURED
    assert members[3] == State.CORPSE


def test_state_enum_values():
    """Assert that all four State string values are exactly 'zombie', 'healthy', 'injured', 'corpse'."""
    assert State.ZOMBIE.value == "zombie"
    assert State.HEALTHY.value == "healthy"
    assert State.INJURED.value == "injured"
    assert State.CORPSE.value == "corpse"


# ---------------------------------------------------------------------------
# ActionState enum — ordering and string values
# ---------------------------------------------------------------------------


def test_action_state_ordering():
    """Assert that ActionState enum members appear in the exact order: save(0), squish(1), skip(2), scram(3)."""
    members = list(ActionState)
    assert members[0] == ActionState.SAVE
    assert members[1] == ActionState.SQUISH
    assert members[2] == ActionState.SKIP
    assert members[3] == ActionState.SCRAM


def test_action_state_values():
    """Assert that all four ActionState string values are exactly 'save', 'squish', 'skip', 'scram'."""
    assert ActionState.SAVE.value == "save"
    assert ActionState.SQUISH.value == "squish"
    assert ActionState.SKIP.value == "skip"
    assert ActionState.SCRAM.value == "scram"


# ---------------------------------------------------------------------------
# ActionCost enum — exact minute values
# ---------------------------------------------------------------------------


def test_action_cost_save():
    """Assert that SAVE costs exactly 30 minutes."""
    assert ActionCost.SAVE.value == 30


def test_action_cost_squish():
    """Assert that SQUISH costs exactly 5 minutes."""
    assert ActionCost.SQUISH.value == 5


def test_action_cost_skip():
    """Assert that SKIP costs exactly 15 minutes."""
    assert ActionCost.SKIP.value == 15


def test_action_cost_scram():
    """Assert that SCRAM costs exactly 120 minutes."""
    assert ActionCost.SCRAM.value == 120


# ---------------------------------------------------------------------------
# Roundtrip: string ↔ int mappings on Humanoid and ScoreKeeper
# ---------------------------------------------------------------------------


def test_map_class_str_to_int_roundtrip():
    """Assert that get_state_idx(get_state_string(i)) == i for all state indices 0..3."""
    for i in range(4):
        string = Humanoid.get_state_string(i)
        recovered = Humanoid.get_state_idx(string)
        assert recovered == i, f"Roundtrip failed at index {i}: got {recovered}"


def test_map_action_str_to_int_roundtrip():
    """Assert that get_action_idx(get_action_string(i)) == i for all action indices 0..3."""
    for i in range(4):
        string = ScoreKeeper.get_action_string(i)
        recovered = ScoreKeeper.get_action_idx(string)
        assert recovered == i, f"Roundtrip failed at index {i}: got {recovered}"


# ---------------------------------------------------------------------------
# COV-34 — get_state_string out-of-range raises IndexError
# ---------------------------------------------------------------------------


def test_get_state_string_raises_index_error_on_invalid_index():
    """Humanoid.get_state_string(99) raises IndexError."""
    with pytest.raises(IndexError):
        Humanoid.get_state_string(99)


# ---------------------------------------------------------------------------
# COV-29 — ActionCost / ActionState structural invariants
# ---------------------------------------------------------------------------


def test_action_cost_and_action_state_same_length():
    """ActionCost and ActionState enums have the same number of members."""
    assert len(list(ActionCost)) == len(list(ActionState))


def test_action_state_length_matches_num_actions():
    """len(ActionState) matches ScoreKeeper.num_actions."""
    from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH

    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    assert len(list(ActionState)) == sk.num_actions
