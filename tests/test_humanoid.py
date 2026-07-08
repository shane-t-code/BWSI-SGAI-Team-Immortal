"""Unit tests for the Humanoid class.

Covers constructor storage, all four state predicates, mutual exclusivity of
predicates, static index/string helpers, and the get_all_states() listing.
No model inference or Tkinter imports are used.
"""

import pytest

from gameplay.humanoid import Humanoid

# ---------------------------------------------------------------------------
# Helpers — build humanoids locally; no fixtures needed for these tests
# ---------------------------------------------------------------------------

_ALL_STATES = ["zombie", "healthy", "injured", "corpse"]


def _make(state: str) -> Humanoid:
    return Humanoid(fp="fake/path.png", state=state)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_humanoid_stores_fp_and_state():
    """Assert that the constructor stores fp and state exactly as passed."""
    h = Humanoid(fp="some/path.png", state="healthy")
    assert h.fp == "some/path.png"
    assert h.state == "healthy"


# ---------------------------------------------------------------------------
# Predicate — is_zombie
# ---------------------------------------------------------------------------


def test_is_zombie_true_for_zombie():
    """Assert that is_zombie() returns True when state is 'zombie'."""
    assert _make("zombie").is_zombie() is True


def test_is_zombie_false_for_healthy():
    """Assert that is_zombie() returns False when state is 'healthy'."""
    assert _make("healthy").is_zombie() is False


# ---------------------------------------------------------------------------
# Predicate — is_healthy
# ---------------------------------------------------------------------------


def test_is_healthy_true_for_healthy():
    """Assert that is_healthy() returns True when state is 'healthy'."""
    assert _make("healthy").is_healthy() is True


# ---------------------------------------------------------------------------
# Predicate — is_injured
# ---------------------------------------------------------------------------


def test_is_injured_true_for_injured():
    """Assert that is_injured() returns True when state is 'injured'."""
    assert _make("injured").is_injured() is True


# ---------------------------------------------------------------------------
# Predicate — is_corpse
# ---------------------------------------------------------------------------


def test_is_corpse_true_for_corpse():
    """Assert that is_corpse() returns True when state is 'corpse'."""
    assert _make("corpse").is_corpse() is True


# ---------------------------------------------------------------------------
# Predicate mutual exclusivity
# ---------------------------------------------------------------------------


def test_each_predicate_is_mutually_exclusive():
    """Assert that exactly one predicate is True for every valid state."""
    for state in _ALL_STATES:
        h = _make(state)
        true_count = sum([h.is_zombie(), h.is_healthy(), h.is_injured(), h.is_corpse()])
        assert (
            true_count == 1
        ), f"Expected exactly 1 True predicate for state='{state}', got {true_count}"


# ---------------------------------------------------------------------------
# Static helpers — get_state_idx / get_state_string
# ---------------------------------------------------------------------------


def test_get_state_idx_returns_correct_index():
    """Assert that get_state_idx returns the correct integer for each state string."""
    assert Humanoid.get_state_idx("zombie") == 0
    assert Humanoid.get_state_idx("healthy") == 1
    assert Humanoid.get_state_idx("injured") == 2
    assert Humanoid.get_state_idx("corpse") == 3


def test_get_state_string_returns_correct_string():
    """Assert that get_state_string returns the correct string for each integer index."""
    assert Humanoid.get_state_string(0) == "zombie"
    assert Humanoid.get_state_string(1) == "healthy"
    assert Humanoid.get_state_string(2) == "injured"
    assert Humanoid.get_state_string(3) == "corpse"


def test_get_state_idx_invalid_raises_key_error():
    """Assert that get_state_idx raises KeyError for an unrecognized state string."""
    with pytest.raises(KeyError):
        Humanoid.get_state_idx("banana")


# ---------------------------------------------------------------------------
# get_all_states
# ---------------------------------------------------------------------------


def test_get_all_states_returns_all_four():
    """Assert that get_all_states() returns a list of exactly 4 valid state strings."""
    states = Humanoid.get_all_states()
    assert len(states) == 4
    for s in states:
        assert s in _ALL_STATES


def test_get_all_states_order_matches_enum():
    """Assert that get_all_states() returns states in enum order: zombie, healthy, injured, corpse."""
    states = Humanoid.get_all_states()
    assert states[0] == "zombie"
    assert states[1] == "healthy"
    assert states[2] == "injured"
    assert states[3] == "corpse"


# ---------------------------------------------------------------------------
# value attribute
# ---------------------------------------------------------------------------


def test_humanoid_value_default():
    """Humanoid created without a value arg has value == 0.0."""
    h = Humanoid(fp="fake/path.png", state="healthy")
    assert h.value == 0.0


def test_humanoid_value_stored():
    """Humanoid created with value=0.75 stores it as self.value == 0.75."""
    h = Humanoid(fp="fake/path.png", state="injured", value=0.75)
    assert h.value == pytest.approx(0.75)


def test_humanoid_value_type():
    """humanoid.value is a float."""
    h = Humanoid(fp="fake/path.png", state="zombie", value=0.5)
    assert isinstance(h.value, float)


def test_get_all_states_mutation_does_not_corrupt_module_mapping():
    """Mutating the returned list must not affect the module-level mapping.

    get_all_states() returns a copy, so callers can safely mutate the result
    without corrupting get_state_idx() or get_state_string().
    """
    states = Humanoid.get_all_states()
    states[0] = "corrupted"
    # Module-level mapping is unaffected — the returned list was a copy
    assert Humanoid.get_state_idx("zombie") == 0
    assert Humanoid.get_state_string(0) == "zombie"
