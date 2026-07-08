"""Unit tests for DataParser and the datarow_to_state() mapping function.

Covers CSV loading, FileNotFoundError on bad paths, get_random() pool
mechanics (isolation, exhaustion, reset), and all four branches of the
datarow_to_state() classification logic.
No model inference or Tkinter imports are used.
"""

import pandas as pd
import pytest

from game_interfaces.data_parser import DataParser, datarow_to_state, datarow_to_value
from gameplay.humanoid import Humanoid

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_data_parser_loads_csv():
    """Assert that DataParser(data_dir='data') succeeds and loads a non-empty DataFrame."""
    parser = DataParser(data_dir="data")
    assert parser.df is not None
    assert len(parser.df) > 0


def test_data_parser_missing_dir_raises():
    """Assert that DataParser raises FileNotFoundError when the data directory does not exist."""
    with pytest.raises(FileNotFoundError):
        DataParser(data_dir="nonexistent/path")


# ---------------------------------------------------------------------------
# get_random() — return type and field validity
# ---------------------------------------------------------------------------


def test_get_random_returns_humanoid(data_parser):
    """Assert that get_random() returns a Humanoid instance."""
    result = data_parser.get_random()
    assert isinstance(result, Humanoid)


def test_get_random_fp_is_string(data_parser):
    """Assert that the Humanoid returned by get_random() has a string fp attribute."""
    result = data_parser.get_random()
    assert isinstance(result.fp, str)


def test_get_random_state_is_valid(data_parser):
    """Assert that the Humanoid returned by get_random() has a state in the set of four valid strings."""
    valid_states = {"zombie", "healthy", "injured", "corpse"}
    result = data_parser.get_random()
    assert result.state in valid_states


# ---------------------------------------------------------------------------
# get_random() — pool mechanics
# ---------------------------------------------------------------------------


def test_get_random_moves_index_to_visited(data_parser):
    """Assert that each call to get_random() moves exactly one index from unvisited to visited."""
    unvisited_before = len(data_parser.unvisited)
    visited_before = len(data_parser.visited)
    data_parser.get_random()
    assert len(data_parser.unvisited) == unvisited_before - 1
    assert len(data_parser.visited) == visited_before + 1


def test_get_random_exhaustion_raises_value_error(data_parser):
    """Assert that get_random() raises ValueError after the entire pool has been drawn."""
    # Exhaust the pool — real dataset is large so we patch unvisited directly
    data_parser.unvisited = set()
    with pytest.raises(ValueError):
        data_parser.get_random()


def test_get_random_real_sequential_exhaustion_raises_value_error():
    """Assert that sequential get_random() calls exhaust the pool and raise ValueError on the next call.

    Uses a tiny inline DataFrame so the real draw-and-remove mechanics are
    exercised without iterating thousands of rows.
    """
    import pandas as pd

    tiny_df = pd.DataFrame(
        {
            "Filename": ["a.png", "b.png", "c.png"],
            "Class": ["Default", "Default", "Zombie"],
            "Injured": [False, True, False],
        }
    )
    parser = DataParser(data_dir="data")
    parser.df = tiny_df
    parser.unvisited = set(tiny_df.index)
    parser.visited = []

    # Draw all three rows
    for _ in range(3):
        parser.get_random()

    assert len(parser.unvisited) == 0
    assert len(parser.visited) == 3

    with pytest.raises(ValueError):
        parser.get_random()


def test_reset_restores_full_pool(data_parser):
    """Assert that reset() after exhausting the pool allows get_random() to succeed again."""
    total = len(data_parser.unvisited)
    data_parser.unvisited = set()
    data_parser.visited = list(range(total))
    data_parser.reset()
    result = data_parser.get_random()
    assert isinstance(result, Humanoid)


# ---------------------------------------------------------------------------
# datarow_to_state() — all four branches
# ---------------------------------------------------------------------------


def test_datarow_to_state_default_not_injured_is_healthy():
    """Assert that Class='Default', Injured=False maps to 'healthy'."""
    row = pd.Series({"Class": "Default", "Injured": False})
    assert datarow_to_state(row) == "healthy"


def test_datarow_to_state_default_injured_is_injured():
    """Assert that Class='Default', Injured=True maps to 'injured'."""
    row = pd.Series({"Class": "Default", "Injured": True})
    assert datarow_to_state(row) == "injured"


def test_datarow_to_state_nondefault_not_injured_is_zombie():
    """Assert that Class='Zombie', Injured=False maps to 'zombie'."""
    row = pd.Series({"Class": "Zombie", "Injured": False})
    assert datarow_to_state(row) == "zombie"


def test_datarow_to_state_nondefault_injured_is_corpse():
    """Assert that Class='Zombie', Injured=True maps to 'corpse'."""
    row = pd.Series({"Class": "Zombie", "Injured": True})
    assert datarow_to_state(row) == "corpse"


# ---------------------------------------------------------------------------
# COV-09 — invalid Filename raises ValueError
# ---------------------------------------------------------------------------


def test_get_random_raises_on_empty_filename():
    """get_random() raises ValueError when Filename is an empty string."""
    parser = DataParser(data_dir="data")
    bad_df = pd.DataFrame({"Filename": [""], "Class": ["Default"], "Injured": [False]})
    parser.df = bad_df
    parser.unvisited = set(bad_df.index)
    parser.visited = []
    with pytest.raises(ValueError, match="Invalid 'Filename'"):
        parser.get_random()


def test_get_random_raises_on_non_string_filename():
    """get_random() raises ValueError when Filename is None (non-string)."""
    parser = DataParser(data_dir="data")
    bad_df = pd.DataFrame(
        {"Filename": [None], "Class": ["Default"], "Injured": [False]}
    )
    parser.df = bad_df
    parser.unvisited = set(bad_df.index)
    parser.visited = []
    with pytest.raises(ValueError, match="Invalid 'Filename'"):
        parser.get_random()


# ---------------------------------------------------------------------------
# COV-10 — without-replacement property
# ---------------------------------------------------------------------------


def test_get_random_no_replacement():
    """Drawing all items from a 3-row dataset yields 3 distinct visited indices."""
    tiny_df = pd.DataFrame(
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
    parser = DataParser(data_dir="data")
    parser.df = tiny_df
    parser.unvisited = set(tiny_df.index)
    parser.visited = []

    for _ in range(3):
        parser.get_random()

    assert len(set(parser.visited)) == 3


# ---------------------------------------------------------------------------
# humanoid.value from get_random()
# ---------------------------------------------------------------------------


def test_get_random_value_in_range(data_parser):
    """humanoid.value from get_random() is in [0.0, 1.0)."""
    h = data_parser.get_random()
    assert 0.0 <= h.value < 1.0


def test_get_random_value_is_float(data_parser):
    """humanoid.value from get_random() is a float."""
    h = data_parser.get_random()
    assert isinstance(h.value, float)


def test_get_random_values_vary():
    """Drawing 20 humanoids produces at least two distinct value scores (probabilistic)."""
    parser = DataParser(data_dir="data", seed=42)
    values = [parser.get_random().value for _ in range(20)]
    assert len(set(values)) > 1, "All 20 humanoid values were identical — unexpected"


# ---------------------------------------------------------------------------
# datarow_to_value()
# ---------------------------------------------------------------------------


def test_datarow_to_value_returns_float():
    """datarow_to_value() called on a real metadata row returns a float in [0.0, 1.0)."""
    dp = DataParser(data_dir="data")
    row = dp.df.iloc[0]
    result = datarow_to_value(row)
    assert isinstance(result, float)
    assert 0.0 <= result < 1.0


def test_reset_restores_full_unvisited_pool():
    """After exhausting the pool and calling reset(), visited==[] and len(unvisited)==original_total."""
    tiny_df = pd.DataFrame(
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
    parser = DataParser(data_dir="data")
    parser.df = tiny_df
    parser.unvisited = set(tiny_df.index)
    parser.visited = []
    original_total = len(tiny_df)

    for _ in range(original_total):
        parser.get_random()

    assert len(parser.unvisited) == 0
    parser.reset()
    assert len(parser.visited) == 0
    assert len(parser.unvisited) == original_total
