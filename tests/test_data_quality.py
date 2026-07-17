"""Unit tests for analysis.data_quality — invalid-trial flagging."""

import pandas as pd

from analysis.data_quality import flag_trials


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # normal, valid response
            {"player_answer_human": True, "reaction_time_ms": 2500},
            # too fast — under the 1000ms floor
            {"player_answer_human": False, "reaction_time_ms": 300},
            # timed out — no answer given
            {"player_answer_human": None, "reaction_time_ms": None},
            # right at the boundary — valid (not strictly less than the floor)
            {"player_answer_human": True, "reaction_time_ms": 1000},
        ]
    )


def test_flag_trials_marks_normal_response_valid():
    """Assert a normal-speed, non-timed-out response is not flagged."""
    out = flag_trials(_sample_df())
    assert out.loc[0, "flagged_invalid"] == False  # noqa: E712


def test_flag_trials_marks_too_fast_response_invalid():
    """Assert a sub-1000ms response is flagged invalid."""
    out = flag_trials(_sample_df())
    assert out.loc[1, "flagged_invalid"] == True  # noqa: E712


def test_flag_trials_marks_timeout_invalid():
    """Assert a timed-out (null answer) response is flagged invalid."""
    out = flag_trials(_sample_df())
    assert out.loc[2, "flagged_invalid"] == True  # noqa: E712


def test_flag_trials_boundary_is_valid():
    """Assert a response exactly at the minimum threshold is NOT flagged (strictly-less-than check)."""
    out = flag_trials(_sample_df())
    assert out.loc[3, "flagged_invalid"] == False  # noqa: E712


def test_flag_trials_respects_custom_threshold():
    """Assert a custom min_response_ms changes which rows get flagged."""
    out = flag_trials(_sample_df(), min_response_ms=3000)
    # Now even the 2500ms "normal" response should be flagged.
    assert out.loc[0, "flagged_invalid"] == True  # noqa: E712


def test_flag_trials_does_not_mutate_input():
    """Assert the original DataFrame is untouched (flag_trials returns a copy)."""
    df = _sample_df()
    flag_trials(df)
    assert "flagged_invalid" not in df.columns
