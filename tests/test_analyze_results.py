"""Unit tests for analysis.analyze_results — tier/condition summary computation."""

import pandas as pd

from analysis.analyze_results import filter_first_session, summarize_by_tier_and_condition
from analysis.data_quality import flag_trials


def _sample_trials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # tier 0, human_alone: 2 correct, 1 incorrect
            {"malice_tier": 0, "condition": "human_alone", "correct": True, "reaction_time_ms": 2000, "player_answer_human": "human"},
            {"malice_tier": 0, "condition": "human_alone", "correct": True, "reaction_time_ms": 2200, "player_answer_human": "human"},
            {"malice_tier": 0, "condition": "human_alone", "correct": False, "reaction_time_ms": 1800, "player_answer_human": "zombie"},
            # tier 6, human_alone: 1 correct, 1 incorrect, plus 1 flagged (too fast)
            {"malice_tier": 6, "condition": "human_alone", "correct": True, "reaction_time_ms": 3000, "player_answer_human": "human"},
            {"malice_tier": 6, "condition": "human_alone", "correct": False, "reaction_time_ms": 2800, "player_answer_human": "human"},
            {"malice_tier": 6, "condition": "human_alone", "correct": False, "reaction_time_ms": 200, "player_answer_human": "human"},
            # tier 6, human_ai: 2 correct
            {"malice_tier": 6, "condition": "human_ai", "correct": True, "reaction_time_ms": 1500, "player_answer_human": "human"},
            {"malice_tier": 6, "condition": "human_ai", "correct": True, "reaction_time_ms": 1600, "player_answer_human": "human"},
        ]
    )


def test_summarize_computes_accuracy_per_cell():
    """Assert accuracy is the mean of `correct` within each (tier, condition) cell, excluding flagged rows."""
    df = flag_trials(_sample_trials())
    summary = summarize_by_tier_and_condition(df)

    tier0 = summary[(summary["malice_tier"] == 0) & (summary["condition"] == "human_alone")].iloc[0]
    assert tier0["n_trials"] == 3
    assert round(tier0["accuracy"], 4) == round(2 / 3, 4)

    tier6_ai = summary[(summary["malice_tier"] == 6) & (summary["condition"] == "human_ai")].iloc[0]
    assert tier6_ai["n_trials"] == 2
    assert tier6_ai["accuracy"] == 1.0


def test_summarize_excludes_flagged_trials_from_stats_but_reports_the_count():
    """Assert the too-fast trial in tier 6/human_alone is excluded from accuracy but counted in n_flagged."""
    df = flag_trials(_sample_trials())
    summary = summarize_by_tier_and_condition(df)

    tier6_alone = summary[(summary["malice_tier"] == 6) & (summary["condition"] == "human_alone")].iloc[0]
    # 3 rows total for this cell, 1 flagged (200ms) -> 2 valid trials remain
    assert tier6_alone["n_trials"] == 2
    assert tier6_alone["n_flagged"] == 1
    assert tier6_alone["accuracy"] == 0.5


def test_summarize_computes_mean_reaction_time():
    """Assert mean_reaction_time_ms averages only the valid (non-flagged) trials."""
    df = flag_trials(_sample_trials())
    summary = summarize_by_tier_and_condition(df)

    tier0 = summary[(summary["malice_tier"] == 0) & (summary["condition"] == "human_alone")].iloc[0]
    assert round(tier0["mean_reaction_time_ms"], 2) == round((2000 + 2200 + 1800) / 3, 2)


def test_summarize_keeps_a_cell_visible_when_every_trial_is_flagged():
    """A cell where every trial is invalid (e.g. everyone timed out) must still
    appear as n_trials=0, n_flagged=N — not disappear from the output, which
    would hide exactly the worst-case flag rate. Regression test for the bug
    Prajwal caught in PR review."""
    df = pd.DataFrame(
        [
            # tier 3, human_ai: both trials timed out -> 100% flagged
            {"malice_tier": 3, "condition": "human_ai", "correct": None, "reaction_time_ms": None, "player_answer_human": None},
            {"malice_tier": 3, "condition": "human_ai", "correct": None, "reaction_time_ms": None, "player_answer_human": None},
        ]
    )
    df = flag_trials(df)
    summary = summarize_by_tier_and_condition(df)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["malice_tier"] == 3
    assert row["condition"] == "human_ai"
    assert row["n_trials"] == 0
    assert row["n_flagged"] == 2
    assert pd.isna(row["accuracy"])


def _sample_trials_with_sessions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # participant A: two sessions, first one earlier -> only this row should survive
            {"participant_id": "A", "session_id": "A-s1", "timestamp": "2026-07-17T10:00:00Z", "malice_tier": 0, "condition": "human_alone", "correct": True, "reaction_time_ms": 2000, "player_answer_human": "human"},
            {"participant_id": "A", "session_id": "A-s2", "timestamp": "2026-07-17T11:00:00Z", "malice_tier": 0, "condition": "human_alone", "correct": False, "reaction_time_ms": 1900, "player_answer_human": "zombie"},
            # participant B: single session
            {"participant_id": "B", "session_id": "B-s1", "timestamp": "2026-07-17T09:30:00Z", "malice_tier": 0, "condition": "human_ai", "correct": True, "reaction_time_ms": 1700, "player_answer_human": "human"},
        ]
    )


def test_filter_first_session_keeps_only_each_participants_earliest_session():
    """Assert repeat sessions from the same participant are dropped, keeping only the earliest by timestamp."""
    df = _sample_trials_with_sessions()
    filtered = filter_first_session(df)

    assert set(filtered["session_id"]) == {"A-s1", "B-s1"}
    assert len(filtered) == 2


def test_filter_first_session_is_a_no_op_when_no_repeats_exist():
    """Assert a dataset with one session per participant is unchanged in row count."""
    df = _sample_trials_with_sessions()
    df = df[df["session_id"] != "A-s2"]  # drop the repeat, leaving one row per participant
    filtered = filter_first_session(df)
    assert len(filtered) == len(df)
