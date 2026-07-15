"""Unit tests for analysis.analyze_results — tier/condition summary computation."""

import pandas as pd

from analysis.analyze_results import summarize_by_tier_and_condition
from analysis.data_quality import flag_trials


def _sample_trials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # tier 0, human_alone: 2 correct, 1 incorrect
            {"malice_tier": 0, "condition": "human_alone", "correct": True, "reaction_time_ms": 2000, "player_answer_human": True},
            {"malice_tier": 0, "condition": "human_alone", "correct": True, "reaction_time_ms": 2200, "player_answer_human": True},
            {"malice_tier": 0, "condition": "human_alone", "correct": False, "reaction_time_ms": 1800, "player_answer_human": False},
            # tier 6, human_alone: 1 correct, 1 incorrect, plus 1 flagged (too fast)
            {"malice_tier": 6, "condition": "human_alone", "correct": True, "reaction_time_ms": 3000, "player_answer_human": True},
            {"malice_tier": 6, "condition": "human_alone", "correct": False, "reaction_time_ms": 2800, "player_answer_human": True},
            {"malice_tier": 6, "condition": "human_alone", "correct": False, "reaction_time_ms": 200, "player_answer_human": True},
            # tier 6, human_ai: 2 correct
            {"malice_tier": 6, "condition": "human_ai", "correct": True, "reaction_time_ms": 1500, "player_answer_human": True},
            {"malice_tier": 6, "condition": "human_ai", "correct": True, "reaction_time_ms": 1600, "player_answer_human": True},
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
