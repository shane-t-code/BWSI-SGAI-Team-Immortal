"""Accuracy / response-time analysis by malice tier and condition.

Reads a trial-level CSV matching docs/data-logging-schema.md and produces the
accuracy-vs-tier and human-alone-vs-human+AI comparison described in
docs/trial-protocol.md. Run directly against an exported trial log:

    python -m analysis.analyze_results path/to/trials.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from analysis.data_quality import flag_trials


def load_trials(path: str) -> pd.DataFrame:
    """Load a trial-level CSV and flag invalid rows.

    Args:
        path: Path to a CSV matching the schema in docs/data-logging-schema.md.

    Returns:
        The loaded DataFrame with a ``flagged_invalid`` column added.
    """
    df = pd.read_csv(path)
    return flag_trials(df)


def summarize_by_tier_and_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy and mean response time per (tier, condition) cell.

    Invalid trials (see analysis.data_quality) are excluded from the
    statistics but their count is reported so a suspiciously high flag rate
    is visible rather than silently dropped.

    Args:
        df: Output of load_trials() — must already have ``flagged_invalid``.

    Returns:
        One row per (malice_tier, condition) with n_trials, n_flagged,
        accuracy, and mean_reaction_time_ms.
    """
    # Every (tier, condition) cell that appears anywhere in df gets a row in
    # the output, even if 100% of its trials are flagged invalid — otherwise
    # a fully-flagged cell (e.g. everyone timed out on tier 6) silently
    # disappears from valid.groupby(...) instead of showing up as
    # n_trials=0, n_flagged=N, which is exactly the worst-case flag rate we
    # most need to see.
    all_cells = df.groupby(["malice_tier", "condition"]).size().to_frame("n_total")
    flagged_counts = (
        df.groupby(["malice_tier", "condition"])["flagged_invalid"]
        .sum()
        .rename("n_flagged")
    )
    valid = df[~df["flagged_invalid"]]
    valid_stats = valid.groupby(["malice_tier", "condition"]).agg(
        n_trials=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_reaction_time_ms=("reaction_time_ms", "mean"),
    )
    summary = (
        all_cells.join(valid_stats)
        .join(flagged_counts)
        .drop(columns="n_total")
        .reset_index()
    )
    summary["n_trials"] = summary["n_trials"].fillna(0).astype(int)
    summary["n_flagged"] = summary["n_flagged"].fillna(0).astype(int)
    return summary


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        print("Usage: python -m analysis.analyze_results <trials.csv>")
        sys.exit(1)
    df = load_trials(argv[1])
    summary = summarize_by_tier_and_condition(df)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
