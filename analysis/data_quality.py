"""Flags invalid trials before they reach the accuracy/response-time analysis.

See docs/trial-protocol.md, "Data quality" section.
"""

from __future__ import annotations

import pandas as pd

MIN_VALID_RESPONSE_MS = 1000


def flag_trials(df: pd.DataFrame, min_response_ms: int = MIN_VALID_RESPONSE_MS) -> pd.DataFrame:
    """Add a ``flagged_invalid`` column marking trials to exclude from analysis.

    A trial is invalid if the player never answered (timed out) or answered
    faster than ``min_response_ms`` — too fast to have actually looked at the
    image. Rows are never dropped here; flagging keeps the raw log intact so
    the analysis step can report how many trials were excluded.

    Args:
        df: Trial-level DataFrame matching docs/data-logging-schema.md —
            must have ``player_answer_human`` and ``reaction_time_ms`` columns.
        min_response_ms: Minimum reaction time to count as a real response.

    Returns:
        A copy of ``df`` with a new boolean ``flagged_invalid`` column.
    """
    out = df.copy()
    timed_out = out["player_answer_human"].isna()
    too_fast = out["reaction_time_ms"].isna() | (out["reaction_time_ms"] < min_response_ms)
    out["flagged_invalid"] = timed_out | too_fast
    return out
