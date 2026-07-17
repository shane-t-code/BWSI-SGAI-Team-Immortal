"""Randomized trial-order generation for the malice-budget experiment.

See docs/trial-protocol.md for the experimental design this supports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Trial:
    """One row in a participant's randomized trial order."""

    tier: int
    repeat_index: int  # 0-based index of this tier's repeats, for tracing


def build_trial_order(
    tiers: list[int], repeats_per_tier: int, seed: int | None = None
) -> list[Trial]:
    """Build a shuffled trial presentation order for one participant.

    Args:
        tiers: Malice tiers to include, e.g. [0, 1, 3, 6].
        repeats_per_tier: How many trials of each tier the participant sees.
        seed: Seed for reproducible shuffling. None means non-deterministic.

    Returns:
        A list of Trial, length len(tiers) * repeats_per_tier, in the order
        they should be presented to the participant.
    """
    trials = [
        Trial(tier=tier, repeat_index=i)
        for tier in tiers
        for i in range(repeats_per_tier)
    ]
    rng = random.Random(seed)
    rng.shuffle(trials)
    return trials


def assign_condition(participant_index: int) -> str:
    """Alternate participants between conditions so both fill up evenly.

    Args:
        participant_index: 0-based index of the participant in enrollment order.

    Returns:
        "human_alone" for even indices, "human_ai" for odd indices.
    """
    return "human_alone" if participant_index % 2 == 0 else "human_ai"
