"""Unit tests for analysis.trial_protocol — randomized trial-order generation."""

from analysis.trial_protocol import assign_condition, build_trial_order


def test_build_trial_order_has_correct_length():
    """Assert the trial order has one row per tier x repeat."""
    order = build_trial_order(tiers=[0, 2, 8, 10], repeats_per_tier=5, seed=1)
    assert len(order) == 4 * 5


def test_build_trial_order_contains_each_tier_the_right_number_of_times():
    """Assert every tier appears exactly repeats_per_tier times, regardless of shuffle."""
    order = build_trial_order(tiers=[0, 2, 8, 10], repeats_per_tier=5, seed=1)
    for tier in [0, 2, 8, 10]:
        assert sum(1 for t in order if t.tier == tier) == 5


def test_build_trial_order_is_deterministic_for_same_seed():
    """Assert the same seed always produces the same presentation order."""
    order_a = build_trial_order(tiers=[0, 2, 8, 10], repeats_per_tier=5, seed=42)
    order_b = build_trial_order(tiers=[0, 2, 8, 10], repeats_per_tier=5, seed=42)
    assert order_a == order_b


def test_build_trial_order_differs_across_seeds():
    """Assert two different seeds produce different orders (sanity check that shuffling happens)."""
    order_a = build_trial_order(tiers=[0, 2, 8, 10], repeats_per_tier=5, seed=1)
    order_b = build_trial_order(tiers=[0, 2, 8, 10], repeats_per_tier=5, seed=2)
    assert order_a != order_b


def test_assign_condition_alternates():
    """Assert participants alternate between conditions in enrollment order."""
    assert assign_condition(0) == "human_alone"
    assert assign_condition(1) == "human_ai"
    assert assign_condition(2) == "human_alone"
    assert assign_condition(3) == "human_ai"
