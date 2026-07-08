"""Integration tests for 7 end-to-end game scenarios.

These are integration tests — they verify end-to-end game loop behavior, not individual components.

- **Unit tests** verify individual components in isolation with controlled inputs.
- **Integration tests** (this file) verify that each mode's game loop runs
  from start to a valid terminal state without crashing — no pinned score
  values, no behavior assertions beyond "something happened and the state is
  sane."
- **Regression tests** would pin specific output values; we intentionally omit
  them here because model weights and data can change.

Each test uses real weights, the real data directory, and real game logic.
Tkinter is never imported — all tests are fully headless.
"""

from __future__ import annotations

from game_interfaces.data_parser import DataParser
from gameplay.enums import ActionState
from gameplay.scorekeeper import ScoreKeeper
from models.CNNClassifier import CNNClassifier
from models.RLRecommender import RLRecommender

# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------


def _assert_valid_end_state(
    data_parser: DataParser,
    scorekeeper: ScoreKeeper,
) -> None:
    """Assert that the game loop exited for a valid reason and left sane state.

    Checks four invariants:
    1. The loop terminated because the unvisited pool is empty **or** time
       ran out — never a silent early exit.
    2. At least one humanoid was processed (the game actually ran).
    3. ``get_score()`` returns a ``dict`` (note: this call has the side effect
       of scraming the ambulance if it is not empty — call it last).
    4. The score dict contains non-negative "killed" and "saved" counts.

    Args:
        data_parser: The DataParser instance used during the game loop.
        scorekeeper: The ScoreKeeper instance used during the game loop.
    """
    # Invariant 1 — valid termination condition
    assert len(data_parser.unvisited) == 0 or scorekeeper.remaining_time <= 0, (
        "Game loop exited early: unvisited pool is not empty "
        "and remaining_time is still positive."
    )

    # Invariant 2 — at least one humanoid was processed
    assert (
        len(data_parser.visited) > 0
    ), "No humanoids were visited — the game loop never executed."

    # Invariant 3 + 4 — score structure (call last due to scram side effect)
    score = scorekeeper.get_score()
    assert isinstance(score, dict), f"get_score() returned {type(score)}, expected dict"
    assert "killed" in score, f"'killed' key missing from score: {score}"
    assert "saved" in score, f"'saved' key missing from score: {score}"
    assert score["killed"] >= 0, f"score['killed'] is negative: {score['killed']}"
    assert score["saved"] >= 0, f"score['saved'] is negative: {score['saved']}"


# ---------------------------------------------------------------------------
# Integration tests — one per game mode
# ---------------------------------------------------------------------------


def test_auto_mode_runs_to_completion(
    data_parser: DataParser,
    sk_fresh: ScoreKeeper,
    cnn_classifier: CNNClassifier,
    rl_recommender: RLRecommender,
) -> None:
    """Verify the 'auto' mode game loop completes headlessly without crashing.

    In auto mode the CNN classifies each humanoid and the RL recommender
    selects every action. Probs are computed once per turn and passed to
    the recommender — mirroring the production _run_auto() loop.

    Args:
        data_parser: Fresh DataParser fixture (function-scoped).
        sk_fresh: Fresh ScoreKeeper fixture (function-scoped).
        cnn_classifier: Session-scoped CNNClassifier with real weights.
        rl_recommender: Session-scoped RLRecommender with real weights.
    """
    while len(data_parser.unvisited) > 0 and sk_fresh.remaining_time > 0:
        humanoid = data_parser.get_random()
        probs = cnn_classifier.classify(humanoid)
        action = rl_recommender.recommend(humanoid, probs, sk_fresh)
        assert action in list(ActionState)  # RL recommendation is always a valid action
        sk_fresh.map_do_action(action, humanoid)

    _assert_valid_end_state(data_parser, sk_fresh)


def test_human_mode_runs_to_completion(
    data_parser: DataParser,
    sk_fresh: ScoreKeeper,
) -> None:
    """Verify the 'human' mode game loop completes headlessly without crashing.

    In human mode there is no ML involvement. SQUISH is used as a
    deterministic stand-in for human input — it always costs 5 minutes so the
    loop terminates on time exhaustion, which mirrors a real human session.

    Args:
        data_parser: Fresh DataParser fixture (function-scoped).
        sk_fresh: Fresh ScoreKeeper fixture (function-scoped).
    """
    while len(data_parser.unvisited) > 0 and sk_fresh.remaining_time > 0:
        humanoid = data_parser.get_random()
        sk_fresh.map_do_action(ActionState.SQUISH, humanoid)

    _assert_valid_end_state(data_parser, sk_fresh)


def test_human_cnn_mode_runs_to_completion(
    data_parser: DataParser,
    sk_fresh: ScoreKeeper,
    cnn_classifier: CNNClassifier,
) -> None:
    """Verify the 'human_cnn' mode game loop completes headlessly without crashing.

    In human_cnn mode the CNN classifies each humanoid (providing a label and
    confidence score to the human), but the action is still chosen by the human.
    SQUISH is used as the fixed human decision so the loop terminates cleanly.
    The classification result is intentionally discarded — the test only
    verifies that classify() does not crash and the loop completes.

    Args:
        data_parser: Fresh DataParser fixture (function-scoped).
        sk_fresh: Fresh ScoreKeeper fixture (function-scoped).
        cnn_classifier: Session-scoped CNNClassifier with real weights.
    """
    while len(data_parser.unvisited) > 0 and sk_fresh.remaining_time > 0:
        humanoid = data_parser.get_random()
        cnn_classifier.classify(humanoid)
        sk_fresh.map_do_action(ActionState.SQUISH, humanoid)

    _assert_valid_end_state(data_parser, sk_fresh)


def test_human_rl_mode_runs_to_completion(
    data_parser: DataParser,
    sk_fresh: ScoreKeeper,
    cnn_classifier: CNNClassifier,
    rl_recommender: RLRecommender,
) -> None:
    """Verify the 'human_rl' mode game loop completes headlessly without crashing.

    In human_rl mode the CNN classifies each humanoid internally (to build the
    RL obs vector) and the RL recommender suggests an action — but the
    classification label is hidden from the player. The human makes the final
    decision. This test simulates a non-compliant human who always skips
    regardless of the recommendation — deliberately distinct from
    ``test_auto_mode``, which follows the RL recommendation every turn.

    Args:
        data_parser: Fresh DataParser fixture (function-scoped).
        sk_fresh: Fresh ScoreKeeper fixture (function-scoped).
        cnn_classifier: Session-scoped CNNClassifier with real weights.
        rl_recommender: Session-scoped RLRecommender with real weights.
    """
    while len(data_parser.unvisited) > 0 and sk_fresh.remaining_time > 0:
        humanoid = data_parser.get_random()
        probs = cnn_classifier.classify(humanoid)
        rl_recommender.recommend(
            humanoid, probs, sk_fresh
        )  # shown to human; human ignores
        sk_fresh.map_do_action(ActionState.SKIP, humanoid)

    _assert_valid_end_state(data_parser, sk_fresh)


def test_human_cnn_rl_mode_runs_to_completion(
    data_parser: DataParser,
    sk_fresh: ScoreKeeper,
    cnn_classifier: CNNClassifier,
    rl_recommender: RLRecommender,
) -> None:
    """Verify the 'human_cnn_rl' mode game loop completes headlessly without crashing.

    In human_cnn_rl mode the CNN classifies each humanoid AND the RL recommender
    suggests an action — both shown to the human — but the human makes the final
    decision. Probs are computed once and passed to the recommender, mirroring
    the production UI behavior (single forward pass per turn).
    This test simulates a compliant human who always follows the RL recommendation,
    contrasting with ``test_human_rl_mode`` which always ignores it.

    Args:
        data_parser: Fresh DataParser fixture (function-scoped).
        sk_fresh: Fresh ScoreKeeper fixture (function-scoped).
        cnn_classifier: Session-scoped CNNClassifier with real weights.
        rl_recommender: Session-scoped RLRecommender with real weights.
    """
    while len(data_parser.unvisited) > 0 and sk_fresh.remaining_time > 0:
        humanoid = data_parser.get_random()
        probs = cnn_classifier.classify(humanoid)
        action = rl_recommender.recommend(humanoid, probs, sk_fresh)  # human follows it
        sk_fresh.map_do_action(action, humanoid)

    _assert_valid_end_state(data_parser, sk_fresh)


# ---------------------------------------------------------------------------
# COV-20 — zombie contamination end-to-end
# ---------------------------------------------------------------------------


def test_zombie_contamination_triggers_on_scram(
    sk_fresh: ScoreKeeper,
) -> None:
    """Loading a zombie then scraming kills all healthy/injured passengers."""
    from gameplay.humanoid import Humanoid

    h_zombie = Humanoid(fp="consolidated_dataset/test_00007.png", state="zombie")
    h_healthy = Humanoid(fp="consolidated_dataset/test_00000.png", state="healthy")

    sk_fresh.save(h_healthy)
    sk_fresh.save(h_healthy)
    sk_fresh.save(h_zombie)
    sk_fresh.map_do_action(ActionState.SCRAM, h_zombie)

    assert sk_fresh.score["killed"] > 0
    assert sk_fresh.score["saved"] == 0


def test_machine_mode_runs_to_completion(
    data_parser: DataParser,
    sk_fresh: ScoreKeeper,
    cnn_classifier: CNNClassifier,
    rl_recommender: RLRecommender,
) -> None:
    """Verify the 'machine' mode game logic completes without crashing.

    In machine mode the CNN classifies each humanoid and the RL recommender
    selects the action, but the human is in the loop — they watch and press
    "Advance Turn" to step forward one turn at a time. This test runs the
    underlying game logic headlessly (no UI), verifying the classify →
    recommend → act pipeline rather than the UI interaction itself.

    Args:
        data_parser: Fresh DataParser fixture (function-scoped).
        sk_fresh: Fresh ScoreKeeper fixture (function-scoped).
        cnn_classifier: Session-scoped CNNClassifier with real weights.
        rl_recommender: Session-scoped RLRecommender with real weights.
    """
    while len(data_parser.unvisited) > 0 and sk_fresh.remaining_time > 0:
        humanoid = data_parser.get_random()
        probs = cnn_classifier.classify(humanoid)
        action = rl_recommender.recommend(humanoid, probs, sk_fresh)
        sk_fresh.map_do_action(action, humanoid)

    _assert_valid_end_state(data_parser, sk_fresh)
