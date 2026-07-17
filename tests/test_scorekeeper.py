"""Unit tests for the ScoreKeeper class.

Covers initial state, every action method (save/squish/skip/scram), time
deduction, ambulance manifest, score tallying, capacity checks, action space
gating, cumulative reward, reset, and static index/string helpers.
No model inference or Tkinter imports are used.
"""

import pytest

from gameplay.constants import AMBULANCE_CAPACITY, SHIFT_LENGTH
from gameplay.enums import ActionState
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_state_is_clean(sk_fresh):
    """Assert that a fresh ScoreKeeper has an empty ambulance, zero score, and full remaining time."""
    assert sk_fresh.remaining_time == SHIFT_LENGTH
    assert sk_fresh.score["killed"] == 0
    assert sk_fresh.score["saved"] == 0
    assert sum(sk_fresh.ambulance.values()) == 0


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_deducts_30_minutes(sk_fresh, h_healthy):
    """Assert that save() deducts exactly 30 minutes from remaining_time."""
    before = sk_fresh.remaining_time
    sk_fresh.save(h_healthy)
    assert sk_fresh.remaining_time == before - 30


def test_save_adds_to_ambulance(sk_fresh, h_healthy):
    """Assert that save() increments the 'healthy' ambulance slot by 1."""
    sk_fresh.save(h_healthy)
    assert sk_fresh.ambulance["healthy"] == 1


def test_save_zombie_adds_to_zombie_slot(sk_fresh, h_zombie):
    """Assert that save() increments the 'zombie' ambulance slot when saving a zombie."""
    sk_fresh.save(h_zombie)
    assert sk_fresh.ambulance["zombie"] == 1


def test_save_injured_adds_to_injured_slot(sk_fresh, h_injured):
    """Assert that save() increments the 'injured' ambulance slot when saving an injured humanoid."""
    sk_fresh.save(h_injured)
    assert sk_fresh.ambulance["injured"] == 1


# ---------------------------------------------------------------------------
# squish()
# ---------------------------------------------------------------------------


def test_squish_deducts_5_minutes(sk_fresh, h_healthy):
    """Assert that squish() deducts exactly 5 minutes from remaining_time."""
    before = sk_fresh.remaining_time
    sk_fresh.squish(h_healthy)
    assert sk_fresh.remaining_time == before - 5


def test_squish_healthy_increments_killed(sk_fresh, h_healthy):
    """Assert that squish() increments score['killed'] when the humanoid is healthy."""
    sk_fresh.squish(h_healthy)
    assert sk_fresh.score["killed"] == 1


def test_squish_injured_increments_killed(sk_fresh, h_injured):
    """Assert that squish() increments score['killed'] when the humanoid is injured."""
    sk_fresh.squish(h_injured)
    assert sk_fresh.score["killed"] == 1


def test_squish_zombie_does_not_increment_killed(sk_fresh, h_zombie):
    """Assert that squish() does NOT increment score['killed'] when the humanoid is a zombie."""
    sk_fresh.squish(h_zombie)
    assert sk_fresh.score["killed"] == 0


def test_squish_corpse_does_not_increment_killed(sk_fresh, h_corpse):
    """Assert that squish() does NOT increment score['killed'] when the humanoid is a corpse."""
    sk_fresh.squish(h_corpse)
    assert sk_fresh.score["killed"] == 0


def test_squish_increments_squished_counter(sk_fresh, h_zombie):
    """Assert that squish() always increments the squished counter, regardless of humanoid type."""
    before = sk_fresh.squished
    sk_fresh.squish(h_zombie)
    assert sk_fresh.squished == before + 1


# ---------------------------------------------------------------------------
# skip()
# ---------------------------------------------------------------------------


def test_skip_deducts_15_minutes(sk_fresh, h_healthy):
    """Assert that skip() deducts exactly 15 minutes from remaining_time."""
    before = sk_fresh.remaining_time
    sk_fresh.skip(h_healthy)
    assert sk_fresh.remaining_time == before - 15


def test_skip_injured_increments_killed(sk_fresh, h_injured):
    """Assert that skip() increments score['killed'] when the skipped humanoid is injured."""
    sk_fresh.skip(h_injured)
    assert sk_fresh.score["killed"] == 1


def test_skip_healthy_does_not_increment_killed(sk_fresh, h_healthy):
    """Assert that skip() does NOT increment score['killed'] when the skipped humanoid is healthy."""
    sk_fresh.skip(h_healthy)
    assert sk_fresh.score["killed"] == 0


def test_skip_zombie_does_not_increment_killed(sk_fresh, h_zombie):
    """Assert that skip() does NOT increment score['killed'] for a zombie (zombies are not 'killed')."""
    sk_fresh.skip(h_zombie)
    assert sk_fresh.score["killed"] == 0


def test_skip_corpse_does_not_increment_killed(sk_fresh, h_corpse):
    """Assert that skip() does NOT increment score['killed'] for a corpse."""
    sk_fresh.skip(h_corpse)
    assert sk_fresh.score["killed"] == 0


# ---------------------------------------------------------------------------
# scram()
# ---------------------------------------------------------------------------


def test_scram_deducts_120_minutes(sk_fresh, h_healthy):
    """Assert that scram() deducts exactly 120 minutes from remaining_time."""
    before = sk_fresh.remaining_time
    sk_fresh.scram(h_healthy)
    assert sk_fresh.remaining_time == before - 120


def test_scram_with_no_zombie_increments_saved(sk_fresh, h_healthy):
    """Assert that scram() counts injured+healthy passengers as saved when no zombie is on board."""
    sk_fresh.ambulance["healthy"] = 2
    sk_fresh.ambulance["injured"] = 1
    sk_fresh.scram()
    assert sk_fresh.score["saved"] == 3


def test_scram_with_zombie_counts_humans_as_killed(sk_fresh, h_healthy):
    """Assert that scram() counts injured+healthy passengers as killed when a zombie is on board (zombie not counted)."""
    sk_fresh.ambulance["zombie"] = 1
    sk_fresh.ambulance["healthy"] = 2
    sk_fresh.ambulance["injured"] = 1
    sk_fresh.scram()
    assert (
        sk_fresh.score["killed"] == 3
    )  # healthy(2) + injured(1); zombie itself not counted


def test_scram_zeros_ambulance(sk_fresh):
    """Assert that scram() clears all ambulance slots to 0 after unloading."""
    sk_fresh.ambulance["healthy"] = 3
    sk_fresh.ambulance["zombie"] = 1
    sk_fresh.ambulance["injured"] = 1
    sk_fresh.ambulance["corpse"] = 1
    sk_fresh.scram()
    assert sum(sk_fresh.ambulance.values()) == 0


def test_scram_corpses_silently_discarded(sk_fresh):
    """Assert that corpse passengers are not counted in saved or killed after scram, but time is still deducted."""
    sk_fresh.ambulance["corpse"] = 3
    saved_before = sk_fresh.score["saved"]
    killed_before = sk_fresh.score["killed"]
    time_before = sk_fresh.remaining_time
    sk_fresh.scram()
    assert sk_fresh.score["saved"] == saved_before
    assert sk_fresh.score["killed"] == killed_before
    assert sk_fresh.remaining_time == time_before - 120


# ---------------------------------------------------------------------------
# at_capacity()
# ---------------------------------------------------------------------------


def test_at_capacity_true_when_full(sk_at_capacity):
    """Assert that at_capacity() returns True when the ambulance is completely full."""
    assert sk_at_capacity.at_capacity() is True


def test_at_capacity_false_when_not_full(sk_fresh):
    """Assert that at_capacity() returns False on a fresh ScoreKeeper with an empty ambulance."""
    assert sk_fresh.at_capacity() is False


# ---------------------------------------------------------------------------
# available_action_space()
# ---------------------------------------------------------------------------


def test_available_action_space_save_disabled_at_capacity(sk_at_capacity):
    """Assert that save is False in the action space when the ambulance is at capacity."""
    space = sk_at_capacity.available_action_space()
    assert space[0] is False  # index 0 == save


def test_available_action_space_save_disabled_when_time_zero(sk_fresh):
    """Assert that save is False in the action space when remaining_time is 0."""
    sk_fresh.remaining_time = 0
    space = sk_fresh.available_action_space()
    assert space[0] is False


def test_available_action_space_scram_always_available(sk_fresh):
    """Assert that scram is always True in the action space even when time is exhausted."""
    sk_fresh.remaining_time = 0
    space = sk_fresh.available_action_space()
    assert space[3] is True  # index 3 == scram


def test_available_action_space_all_true_when_fresh(sk_fresh):
    """Assert that all four actions are available on a fresh ScoreKeeper."""
    space = sk_fresh.available_action_space()
    assert space == [True, True, True, True]


# ---------------------------------------------------------------------------
# map_do_action()
# ---------------------------------------------------------------------------


def test_map_do_action_save_blocked_at_capacity_returns_false(
    sk_at_capacity, h_healthy
):
    """Assert that map_do_action() returns False for SAVE when the ambulance is at capacity."""
    result = sk_at_capacity.map_do_action(ActionState.SAVE, h_healthy)
    assert result is False


def test_map_do_action_squish_blocked_at_time_zero_returns_false(sk_fresh, h_zombie):
    """Assert that map_do_action() returns False for SQUISH when remaining_time is 0."""
    sk_fresh.remaining_time = 0
    result = sk_fresh.map_do_action(ActionState.SQUISH, h_zombie)
    assert result is False


def test_map_do_action_skip_blocked_at_time_zero_returns_false(sk_fresh, h_healthy):
    """Assert that map_do_action() returns False for SKIP when remaining_time is 0."""
    sk_fresh.remaining_time = 0
    result = sk_fresh.map_do_action(ActionState.SKIP, h_healthy)
    assert result is False


def test_map_do_action_scram_succeeds_at_time_zero(sk_fresh, h_healthy):
    """Assert that map_do_action() executes SCRAM even when remaining_time is 0."""
    sk_fresh.remaining_time = 0
    result = sk_fresh.map_do_action(ActionState.SCRAM, h_healthy)
    assert result is True


# ---------------------------------------------------------------------------
# get_score() — side effect contract
# ---------------------------------------------------------------------------


def test_get_score_depletes_ambulance_via_scram(sk_fresh, h_healthy):
    """Assert that get_score() calls scram() when the ambulance is non-empty, zeroing all slots."""
    sk_fresh.ambulance["healthy"] = 2
    sk_fresh.get_score()
    assert sum(sk_fresh.ambulance.values()) == 0


def test_get_score_does_not_deduct_time_when_ambulance_nonempty(sk_fresh, h_healthy):
    """Assert that get_score() does not deduct time from remaining_time.

    get_score() is a post-game finalization step — it calls scram(deduct_time=False)
    so that remaining_time is not driven further negative after the shift has ended.
    """
    sk_fresh.ambulance["healthy"] = 1
    before = sk_fresh.remaining_time
    sk_fresh.get_score()
    assert sk_fresh.remaining_time == before


def test_get_score_empty_ambulance_does_not_deduct_time(sk_fresh):
    """Assert that get_score() does not call scram() when the ambulance is empty."""
    before = sk_fresh.remaining_time
    sk_fresh.get_score()
    assert sk_fresh.remaining_time == before


# ---------------------------------------------------------------------------
# sk_near_end boundary behavior
# ---------------------------------------------------------------------------


def test_near_end_squish_still_executes(sk_near_end, h_zombie):
    """Assert that squish() executes and deducts time when remaining_time is 5 (cost==5, gate is <=0)."""
    before = sk_near_end.remaining_time
    sk_near_end.squish(h_zombie)
    assert sk_near_end.remaining_time == before - 5


def test_near_end_save_blocked_at_exactly_zero(sk_fresh, h_healthy):
    """Assert that map_do_action() blocks save when remaining_time is exactly 0 (gate is <=0, not <cost)."""
    sk_fresh.remaining_time = 0
    result = sk_fresh.map_do_action(ActionState.SAVE, h_healthy)
    assert result is False


# ---------------------------------------------------------------------------
# get_cumulative_reward()
# ---------------------------------------------------------------------------


def test_get_cumulative_reward_empty_ambulance(sk_fresh):
    """Assert that get_cumulative_reward() returns 0 when no actions have been taken."""
    assert sk_fresh.get_cumulative_reward() == 0


def test_get_cumulative_reward_with_passengers_no_zombie(sk_fresh):
    """get_cumulative_reward() with 3 healthy on board == 0.5*3 + 0.5*0.0 == 1.5."""
    sk_fresh.ambulance["healthy"] = 3
    assert sk_fresh.ambulance_value == pytest.approx(
        0.0
    )  # precondition: testing headcount-only path
    assert sk_fresh.get_cumulative_reward() == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_state(sk_fresh, h_healthy, h_zombie):
    """Assert that reset() clears ambulance, score, squished counter, and restores remaining_time."""
    sk_fresh.save(h_healthy)
    sk_fresh.squish(h_zombie)
    sk_fresh.reset()
    assert sum(sk_fresh.ambulance.values()) == 0
    assert sk_fresh.score["killed"] == 0
    assert sk_fresh.score["saved"] == 0
    assert sk_fresh.squished == 0
    assert sk_fresh.remaining_time == SHIFT_LENGTH


# ---------------------------------------------------------------------------
# Static helpers — get_action_idx / get_action_string
# ---------------------------------------------------------------------------


def test_get_action_idx_save_is_0():
    """Assert that get_action_idx('save') returns 0."""
    assert ScoreKeeper.get_action_idx("save") == 0


def test_get_action_string_0_is_save():
    """Assert that get_action_string(0) returns 'save'."""
    assert ScoreKeeper.get_action_string(0) == "save"


# ---------------------------------------------------------------------------
# COV-03 — save_log()
# ---------------------------------------------------------------------------


def test_save_log_writes_csv_with_correct_episode_ids(tmp_path, monkeypatch, h_healthy):
    """save_log() writes log.csv with a distinct UUID4 run_id per episode."""
    import uuid

    import pandas as pd

    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)
    sk.reset()  # flushes logger into all_logs
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)
    sk.save_log()

    df = pd.read_csv(tmp_path / "log.csv")
    run_ids = df["run_id"].unique()
    assert len(run_ids) == 2
    for rid in run_ids:
        uuid.UUID(rid, version=4)  # raises if not a valid UUID4


def test_save_log_no_double_append_after_reset(tmp_path, monkeypatch, h_healthy):
    """save_log() after one action + reset writes exactly two data rows.

    reset() with passengers on the ambulance calls get_score() to finalize
    in-flight passengers, which appends a synthetic 'scram' log entry. So the
    log for this episode contains: the original 'save' turn + the auto-scram.
    """
    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)
    sk.reset()
    sk.save_log()

    import pandas as pd

    df = pd.read_csv(tmp_path / "log.csv")
    assert len(df) == 2  # "save" turn + auto-scram from reset() finalization
    assert list(df["action"]) == ["save", "scram"]


def test_save_log_returns_early_when_nothing_to_write(tmp_path, monkeypatch):
    """save_log() writes neither log.csv nor summary.csv when there is nothing to log."""
    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.save_log()
    assert not (tmp_path / "log.csv").exists()
    assert not (tmp_path / "summary.csv").exists()


# ---------------------------------------------------------------------------
# COV-04 — reset() logger guard
# ---------------------------------------------------------------------------


def test_reset_appends_nonempty_logger_to_all_logs(sk_fresh, h_healthy):
    """reset() after a logged action appends an episode record to all_logs and clears logger."""
    sk_fresh.log_turn(h_healthy, "save")
    sk_fresh.save(h_healthy)
    sk_fresh.reset()
    assert len(sk_fresh.all_logs) == 1
    assert set(sk_fresh.all_logs[0].keys()) == {"run_id", "log", "score"}
    assert sk_fresh.logger == []


def test_reset_does_not_append_empty_logger(sk_fresh, h_healthy):
    """A second reset() with an empty logger does not grow all_logs further."""
    sk_fresh.log_turn(h_healthy, "save")
    sk_fresh.save(h_healthy)
    sk_fresh.reset()  # appends
    sk_fresh.reset()  # logger is empty — should NOT append
    assert len(sk_fresh.all_logs) == 1


# ---------------------------------------------------------------------------
# COV-05 — get_cumulative_reward() zombie contamination
# ---------------------------------------------------------------------------


def test_get_cumulative_reward_with_zombie_on_board(sk_fresh):
    """Ambulance with 1 zombie + 3 injured → cumulative reward == 0.5*(0-3) + 0.5*0.0 == -1.5."""
    sk_fresh.ambulance["zombie"] = 1
    sk_fresh.ambulance["injured"] = 3
    assert sk_fresh.get_cumulative_reward() == pytest.approx(-1.5)


# ---------------------------------------------------------------------------
# COV-06 — map_do_action() simultaneous time=0 and at-capacity
# ---------------------------------------------------------------------------


def test_map_do_action_save_blocked_when_time_zero_and_at_capacity(h_healthy):
    """map_do_action(SAVE) returns False when both time=0 and ambulance is full."""
    sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY)
    sk.remaining_time = 0
    sk.ambulance["healthy"] = sk.capacity
    result = sk.map_do_action(ActionState.SAVE, h_healthy)
    assert result is False


# ---------------------------------------------------------------------------
# COV-11 — scram() logging guard
# ---------------------------------------------------------------------------


def test_scram_without_humanoid_does_not_log(sk_fresh):
    """scram() with no prior log_turn call leaves logger empty."""
    sk_fresh.scram()
    assert sk_fresh.logger == []


def test_log_turn_before_scram_records_entry(sk_fresh, h_healthy):
    """log_turn() called before scram() appends exactly one entry to the logger."""
    sk_fresh.log_turn(h_healthy, "scram")
    sk_fresh.scram(h_healthy)
    assert len(sk_fresh.logger) == 1
    assert sk_fresh.logger[0]["action"] == "scram"


# ---------------------------------------------------------------------------
# COV-12 — available_action_space() all disabled except scram at time=0
# ---------------------------------------------------------------------------


def test_available_action_space_all_disabled_except_scram_at_time_zero(sk_fresh):
    """At remaining_time=0, action space is [False, False, False, True]."""
    sk_fresh.remaining_time = 0
    assert sk_fresh.available_action_space() == [False, False, False, True]


# ---------------------------------------------------------------------------
# COV-13 — log() key/value correctness
# ---------------------------------------------------------------------------


def test_log_turn_appends_correct_keys_and_values(sk_fresh, h_healthy):
    """log_turn() before save() records correct keys and values in logger[0]."""
    sk_fresh.log_turn(h_healthy, "save", predicted_class=1, recommended_action="save")
    sk_fresh.save(h_healthy)
    entry = sk_fresh.logger[0]
    assert entry["humanoid_ground_truth"] == h_healthy.state
    assert entry["humanoid_predicted_class"] == 1
    assert entry["recommended_action"] == "save"
    assert entry["humanoid_fp"] == h_healthy.fp
    assert entry["action"] == "save"
    assert isinstance(entry["remaining_time"], int)
    assert isinstance(entry["capacity"], int)


# ---------------------------------------------------------------------------
# COV-17 — scram() both sides of contamination
# ---------------------------------------------------------------------------


def test_scram_with_zombie_counts_all_as_killed_saved_is_zero(sk_fresh):
    """When zombie is on board, score['saved'] remains 0 after scram."""
    sk_fresh.ambulance["zombie"] = 1
    sk_fresh.ambulance["healthy"] = 2
    sk_fresh.ambulance["injured"] = 1
    sk_fresh.scram()
    assert sk_fresh.score["saved"] == 0


def test_scram_without_zombie_does_not_kill(sk_fresh):
    """scram() with no zombie on board does not increment score['killed']."""
    sk_fresh.ambulance["healthy"] = 3
    sk_fresh.scram()
    assert sk_fresh.score["killed"] == 0


# ---------------------------------------------------------------------------
# COV-23 — get_action_idx() invalid key raises KeyError
# ---------------------------------------------------------------------------


def test_get_action_idx_raises_key_error_on_invalid_action():
    """get_action_idx() raises KeyError for an unrecognized action string."""
    with pytest.raises(KeyError):
        ScoreKeeper.get_action_idx("teleport")


# ---------------------------------------------------------------------------
# COV-24 — get_current_capacity()
# ---------------------------------------------------------------------------


def test_get_current_capacity_zero_when_empty(sk_fresh):
    """get_current_capacity() returns 0 on a fresh ScoreKeeper."""
    assert sk_fresh.get_current_capacity() == 0


def test_get_current_capacity_increments_with_passengers(
    sk_fresh, h_healthy, h_injured
):
    """get_current_capacity() reflects the total number of passengers on board."""
    sk_fresh.ambulance["healthy"] = 2
    sk_fresh.ambulance["injured"] = 1
    assert sk_fresh.get_current_capacity() == 3


# ---------------------------------------------------------------------------
# humanoid value tracking
# ---------------------------------------------------------------------------


def test_saved_value_initial_zero():
    """After init, score['saved_value'] == 0.0."""
    sk = ScoreKeeper(shift_length=720, capacity=10)
    assert sk.score["saved_value"] == 0.0


def test_ambulance_value_initial_zero():
    """After init, ambulance_value == 0.0."""
    sk = ScoreKeeper(shift_length=720, capacity=10)
    assert sk.ambulance_value == 0.0


def test_save_accumulates_ambulance_value():
    """Saving a humanoid with value=0.6 sets ambulance_value == 0.6."""
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="injured", value=0.6)
    sk.save(h)
    assert sk.ambulance_value == pytest.approx(0.6)


def test_scram_clean_delivers_value():
    """Scram with no zombie delivers ambulance_value to saved_value and clears ambulance_value."""
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="injured", value=0.4)
    sk.save(h)
    sk.scram()
    assert sk.score["saved_value"] == pytest.approx(0.4)
    assert sk.ambulance_value == 0.0


def test_scram_zombie_wipes_value():
    """Scram with zombie on board zeroes saved_value and clears ambulance_value.

    Asserts ambulance_value is non-zero before the scram so the post-scram
    zero is meaningful — without this, the assertion would pass trivially
    because saved_value starts at 0.0 regardless.
    """
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h_injured = Humanoid(fp="injured.png", state="injured", value=0.4)
    h_zombie = Humanoid(fp="zombie.png", state="zombie", value=0.0)
    sk.save(h_injured)
    sk.save(h_zombie)
    assert sk.ambulance_value == pytest.approx(0.4)  # value present before scram
    sk.scram()
    assert sk.score["saved_value"] == 0.0
    assert sk.ambulance_value == 0.0


def test_get_cumulative_reward_includes_value():
    """get_cumulative_reward() for an injured humanoid with value=0.8 returns 0.5*1 + 0.5*0.8 == 0.9."""
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="injured", value=0.8)
    sk.save(h)
    assert sk.get_cumulative_reward() == pytest.approx(0.9)


def test_reset_clears_value():
    """reset() clears saved_value and ambulance_value back to 0.0."""
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="injured", value=0.5)
    sk.save(h)
    sk.scram()
    sk.reset()
    assert sk.score["saved_value"] == 0.0
    assert sk.ambulance_value == 0.0


def test_get_score_has_saved_value_key():
    """get_score() returns a dict that includes the 'saved_value' key."""
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="injured", value=0.3)
    sk.save(h)
    sk.scram()
    result = sk.get_score()
    assert "saved_value" in result


# ---------------------------------------------------------------------------
# COV-30 — save_log() column correctness and append behavior
# ---------------------------------------------------------------------------

LOG_COLUMNS = {
    "humanoid_ground_truth",
    "humanoid_predicted_class",
    "recommended_action",
    "confidence",
    "confidence_score",
    "humanoid_fp",
    "action",
    "remaining_time",
    "humanoid_value",
    "capacity",
    "amb_zombie",
    "amb_injured",
    "amb_healthy",
    "amb_corpse",
    "obs_json",
    "run_id",
}
SUMMARY_COLUMNS = {
    "run_id",
    "shift_length",
    "capacity",
    "num_turns",
    "killed",
    "saved",
    "saved_value",
    "reward",
    "seed",
}


def test_save_log_log_csv_columns(tmp_path, monkeypatch, h_healthy):
    """log.csv has exactly the expected column set."""
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=720, capacity=10)
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)
    sk.save_log()
    df = pd.read_csv(tmp_path / "log.csv")
    assert set(df.columns) == LOG_COLUMNS


def test_save_log_summary_csv_columns(tmp_path, monkeypatch, h_healthy):
    """summary.csv has exactly the expected column set, including seed."""
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=720, capacity=10, seed=42)
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)
    sk.save_log()
    df = pd.read_csv(tmp_path / "summary.csv")
    assert set(df.columns) == SUMMARY_COLUMNS


def test_save_log_summary_seed_recorded(tmp_path, monkeypatch, h_healthy):
    """summary.csv records the seed passed to ScoreKeeper."""
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=720, capacity=10, seed=99)
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)
    sk.save_log()
    df = pd.read_csv(tmp_path / "summary.csv")
    assert df["seed"].iloc[0] == 99


def test_save_log_summary_reward_formula(tmp_path, monkeypatch):
    """summary.csv reward == 0.5*(saved-killed) + 0.5*saved_value."""
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="injured", value=0.4)
    sk.log_turn(h, "save")
    sk.save(h)
    sk.log_turn(h, "scram")
    sk.scram()  # delivers 1 injured, saved_value=0.4
    sk.save_log()
    df = pd.read_csv(tmp_path / "summary.csv")
    expected = 0.5 * (1 - 0) + 0.5 * 0.4
    assert df["reward"].iloc[0] == pytest.approx(expected, abs=1e-5)


def test_save_log_appends_across_invocations(tmp_path, monkeypatch, h_healthy):
    """Two episodes on one ScoreKeeper accumulate rows — save_log() at end covers all_logs path.

    Uses a single ScoreKeeper with reset() between episodes so that the
    all_logs accumulation path is exercised. reset() auto-scraps in-flight
    passengers and appends the completed episode to all_logs; calling
    save_log() once at the end writes out both episodes in one pass.
    """
    import pandas as pd

    monkeypatch.chdir(tmp_path)

    sk = ScoreKeeper(shift_length=720, capacity=10)

    # Episode 1: save one healthy humanoid, then reset (auto-scrams in-flight)
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)
    sk.reset()  # finalizes episode 1 → appended to all_logs; auto-scram adds a log entry

    # Episode 2: save again on the same instance
    sk.log_turn(h_healthy, "save")
    sk.save(h_healthy)

    # Single save_log() call writes all_logs (episode 1) + current logger (episode 2)
    sk.save_log()

    log_df = pd.read_csv(tmp_path / "log.csv")
    summary_df = pd.read_csv(tmp_path / "summary.csv")
    # Episode 1 contributes 2 log turns (explicit "save" + auto-scram from reset())
    # Episode 2 contributes 1 log turn (explicit "save", ambulance not yet scrammed)
    assert len(log_df) == 3
    assert len(summary_df) == 2  # one summary row per episode
    assert log_df["run_id"].nunique() == 2
    assert summary_df["run_id"].nunique() == 2


def test_save_log_captures_score_after_get_score(tmp_path, monkeypatch):
    """Regression: passengers on the ambulance are counted in summary.csv when
    get_score() is called before save_log() (as _run_auto does).
    Prior bug: save_log() was called first, so in-flight passengers were missing."""
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="healthy", value=0.5)
    sk.log_turn(h, "save")
    sk.save(h)  # passenger on ambulance, not yet scrammed
    sk.get_score()  # scrames → counts as saved
    sk.save_log()

    df = pd.read_csv(tmp_path / "summary.csv")
    assert df["saved"].iloc[0] == 1
    assert df["saved_value"].iloc[0] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# COV-33 — log_turn() with humanoid=None (auto-scram from get_score)
# ---------------------------------------------------------------------------


def test_log_turn_none_humanoid_csv_roundtrip(tmp_path, monkeypatch):
    """log_turn() with humanoid=None writes None fields that read back as NaN.

    Documents the actual serialization contract: None → NaN via pandas CSV
    round-trip. Covers the synthetic scram entry logged by get_score() when
    passengers are still on board at game end.
    """
    import math

    import pandas as pd

    monkeypatch.chdir(tmp_path)
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="test.png", state="healthy", value=0.5)
    sk.log_turn(h, "save")
    sk.save(h)
    # get_score() triggers an internal scram and logs humanoid=None
    sk.get_score()
    sk.save_log()

    df = pd.read_csv(tmp_path / "log.csv")
    # Last row is the synthetic auto-scram entry
    auto_scram_row = df[df["action"] == "scram"].iloc[-1]
    assert math.isnan(auto_scram_row["humanoid_predicted_class"])
    assert pd.isna(auto_scram_row["humanoid_fp"])
    assert pd.isna(auto_scram_row["humanoid_ground_truth"])
