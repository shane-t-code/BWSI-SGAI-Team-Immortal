import json
import os
import uuid
from typing import Any

import pandas as pd

from gameplay.constants import HEADCOUNT_WEIGHT, VALUE_WEIGHT
from gameplay.enums import ActionCost, ActionState
from gameplay.humanoid import Humanoid

# Convenience mappings to make converting these throughout the code less error prone
MAP_ACTION_STR_TO_INT: dict[str, int] = {s.value: i for i, s in enumerate(ActionState)}
MAP_ACTION_INT_TO_STR: list[str] = [s.value for s in ActionState]


class ScoreKeeper:
    """Tracks game state, scoring, and action history.

    Maintains the ambulance manifest, kill/save tallies, remaining time, and
    a per-episode action log. Also exposes the action helpers used by the RL
    environment (``map_do_action``) and the UI (``save``, ``squish``, etc.).

    Attributes:
        shift_length: Total game duration in minutes.
        capacity: Maximum number of humanoids the ambulance can hold at once.
        seed: Random seed recorded in ``summary.csv``. ``None`` if no seed was set.
        num_actions: Number of distinct actions (length of :class:`~gameplay.enums.ActionState`).
        current_run_id: UUID4 string identifying the current episode. Generated
            at construction and after each ``reset()``. Written to both
            ``log.csv`` and ``summary.csv`` so the two files can be joined.
        logger: Action log for the current episode.
        all_logs: Completed episode records. Each entry is a dict with keys
            ``run_id`` (str), ``log`` (list of action dicts), and ``score``
            (final score dict captured before reset).
        ambulance: Count of each humanoid type currently on board.
        score: Cumulative kill/save tallies for the current episode; keys are
            ``"killed"``, ``"saved"``, and ``"saved_value"`` (float — cumulative
            value delivered to hospital, post-scram, zombie-free).
        squished: Total number of squish actions taken this episode.
        remaining_time: Minutes left in the current shift.
        ambulance_value: Running sum of the values of humanoids currently on the ambulance.
    """

    shift_length: int
    capacity: int
    seed: int | None
    num_actions: int
    current_run_id: str
    logger: list[dict[str, Any]]
    all_logs: list[dict[str, Any]]
    ambulance: dict[str, int]
    score: dict[str, int | float]
    squished: int
    remaining_time: int
    ambulance_value: float

    def __init__(
        self, shift_length: int, capacity: int, seed: int | None = None
    ) -> None:
        """Initialize a ScoreKeeper.

        Args:
            shift_length: Total game duration in minutes.
            capacity: Maximum number of humanoids the ambulance can hold at once.
            seed: Random seed used for this run, recorded in ``summary.csv``.
                Pass ``None`` (default) if no seed was set.
        """
        self.shift_length = int(shift_length)  # minutes
        self.capacity = capacity
        self.seed = seed
        self.num_actions = len(ActionState)
        self.logger = []
        self.all_logs = []
        self._all_logs_written = (
            0  # cursor: how many all_logs entries have been written
        )
        self.current_run_id = str(uuid.uuid4())
        self.reset()

    def reset(self) -> None:
        """Reset game state for a new episode.

        Captures the completed episode's score and logger into ``all_logs``
        (if any actions were taken), generates a fresh UUID for the next
        episode, then clears per-episode state.

        If the ambulance has passengers at reset time (e.g. the RL training
        loop called reset() without calling get_score() first), a final scram
        is issued so in-flight passengers are correctly reflected in all_logs.
        """
        if sum(getattr(self, "ambulance", {}).values()) > 0:
            self.get_score()  # finalize in-flight passengers before capturing score
        prev_score = dict(getattr(self, "score", {}))
        if self.logger:
            self.all_logs.append(
                {
                    "run_id": self.current_run_id,
                    "log": self.logger,
                    "score": prev_score,  # captured before reset
                }
            )
            self.current_run_id = str(uuid.uuid4())

        self.ambulance = {"zombie": 0, "injured": 0, "healthy": 0, "corpse": 0}
        self.ambulance_value = 0.0
        self.score = {
            "killed": 0,
            "saved": 0,
            "saved_value": 0.0,
        }
        self.squished = 0
        self.remaining_time = int(self.shift_length)  # minutes
        self.logger = []

    def log_turn(
        self,
        humanoid: "Humanoid | None",
        action: str,
        predicted_class: int | None = None,
        recommended_action: str | None = None,
        confidence: str | None = None,
        obs: list[float] | None = None,
    ) -> None:
        """Append an action entry to the current episode log.

        Call this *after* confirming the action was executed (not blocked) so
        that no ghost rows are written for blocked actions. ``remaining_time``
        and ``capacity`` reflect the pre-action state at call time.

        When ``humanoid`` is ``None`` (e.g. the auto-scram triggered by
        ``get_score()`` at game end), humanoid fields are logged as ``None``.
        These appear as ``NaN`` after a CSV round-trip via pandas.

        Args:
            humanoid: The humanoid that was acted on, or ``None`` for synthetic
                entries (e.g. the final auto-scram from ``get_score()``).
            action: String name of the action taken (e.g. ``"save"``, ``"squish"``).
            predicted_class: Integer class index predicted by the classifier, or
                ``None`` if no classifier was used.
            recommended_action: Action string suggested by the recommender, or
                ``None`` if no recommender was used.
            confidence: Classifier confidence bucket — ``"low"``, ``"mid"``, or
                ``"high"`` — or ``None`` if no classifier was used.
            obs: Raw 15-float RL observation vector (pre-action) as a Python
                list, or ``None`` if not available (e.g. human UI modes).
                Stored as a JSON array string in the log for RL debugging.
        """
        self.logger.append(
            {
                "humanoid_ground_truth": (
                    humanoid.state if humanoid is not None else None
                ),
                "humanoid_predicted_class": predicted_class,
                "recommended_action": recommended_action,
                "confidence": confidence,
                "humanoid_fp": humanoid.fp if humanoid is not None else None,
                "action": action,
                "remaining_time": self.remaining_time,
                "humanoid_value": humanoid.value if humanoid is not None else None,
                "capacity": self.get_current_capacity(),
                "amb_zombie": self.ambulance["zombie"],
                "amb_injured": self.ambulance["injured"],
                "amb_healthy": self.ambulance["healthy"],
                "amb_corpse": self.ambulance["corpse"],
                "obs_json": json.dumps(obs) if obs is not None else None,
            }
        )

    def save_log(self, log_dir: str | None = None) -> None:
        """Append per-turn and per-game logs to ``log.csv`` and ``summary.csv``.

        Both files accumulate across runs — each call appends only new rows
        (episodes not yet written) rather than re-writing the full history.
        ``run_id`` (UUID4) identifies each episode and joins the two files:
        ``df = log.merge(summary, on="run_id")``.

        Safe to call multiple times — a write cursor (``_all_logs_written``)
        tracks which committed episodes have already been persisted, so
        double-calls and periodic mid-training saves produce no duplicates.

        ``log.csv`` — one row per turn, keyed by ``run_id``.
        ``summary.csv`` — one row per episode with final scores and reward.

        Args:
            log_dir: Directory to write log files into. Defaults to the
                current working directory if ``None``.
        """
        base = log_dir if log_dir is not None else "."
        # Only write committed episodes that haven't been persisted yet.
        new_episodes = self.all_logs[self._all_logs_written :]
        self._all_logs_written = len(self.all_logs)
        if self.logger:
            new_episodes = list(new_episodes) + [
                {
                    "run_id": self.current_run_id,
                    "log": list(
                        self.logger
                    ),  # copy — mid-episode append would corrupt the snapshot
                    "score": dict(self.score),
                }
            ]
        episodes = new_episodes
        if not episodes:
            return

        # log.csv — turn-by-turn (append; write header only on first write)
        turn_dfs = []
        for ep in episodes:
            df = pd.DataFrame(ep["log"])
            df["run_id"] = ep["run_id"]
            turn_dfs.append(df)
        log_path = os.path.join(base, "log.csv")
        log_exists = os.path.exists(log_path)
        pd.concat(turn_dfs).to_csv(
            log_path, mode="a", index=False, header=not log_exists
        )

        # summary.csv — one row per game (append; write header only on first write)
        summary_rows = []
        for ep in episodes:
            s = ep["score"]
            reward = (
                HEADCOUNT_WEIGHT * (s["saved"] - s["killed"])
                + VALUE_WEIGHT * s["saved_value"]
            )
            summary_rows.append(
                {
                    "run_id": ep["run_id"],
                    "shift_length": self.shift_length,
                    "capacity": self.capacity,
                    "num_turns": len(ep["log"]),
                    "killed": s["killed"],
                    "saved": s["saved"],
                    "saved_value": round(s["saved_value"], 6),
                    "reward": round(reward, 6),
                    "seed": self.seed,
                }
            )
        summary_path = os.path.join(base, "summary.csv")
        summary_exists = os.path.exists(summary_path)
        pd.DataFrame(summary_rows).to_csv(
            summary_path, mode="a", index=False, header=not summary_exists
        )

    def save(self, humanoid: Humanoid) -> None:
        """Load a humanoid onto the ambulance.

        Deducts :attr:`~gameplay.enums.ActionCost.SAVE` minutes from
        ``remaining_time`` and increments the appropriate ambulance slot.

        Args:
            humanoid: The humanoid to save.
        """
        self.remaining_time -= ActionCost.SAVE.value
        if humanoid.is_zombie():
            self.ambulance["zombie"] += 1
        elif humanoid.is_corpse():
            # Corpse takes a slot but is harmless — discarded at scram, no kill-rule trigger.
            # Extension point: consider penalizing wasted capacity here.
            self.ambulance["corpse"] += 1
        elif humanoid.is_injured():
            self.ambulance["injured"] += 1
        else:
            self.ambulance["healthy"] += 1
        if humanoid.is_injured() or humanoid.is_healthy():
            self.ambulance_value += humanoid.value

    def squish(self, humanoid: Humanoid) -> None:
        """Squish a humanoid (destroy in place).

        Deducts :attr:`~gameplay.enums.ActionCost.SQUISH` minutes. Counts as a
        kill if the humanoid is not a zombie or corpse.

        Args:
            humanoid: The humanoid to squish.
        """
        self.remaining_time -= ActionCost.SQUISH.value
        self.squished += 1
        # Extension point: reward squishing zombies or penalize squishing humans.
        # This is a two-step change: (1) add a counter here (e.g. self.zombie_squished += 1
        # if humanoid.is_zombie()), then (2) use that counter in get_cumulative_reward()
        # to adjust the saved/killed totals. Changes here alone have no effect on reward.
        if not (humanoid.is_zombie() or humanoid.is_corpse()):
            self.score["killed"] += 1

    def skip(self, humanoid: Humanoid) -> None:
        """Skip a humanoid (leave them behind).

        Deducts :attr:`~gameplay.enums.ActionCost.SKIP` minutes. Counts as a
        kill if the humanoid is injured (they won't survive without rescue).
        Skipping a healthy, zombie, or corpse humanoid deducts time but has no
        effect on the score.

        Args:
            humanoid: The humanoid to skip.
        """
        self.remaining_time -= ActionCost.SKIP.value
        if humanoid.is_injured():
            self.score["killed"] += 1

    def scram(self, humanoid: Humanoid | None = None, deduct_time: bool = True) -> None:
        """Drive the ambulance to the hospital and unload all passengers.

        ``humanoid`` is accepted for API consistency but unused — scram applies
        to all passengers currently on board, not a single humanoid.

        Time is deducted even when the ambulance is empty and even when
        ``remaining_time`` is already 0 or negative (no floor).

        Deducts :attr:`~gameplay.enums.ActionCost.SCRAM` minutes. If any zombie
        is on board, all injured and healthy passengers are killed. Corpses are
        silently discarded (already dead, no kill-rule trigger). All ambulance
        slots are cleared after unloading.

        Args:
            humanoid: The humanoid present when scram was triggered, or ``None``
                if called internally (e.g. from ``get_score``).
            deduct_time: If ``False``, skip the time deduction. Used by
                ``get_score()`` when scramming at game-end to prevent driving
                ``remaining_time`` further below zero after the shift has ended.
        """
        if deduct_time:
            self.remaining_time -= ActionCost.SCRAM.value
        if self.ambulance["zombie"] > 0:
            # Any zombie on board kills all injured and healthy passengers.
            # Corpses are unaffected — they are already dead and are simply discarded.
            self.score["killed"] += (
                self.ambulance["injured"] + self.ambulance["healthy"]
            )
        else:
            self.score["saved"] += self.ambulance["injured"] + self.ambulance["healthy"]
            self.score["saved_value"] += self.ambulance_value

        self.ambulance["zombie"] = 0
        self.ambulance["injured"] = 0
        self.ambulance["healthy"] = 0
        self.ambulance["corpse"] = 0
        self.ambulance_value = 0.0

    def available_action_space(self) -> list[bool]:
        """Return which actions are currently available.

        Returns:
            A list of bools in :class:`~gameplay.enums.ActionState` enum order.
            ``True`` means the action is available; ``False`` means it is
            disabled (e.g. Save is disabled when the ambulance is full).
        """
        action_dict = {s.value: True for s in ActionState}
        if self.remaining_time <= 0:
            action_dict["save"] = False
            action_dict["squish"] = False
            action_dict["skip"] = False
            # scram is always available — never disabled per design doc §8
        if self.at_capacity():
            action_dict["save"] = False
        return [action_dict[s.value] for s in ActionState]

    def map_do_action(self, action: ActionState, humanoid: Humanoid) -> bool:
        """Execute an action. Intended for RL use.

        Validates preconditions before acting (time remaining, capacity). Does
        nothing and returns ``False`` if the action is not currently available.

        Note:
            SCRAM is never blocked — it has no precondition guard. Calling SCRAM
            repeatedly continues to deduct ``ActionCost.SCRAM`` minutes on each
            call, with no floor on ``remaining_time``.

        Args:
            action: An :class:`~gameplay.enums.ActionState` member.
            humanoid: The humanoid to act on.

        Returns:
            ``True`` if the action was executed, ``False`` if it was blocked.
        """
        if action == ActionState.SAVE:
            if self.remaining_time <= 0 or self.at_capacity():
                return False
            self.save(humanoid)
        elif action == ActionState.SQUISH:
            if self.remaining_time <= 0:
                return False
            self.squish(humanoid)
        elif action == ActionState.SKIP:
            if self.remaining_time <= 0:
                return False
            self.skip(humanoid)
        elif action == ActionState.SCRAM:
            self.scram(humanoid)
        return True

    def get_cumulative_reward(self) -> float:
        """Return the current weighted reward combining headcount and value signals.

        Accounts for humanoids currently on the ambulance as if scram were
        called now, without modifying state.

        The step reward in ``train_env.step()`` is the *delta* between two
        calls: ``reward = get_cumulative_reward() - previous``. This means the
        reward for a single action can be a large swing, not just ±1.

        Example — zombie-contamination swing:
            Assumptions: no previously scrammed humanoids (score starts at 0);
            each injured humanoid has ``value = 1.0``.
            Ambulance has 3 injured (hypothetical saved = +3, saved_value = +3.0).
            Cumulative reward before save: 0.5 * (3 - 0) + 0.5 * 3.0 = 3.0.
            Agent saves a zombie: ambulance now has 1 zombie + 3 injured.
            Hypothetical saved drops to 0; killed jumps to +3; ambulance_value is lost.
            Cumulative reward after save: 0.5 * (0 - 3) + 0.5 * 0.0 = -1.5.
            Step delta = -1.5 - 3.0 = -4.5 for that one save action.

        Returns:
            Weighted reward: ``α × (saved − killed) + β × saved_value``, where
            α = ``HEADCOUNT_WEIGHT`` and β = ``VALUE_WEIGHT`` (both ``0.5`` by
            default — adjust in ``gameplay/constants.py``). Includes undelivered
            ambulance passengers hypothetically.
            Zombie contamination sets ``saved_value`` contribution to zero for
            in-flight passengers (their value is lost).
        """
        killed = self.score["killed"]
        saved = self.score["saved"]
        saved_value = self.score["saved_value"]
        if self.ambulance["zombie"] > 0:
            killed += self.ambulance["injured"] + self.ambulance["healthy"]
            # ambulance_value is lost — zombie contamination wipes all value
        else:
            saved += self.ambulance["injured"] + self.ambulance["healthy"]
            saved_value += self.ambulance_value

        # Reward = HEADCOUNT_WEIGHT × (saved − killed) + VALUE_WEIGHT × saved_value
        # Both weights are defined in gameplay/constants.py and must sum to 1.
        #
        # Notice that (saved - killed) can swing by up to
        # 2 × (capacity − 1) = 18 on a zombie contamination event (worst case:
        # ambulance full with capacity−1 injured plus 1 zombie), while
        # saved_value changes by at most 1.0
        # per step. Try adjusting HEADCOUNT_WEIGHT in constants.py — what happens
        # to agent behavior at the extremes (1.0 vs 0.0)?
        return HEADCOUNT_WEIGHT * (saved - killed) + VALUE_WEIGHT * saved_value

    def get_current_capacity(self) -> int:
        """Return the number of humanoids currently on the ambulance.

        Returns:
            Total occupied ambulance slots.
        """
        return sum(self.ambulance.values())

    def at_capacity(self) -> bool:
        """Return True if the ambulance is full.

        Returns:
            ``True`` if occupied slots >= ``capacity``.
        """
        return sum(self.ambulance.values()) >= self.capacity

    def get_score(self) -> dict[str, int | float]:
        """Return the final score, scramming first if passengers remain.

        If the ambulance is non-empty, logs a synthetic scram entry
        (``humanoid=None``) and then calls ``scram()``. The log entry captures
        the final game-end state — it's a real game event with real consequences
        (passengers on board are saved or killed depending on zombie presence).
        Humanoid fields in this entry are ``None``, which appear as ``NaN``
        after a CSV round-trip via pandas.

        Returns:
            Dict with ``"killed"`` (int), ``"saved"`` (int), and
            ``"saved_value"`` (float) totals.
        """
        if sum(self.ambulance.values()) > 0:
            self.log_turn(None, "scram")
            self.scram(
                deduct_time=False
            )  # game is already over — don't push remaining_time further negative
        return self.score

    @staticmethod
    def get_action_idx(class_string: str) -> int:
        """Return the integer index for an action string.

        Args:
            class_string: An :class:`~gameplay.enums.ActionState` value string
                (e.g. ``"save"``).

        Returns:
            Integer index corresponding to the action in enum order.
        """
        return MAP_ACTION_STR_TO_INT[class_string]

    @staticmethod
    def get_action_string(class_idx: int) -> str:
        """Return the action string for an integer index.

        Args:
            class_idx: Integer index of the action in enum order.

        Returns:
            Action string (e.g. ``"save"``).
        """
        return MAP_ACTION_INT_TO_STR[class_idx]

    @staticmethod
    def get_all_actions() -> list[str]:
        """Return all action strings in enum order.

        Returns:
            List of action strings (e.g. ``["save", "squish", "skip", "scram"]``).

        Note:
            Returns a copy — safe to mutate; the original mapping is not affected.
        """
        return list(MAP_ACTION_INT_TO_STR)
