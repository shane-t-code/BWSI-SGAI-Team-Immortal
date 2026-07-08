from typing import Any
import numpy as np

from gameplay.constants import (
    BLOCKED_ACTION_PENALTY,
    CNN_CONFIDENCE_HIGH,
    CNN_CONFIDENCE_MID,
)
from gameplay.enums import ActionState
from gameplay.humanoid import Humanoid
from game_interfaces.base_env import GameEnv
from game_interfaces.data_parser import DataParser
from game_interfaces.obs_builder import build_obs
from gameplay.scorekeeper import ScoreKeeper
from models.HumanoidClassifier import HumanoidClassifier


class TrainEnv(GameEnv):
    """RL training environment.

    Extends :class:`~game_interfaces.base_env.GameEnv` with classifier-based
    humanoid classification. Each step classifies the current humanoid, fills in
    the ``humanoid_class``, ``humanoid_confidence``, and ``humanoid_value_bin``
    observation slots, and executes the chosen action.

    Attributes:
        classifier: HumanoidClassifier used to produce humanoid class probabilities.
        humanoid_probs: Raw 4-float probability vector for the current humanoid.
        humanoid: The humanoid drawn at the start of the current step.
    """

    classifier: HumanoidClassifier
    humanoid_probs: np.ndarray
    humanoid: Humanoid | None

    def __init__(
        self,
        root: Any,
        w: int,
        h: int,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
        classifier: HumanoidClassifier,
        img_data_root: str = "data",
        display: bool = False,
    ) -> None:
        """Initialize the RL training environment.

        Args:
            root: Tkinter root window, or ``None`` when ``display`` is False.
            w: Window width in pixels (used only when ``display`` is True).
            h: Window height in pixels (used only when ``display`` is True).
            data_parser: DataParser supplying humanoid images and metadata.
            scorekeeper: ScoreKeeper tracking actions, score, and reward.
            classifier: HumanoidClassifier used to build the RL observation state.
            img_data_root: Root path for image files.
            display: If True, render a Tkinter overlay during training.
        """
        super().__init__(root, w, h, data_parser, scorekeeper, img_data_root, display)
        self.classifier = classifier
        self.humanoid_probs = np.zeros(
            4, dtype=np.float32
        )  # raw 4-float probability output
        self.humanoid = (
            None  # set by reset() → get_humanoid(); guards step() before reset
        )
        self._episode_done = (
            False  # set True when step() returns done; guards double-step
        )
        self.reset()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and load the first humanoid.

        Calls ``super().reset()`` to zero ``obs`` and reset ``scorekeeper`` and
        ``data_parser``, then draws the first humanoid and populates the
        observation vector.

        Follows the Gymnasium 0.26+ API: returns ``(obs, info)`` and
        accepts keyword-only ``seed`` and ``options`` arguments.

        Returns:
            Tuple of ``(obs_copy, {})`` — populated initial observation vector
            after classifying the first humanoid, and an empty info dict.
        """
        super().reset(
            seed=seed, options=options
        )  # zeroes self.obs, resets scorekeeper + data_parser
        self._episode_done = False
        self.get_humanoid()
        self._build_obs()  # populate humanoid_class + confidence slots
        return self.obs.copy(), {}

    def get_humanoid(self) -> None:
        """Draw a random humanoid and classify it.

        Stores the raw 4-float probability output in ``humanoid_probs``.
        The argmax of ``humanoid_probs`` is the predicted class index
        (matches :class:`~gameplay.enums.State` enum order).
        """
        self.humanoid = self.data_parser.get_random()
        self.humanoid_probs = self.classifier.classify(self.humanoid)

    def _build_obs(self) -> np.ndarray:
        """Update ``obs`` with current game state and humanoid classification.

        Delegates to :func:`~game_interfaces.obs_builder.build_obs`, which is
        the canonical single-source-of-truth for the 15-slot observation vector.
        Updates ``self.obs`` in-place so callers holding a reference see the
        new values immediately.

        Returns:
            Reference to ``self.obs``.
        """
        self.obs[:] = build_obs(self.scorekeeper, self.humanoid, self.humanoid_probs)
        return self.obs

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute an action and return the Gymnasium step tuple.

        ``class_counts`` in the returned obs reflects the post-action ambulance
        state because ``_build_obs()`` reads from
        ``scorekeeper.ambulance`` directly — no manual tracking needed.

        Args:
            action_idx: Index into the action space
                (0=save, 1=squish, 2=skip, 3=scram).

        Returns:
            Tuple of ``(obs_copy, reward, episode_done, False, {})``.
        """
        if self.humanoid is None:
            raise RuntimeError(
                "TrainEnv.step() called before reset(). Call env.reset() first."
            )
        if self._episode_done:
            raise RuntimeError(
                "TrainEnv.step() called after episode ended. "
                "Call env.reset() before starting the next episode."
            )

        reward = 0.0
        episode_done = False

        action = list(ActionState)[action_idx]
        predicted_class = int(np.argmax(self.humanoid_probs))
        max_prob = float(np.max(self.humanoid_probs))
        confidence = (
            "high"
            if max_prob >= CNN_CONFIDENCE_HIGH
            else "mid" if max_prob >= CNN_CONFIDENCE_MID else "low"
        )
        pre_action_obs = self.obs.tolist()
        action_executed = self.scorekeeper.map_do_action(action, self.humanoid)

        if action_executed:
            self.scorekeeper.log_turn(
                self.humanoid,
                action.value,
                predicted_class=predicted_class,
                recommended_action=None,
                confidence=confidence,
                obs=pre_action_obs,
            )
            current_cum_reward = self.scorekeeper.get_cumulative_reward()
            reward = current_cum_reward - self.previous_cum_reward
            self.previous_cum_reward = current_cum_reward
        else:
            reward += BLOCKED_ACTION_PENALTY  # nudge to stop agents for doing nothing, forces action

        if self.scorekeeper.remaining_time <= 0 or len(self.data_parser.unvisited) == 0:
            episode_done = True
            # Do NOT reset here — follow the standard gym API: the caller is
            # responsible for calling env.reset() after receiving done=True.
            # Auto-resetting inside step() causes a double-reset when the
            # training loop calls reset() at the top of the next episode.
        else:
            self.get_humanoid()

        self._episode_done = episode_done
        self._build_obs()
        # truncated=False: this env only terminates (time expired or humanoids exhausted).
        # Truncation (hitting a step limit without a natural terminal state) is not used here.
        return self.obs.copy(), reward, episode_done, False, {}
