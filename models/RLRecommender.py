"""RLRecommender — ActionRecommender backed by the trained DQN policy.

Wraps the observation -> DQN pipeline in a single object. The caller is
responsible for computing ``probs`` (via a HumanoidClassifier) and passing
them to :meth:`recommend`. Does NOT subclass TrainEnv — delegates observation
building to :func:`~game_interfaces.obs_builder.build_obs`, the canonical
single source of truth for the 15-slot observation vector.
"""

import os
import numpy as np

from gameplay.constants import OBS_SIZE, RL_MODEL_PATH
from gameplay.enums import ActionState
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper
from game_interfaces.obs_builder import build_obs
from models.DQNAgent import DQNAgent
from models.ActionRecommender import ActionRecommender


class RLRecommender(ActionRecommender):
    """Recommends actions using the trained DQN policy.

    Builds the 15-float observation vector from the passed-in classifier
    probs + scorekeeper state, then queries
    :meth:`~models.DQNAgent.DQNAgent.select_action` in greedy mode.
    Raises ``FileNotFoundError`` at construction if the RL weights file is
    missing — fail fast rather than silently degrading.

    The caller is responsible for computing ``probs`` via a
    :class:`~models.HumanoidClassifier.HumanoidClassifier` before calling
    :meth:`recommend`.

    Attributes:
        agent: DQN policy used to select actions.
    """

    agent: DQNAgent

    def __init__(
        self,
        rl_model_file: str = RL_MODEL_PATH,
    ) -> None:
        """Initialize the RL recommender.

        Args:
            rl_model_file: Path to DQN policy weights.
        """
        if not os.path.exists(rl_model_file):
            raise FileNotFoundError(
                f"RL weights not found at '{rl_model_file}'. "
                "Train the model first ('python main.py train --type rl') or update "
                "RL_MODEL_PATH in gameplay/constants.py."
            )
        self.agent = DQNAgent(obs_size=OBS_SIZE, num_actions=4)
        self.agent.load(rl_model_file)

    def recommend(
        self,
        humanoid: Humanoid,
        probs: np.ndarray,
        scorekeeper: ScoreKeeper,
    ) -> ActionState:
        """Recommend an action for the current humanoid using the DQN policy.

        Applies an action mask before argmax so the agent never recommends an
        unavailable action (e.g. save when the ambulance is at capacity).

        Args:
            humanoid: Humanoid instance (used for ``humanoid.value``).
            probs: ``np.ndarray`` of shape ``(4,)`` from the classifier, as
                returned by
                :meth:`~models.HumanoidClassifier.HumanoidClassifier.classify`.
            scorekeeper: ScoreKeeper tracking game time, capacity, and score.

        Returns:
            An :class:`~gameplay.enums.ActionState` member.
        """
        obs = self._build_obs(humanoid, probs, scorekeeper)
        available = scorekeeper.available_action_space()  # list[bool], length 4
        action_idx = self.agent.select_action(obs, greedy=True, available=available)
        return list(ActionState)[action_idx]

    @staticmethod
    def _build_obs(
        humanoid: Humanoid,
        probs: np.ndarray,
        scorekeeper: ScoreKeeper,
    ) -> np.ndarray:
        """Build the 15-float observation vector from classifier probs + scorekeeper state.

        Delegates to :func:`~game_interfaces.obs_builder.build_obs` — the
        canonical single source of truth for the observation encoding.

        Args:
            humanoid: Humanoid instance (used for ``humanoid.value``).
            probs: ``np.ndarray`` of shape ``(4,)`` from the classifier.
            scorekeeper: ScoreKeeper with current game state.

        Returns:
            Float32 array of shape (15,), all values in [0, 1].
        """
        return build_obs(scorekeeper, humanoid, probs)
