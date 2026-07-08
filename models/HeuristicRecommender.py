"""HeuristicRecommender — rule-based ActionRecommender using classifier predictions.

Decision rules:
  - If ambulance is at capacity → scram
  - If predicted zombie or corpse    → squish
  - If predicted injured or healthy  → save

This is intentionally simple — override recommend() to implement
your own decision rules, reward shaping, or threshold tuning.
"""

import numpy as np

from gameplay.enums import ActionState, State
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper
from models.ActionRecommender import ActionRecommender


class HeuristicRecommender(ActionRecommender):
    """Rule-based recommender: classifier probs drive fixed action rules.

    The caller passes ``probs`` (from a HumanoidClassifier) to
    :meth:`recommend` — this recommender does not own a classifier.
    """

    def recommend(
        self,
        humanoid: Humanoid,
        probs: np.ndarray,
        scorekeeper: ScoreKeeper,
    ) -> ActionState:
        """Recommend an action using fixed classifier-to-action rules.

        Args:
            humanoid: Humanoid instance (unused by this recommender, but
                part of the :class:`~models.ActionRecommender.ActionRecommender`
                contract).
            probs: 4-float probability vector from the classifier, as
                returned by
                :meth:`~models.HumanoidClassifier.HumanoidClassifier.classify`.
            scorekeeper: ScoreKeeper tracking game time, capacity, and score.

        Returns:
            An :class:`~gameplay.enums.ActionState` member.
        """
        if scorekeeper.remaining_time <= 0 or scorekeeper.at_capacity():
            return ActionState.SCRAM

        predicted_idx = int(np.argmax(probs))
        predicted_state = list(State)[predicted_idx]

        if predicted_state in (State.ZOMBIE, State.CORPSE):
            return ActionState.SQUISH
        return ActionState.SAVE
