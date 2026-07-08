"""Abstract base class for action recommendation.

Defines the contract for deciding what action to TAKE for a humanoid — the
recommendation half of each game turn. Implement this class to plug in your
own policy or agent.

A game mode is a named combination of a classifier choice and a recommender
choice (plus a GUI flag). Setting the recommender to ``None`` means no
action recommendation is shown to the player (or auto-executed in machine
mode).

The caller is responsible for computing ``probs`` (a ``np.ndarray`` of
shape ``(4,)`` over :class:`~gameplay.enums.State`) before calling ``recommend()``.
This keeps the recommender stateless with respect to classification — any
:class:`~models.HumanoidClassifier.HumanoidClassifier` can supply the probs.

Implement this ABC to plug in your own recommender.

Example
-------
    class MyRecommender(ActionRecommender):
        def recommend(self, humanoid, probs, scorekeeper):
            action_idx = my_policy.act(probs, scorekeeper)
            return list(ActionState)[action_idx]
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from gameplay.enums import ActionState
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper

# ---------------------------------------------------------------------------
# What is an Abstract Base Class (ABC)?
#
# An ABC defines a *contract* — a set of methods that every subclass must
# implement. If a subclass forgets to implement an abstract method, Python
# raises a TypeError at instantiation time rather than silently producing a
# broken object.
#
# Reference: https://docs.python.org/3/library/abc.html
# ---------------------------------------------------------------------------


class ActionRecommender(ABC):
    """Decides what action to TAKE for a given humanoid.

    Implementations:
    RLRecommender (uses a trained DQN policy),
    HeuristicRecommender (fixed classifier-based rules).

    The UI displays the returned action string when a recommender is
    injected. In machine mode the recommended action is also auto-executed.

    The caller computes ``probs`` via a
    :class:`~models.HumanoidClassifier.HumanoidClassifier` and passes them
    in — recommenders do not own a classifier.
    """

    @abstractmethod
    def recommend(
        self,
        humanoid: Humanoid,
        probs: np.ndarray,
        scorekeeper: ScoreKeeper,
    ) -> ActionState:
        """Recommend an action for the current humanoid.

        Args:
            humanoid: Humanoid instance (used for value and metadata).
            probs: ``np.ndarray`` of shape ``(4,)`` — probability distribution
                over :class:`~gameplay.enums.State` classes, as returned by
                :meth:`~models.HumanoidClassifier.HumanoidClassifier.classify`.
                Order matches :data:`~gameplay.humanoid.MAP_CLASS_INT_TO_STR`.
            scorekeeper: ScoreKeeper tracking time, capacity, and score.

        Returns:
            An :class:`~gameplay.enums.ActionState` member:
            ``ActionState.SAVE``, ``ActionState.SQUISH``,
            ``ActionState.SKIP``, or ``ActionState.SCRAM``.
            Use ``.value`` to get the corresponding string
            (``'save'``, ``'squish'``, ``'skip'``, ``'scram'``).
        """
