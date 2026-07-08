"""Abstract base class for humanoid classification.

Defines the contract for deciding what a humanoid IS — the classification
half of each game turn. Implement this class to plug in your own classifier.

A game mode is a named combination of a classifier choice and a recommender
choice (plus a GUI flag). Setting the classifier to ``None`` means no
classification information is shown to the player.

Implement this ABC to plug in your own classifier.

Example
-------
    class MyClassifier(HumanoidClassifier):
        def classify(self, humanoid):
            # Return a probability distribution over [zombie, healthy, injured, corpse]
            return my_model.predict_proba(humanoid.fp)  # shape: (4,)

    probs = MyClassifier().classify(humanoid)   # np.ndarray, shape (4,), sums to 1.0
    label, confidence = probs_to_class(probs)   # (State.HEALTHY, 0.92) — caller decides
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from gameplay.enums import State
from gameplay.humanoid import Humanoid

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


def probs_to_class(probs: np.ndarray) -> tuple[State, float]:
    """Convert a probability array to a (label, confidence) pair.

    Args:
        probs: 1-D float32 array of shape ``(4,)`` as returned by
            :meth:`HumanoidClassifier.classify`. Class order:
            ``[zombie, healthy, injured, corpse]``.

    Returns:
        Tuple of ``(label, confidence)`` where:

        - ``label`` is the :class:`~gameplay.enums.State` with the highest
          probability. Use ``label.value`` for the string name.
        - ``confidence`` is the probability at that index — a float in
          ``[0, 1]`` (1.0 = certain).
    """
    if probs.ndim != 1 or len(probs) != 4:
        raise ValueError(
            f"probs_to_class expects a 1-D array of length 4, got shape {probs.shape}. "
            "Check that your classifier returns a (4,) probability vector."
        )
    idx = int(np.argmax(probs))
    # COUPLING: list(State) order must match the class order of the probs vector
    # ([zombie, healthy, injured, corpse] per MAP_CLASS_INT_TO_STR / State enum order).
    # If the State enum is reordered, this label lookup silently returns the wrong class.
    return list(State)[idx], float(probs[idx])


class HumanoidClassifier(ABC):
    """Decides what a humanoid IS.

    Implementations: CNNClassifier (uses a trained CNN), OracleClassifier
    (reads ground-truth state directly — for debugging and teaching).

    The UI displays the returned label and confidence when a classifier
    is injected; no classifier means no classification display.

    Subclasses must implement :meth:`classify`. Use the module-level
    :func:`probs_to_class` helper to convert the returned probability array
    into a ``(State, float)`` pair.
    """

    @abstractmethod
    def classify(self, humanoid: Humanoid) -> np.ndarray:
        """Return a probability distribution over all humanoid classes.

        Args:
            humanoid: Humanoid instance with a ``.fp`` filepath attribute.

        Returns:
            1-D float32 array of shape ``(4,)`` with probabilities summing
            to 1.0 in class order: ``[zombie, healthy, injured, corpse]``
            (matching :data:`~gameplay.humanoid.MAP_CLASS_INT_TO_STR`).

        Example::

            probs = classifier.classify(humanoid)
            label, confidence = probs_to_class(probs)
        """
