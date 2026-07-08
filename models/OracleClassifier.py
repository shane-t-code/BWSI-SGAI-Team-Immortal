"""OracleClassifier — ground-truth HumanoidClassifier for debugging and teaching.

The oracle reads humanoid.state directly from the CSV metadata instead of
running a CNN. It returns a perfect one-hot probability distribution
(1.0 at the correct class, 0.0 everywhere else).

Teaching pattern
----------------
Swap CNNClassifier -> OracleClassifier in the mode factory to isolate RL
policy quality from CNN perception quality. If the game plays well with
the oracle but poorly with the CNN, the perception model is the bottleneck.
If it plays poorly with the oracle too, the RL policy itself needs work.

Example
-------
    from models.OracleClassifier import OracleClassifier
    classifier = OracleClassifier()
    probs = classifier.classify(humanoid)            # one-hot, e.g. [1., 0., 0., 0.]
    label, confidence = probs_to_class(probs)        # (State.ZOMBIE, 1.0)
"""

import numpy as np

from gameplay.humanoid import MAP_CLASS_INT_TO_STR, MAP_CLASS_STR_TO_INT, Humanoid
from models.HumanoidClassifier import HumanoidClassifier


class OracleClassifier(HumanoidClassifier):
    """Perfect-knowledge classifier that reads humanoid.state directly.

    humanoid.state is set from the consolidated_metadata.csv ground-truth
    label at data-parse time, so this bypasses all image inference entirely.
    Use it to test game logic and RL policy quality without CNN noise.
    """

    def classify(self, humanoid: Humanoid) -> np.ndarray:
        """Return a perfect one-hot probability distribution for the humanoid.

        Args:
            humanoid: Humanoid instance with a .state string attribute
                (``'zombie'``, ``'healthy'``, ``'injured'``, or ``'corpse'``).

        Returns:
            Float32 array of shape ``(4,)`` with 1.0 at the index corresponding
            to ``humanoid.state`` and 0.0 elsewhere, in
            :data:`~gameplay.humanoid.MAP_CLASS_INT_TO_STR` order.
        """
        idx = MAP_CLASS_STR_TO_INT[humanoid.state]
        one_hot = np.zeros(len(MAP_CLASS_INT_TO_STR), dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot
