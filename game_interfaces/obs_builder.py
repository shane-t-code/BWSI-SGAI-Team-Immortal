"""obs_builder — canonical free function for building the 15-slot RL observation vector.

Extracted from RLRecommender._build_obs() (the standalone reference implementation).
Use this instead of duplicating obs-encoding logic across TrainEnv and RLRecommender.

Slot layout (all values in [0, 1]):
    [0]     time_bin            normalized remaining-time bin ÷ 4
    [1-4]   class_counts        per-class ambulance counts ÷ capacity
    [5-8]   humanoid_class      one-hot over 4 classes (classifier argmax)
    [9]     humanoid_confidence confidence bin (0|1|2) ÷ 2
    [10-13] doable_actions      binary action availability
    [14]    humanoid_value_bin  value bin (0=very low … 4=very high) ÷ 4
"""

import numpy as np

from gameplay.constants import CNN_CONFIDENCE_HIGH, CNN_CONFIDENCE_MID, OBS_SIZE
from gameplay.humanoid import MAP_CLASS_INT_TO_STR, Humanoid
from gameplay.scorekeeper import ScoreKeeper


def build_obs(
    scorekeeper: ScoreKeeper,
    humanoid: Humanoid | None = None,
    probs: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 15-float observation vector from classifier probs and scorekeeper state.

    Encodes the current game state into a flat float32 vector matching the
    layout defined in ``GameEnv.obs_idx``. All values are in ``[0, 1]``.

    Args:
        scorekeeper: ScoreKeeper with current game state (time, ambulance, actions).
        humanoid: Humanoid instance for the current turn (used for ``humanoid.value``).
        probs: ``np.ndarray`` of shape ``(4,)`` from the classifier, as returned by
            :meth:`~models.HumanoidClassifier.HumanoidClassifier.classify`.

    Returns:
        Float32 array of shape ``(OBS_SIZE,)`` (currently 15), all values in ``[0, 1]``.

    Raises:
        ValueError: If both ``humanoid`` and ``probs`` are ``None``.
        ValueError: If ``probs`` is provided but ``humanoid`` is ``None``.
        ValueError: If ``humanoid`` is provided but ``probs`` is ``None``.
    """
    if humanoid is None and probs is None:
        raise ValueError(
            "build_obs() called with neither humanoid nor probs. "
            "Ensure get_humanoid() and classify() run before build_obs()."
        )
    if humanoid is None:
        raise ValueError("build_obs() called with probs but no humanoid.")
    if probs is None:
        raise ValueError(
            "build_obs() called with humanoid but no probs — classify() must run before build_obs()."
        )

    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # Slot 0: time_bin — divide shift into 5 equal windows, normalize ÷ 4.
    # Coarseness note: each bin spans shift_length/5 minutes. At the default
    # 720-min shift that is 144 minutes per bin — the agent cannot distinguish
    # within a bin (e.g. "just started" vs "top 20% remaining" look identical).
    # Clamp to 0 if remaining_time went negative (e.g. scram overshoot).
    clamped_time = max(0, scorekeeper.remaining_time)
    fraction_remaining = (
        clamped_time / scorekeeper.shift_length if scorekeeper.shift_length > 0 else 0.0
    )
    time_bin = min(4, int(fraction_remaining * 5))
    obs[0] = time_bin / 4.0

    # Slots 1-4: class_counts — normalized per-class ambulance occupancy.
    # Order follows MAP_CLASS_INT_TO_STR (derived from State enum): zombie, healthy, injured, corpse.
    # COUPLING: reordering the State enum silently breaks obs alignment — both this function
    # and the one-hot in slots 5-8 iterate MAP_CLASS_INT_TO_STR, which mirrors enum order.
    ambulance = scorekeeper.ambulance
    capacity = scorekeeper.capacity
    for i, cls in enumerate(MAP_CLASS_INT_TO_STR):
        obs[1 + i] = ambulance.get(cls, 0) / capacity

    # Slots 5-8: humanoid_class — one-hot of classifier argmax prediction.
    predicted_class = int(np.argmax(probs))
    obs[5 + predicted_class] = 1.0

    # Slot 9: humanoid_confidence — bin by top probability, normalize to [0, 1].
    max_prob = float(np.max(probs))
    if max_prob >= CNN_CONFIDENCE_HIGH:
        confidence_bin = 2  # high
    elif max_prob >= CNN_CONFIDENCE_MID:
        confidence_bin = 1  # medium
    else:
        confidence_bin = 0  # low
    obs[9] = confidence_bin / 2.0

    # Slots 10-13: doable_actions — binary availability of each action.
    obs[10:14] = np.array(scorekeeper.available_action_space(), dtype=np.float32)

    # Slot [14]: humanoid_value_bin — discretize value into 5 equal bins and normalize.
    # bin = min(int(value * 5), 4) maps [0.0, 1.0) → {0, 1, 2, 3, 4} (guards against value==1.0).
    # Dividing by 4.0 maps {0,1,2,3,4} → {0.0, 0.25, 0.5, 0.75, 1.0} — identical pattern to time_bin.
    value_bin = min(int(humanoid.value * 5), 4)
    obs[14] = value_bin / 4.0

    return obs
