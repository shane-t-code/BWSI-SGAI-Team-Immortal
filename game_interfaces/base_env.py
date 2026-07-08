import math
from abc import abstractmethod
from typing import Any
import numpy as np
import tkinter as tk
from gymnasium import Env, spaces

from game_interfaces.data_parser import DataParser
from gameplay.constants import OBS_SIZE
from gameplay.scorekeeper import ScoreKeeper


class GameEnv(Env):
    """Shared base class for TrainEnv (RL training gym) and custom extensions.

    Observation space — 15 floats, all in [0, 1]:

        Index  Name                  Encoding
        0      time_bin              5 bins (0=critical … 4=fresh), normalized ÷ 4
        1-4    class_counts          count of each class [z, h, i, c] in ambulance,
                                     normalized ÷ capacity
        5-8    humanoid_class        one-hot over 4 classes (set by subclass)
        9      humanoid_confidence   CNN confidence bin (0=low, 1=med, 2=high), normalized ÷ 2
        10-13  doable_actions        binary availability of each action (as-is)
        14     humanoid_value_bin    5-bin value score (0=very low … 4=very high), normalized ÷ 4

    obs_idx slices:
        self.obs_idx["time_bin"]            → slice(0, 1)
        self.obs_idx["class_counts"]        → slice(1, 5)
        self.obs_idx["humanoid_class"]      → slice(5, 9)
        self.obs_idx["humanoid_confidence"] → slice(9, 10)
        self.obs_idx["doable_actions"]      → slice(10, 14)
        self.obs_idx["humanoid_value_bin"]  → slice(14, 15)

    To extend the game environment, subclass TrainEnv,
    not this class directly.

    Attributes:
        img_data_root: Root path for image files.
        data_parser: DataParser supplying humanoids each episode.
        scorekeeper: ScoreKeeper tracking game state and score.
        display: Whether to render a Tkinter overlay during training.
        text: Current display text (used when display is True).
        previous_cum_reward: Cumulative reward at the end of the last step,
            used to compute step-level reward deltas.
        obs_idx: Named slices into ``obs`` for each feature group.
        obs_size: Length of the flat observation vector (15).
        obs: Current flat float32 observation vector.
        observation_space: Gymnasium Box space matching ``obs``.
        action_space: Gymnasium Discrete space over the four actions.
    """

    img_data_root: str
    data_parser: DataParser
    scorekeeper: ScoreKeeper
    display: bool
    text: str
    previous_cum_reward: float
    obs_idx: dict[str, slice]
    obs_size: int
    obs: np.ndarray
    observation_space: spaces.Box
    action_space: spaces.Discrete
    # Display-mode attrs — only set when display=True
    canvas: tk.Canvas | None
    label: tk.Label | None
    suggestion: tk.Label | None

    def __init__(
        self,
        root: Any,
        w: int,
        h: int,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
        img_data_root: str = "data",
        display: bool = False,
    ) -> None:
        """Initialize the base game environment.

        Args:
            root: Tkinter root window, or ``None`` when ``display`` is False.
            w: Window width in pixels (used only when ``display`` is True).
            h: Window height in pixels (used only when ``display`` is True).
            data_parser: DataParser supplying humanoids for this episode.
            scorekeeper: ScoreKeeper tracking game state and scoring.
            img_data_root: Root path for image files.
            display: If True, render a Tkinter overlay during training.
        """
        self.img_data_root = img_data_root
        self.data_parser = data_parser
        self.scorekeeper = scorekeeper
        self.display = display
        self.text = ""
        self.previous_cum_reward = 0.0

        # Named slices into self.obs — use these instead of raw indices everywhere.
        # COUPLING: these slices must stay consistent with OBS_SIZE (constants.py) and
        # the slot layout documented in the class docstring. A slot change requires
        # updating this dict, OBS_SIZE, and game_interfaces/obs_builder.py.
        self.obs_idx = {
            "time_bin": slice(0, 1),
            "class_counts": slice(1, 5),
            "humanoid_class": slice(5, 9),
            "humanoid_confidence": slice(9, 10),
            "doable_actions": slice(10, 14),
            "humanoid_value_bin": slice(14, 15),
        }
        self.obs_size = OBS_SIZE

        # Flat float32 observation vector — all values in [0, 1]
        self.obs = self._make_empty_obs()

        # Observation space is spaces.Box (not MultiDiscrete) intentionally:
        # - Our custom DQN network expects a flat float32 vector; MultiDiscrete
        #   returns integer indices that would require manual one-hot encoding before
        #   reaching the Linear layer — same result, extra plumbing, no benefit.
        # - All features are already encoded as [0,1] floats. Individually discrete
        #   features (time_bin, confidence_bin, value_bin) use only a fixed set of
        #   float values within the Box — effectively discrete, without changing the
        #   space type or network architecture.
        # Reference: https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.scorekeeper.num_actions)

        self.canvas = None
        self.label = None
        self.suggestion = None
        if self.display:
            self._setup_display(root, w, h)

    @staticmethod
    def one_hot(value: int, num_classes: int) -> np.ndarray:
        """Return a one-hot encoded array for a categorical variable.

        One-hot encoding is used here — rather than a raw integer — because
        humanoid class is *nominal* (no natural ordering between zombie, healthy,
        injured, corpse). Feeding a raw index like 2 to a neural network implies
        that class 2 is "twice" class 1, which is meaningless and can mislead
        gradient-based learning. One-hot removes that false ordinal relationship
        by giving each class its own independent binary feature.

        Reference: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features

        Args:
            value: Integer class index (0-indexed).
            num_classes: Total number of classes.

        Returns:
            Float32 array of shape ``(num_classes,)`` with a ``1.0`` at
            position ``value`` and ``0.0`` everywhere else.
        """
        v = np.zeros(num_classes, dtype=np.float32)
        v[value] = 1.0
        return v

    def _make_empty_obs(self) -> np.ndarray:
        """Return a zeroed-out 15-float observation vector.

        Slot layout:
            [0]     time_bin            — normalized remaining-time bin
            [1-4]   class_counts        — normalized per-class ambulance occupancy
            [5-8]   humanoid_class      — one-hot class of current humanoid (set by subclass)
            [9]     humanoid_confidence — normalized CNN confidence bin (set by subclass)
            [10-13] doable_actions      — binary action availability
            [14]    humanoid_value_bin  — normalized value bin (set by subclass)

        Returns:
            Float32 zeros array of shape ``(obs_size,)`` (currently 15 slots).
        """
        return np.zeros(self.obs_size, dtype=np.float32)

    def _setup_display(self, root: Any, w: int, h: int) -> None:
        """Create the Tkinter training overlay.

        Args:
            root: Tkinter root window.
            w: Window width in pixels.
            h: Window height in pixels.
        """
        self.canvas = tk.Canvas(
            root, width=math.floor(0.2 * w), height=math.floor(0.1 * h)
        )
        self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.75 * h))
        self.label = tk.Label(self.canvas, text="Machine says...", font=("Arial", 20))
        self.label.pack(
            side=tk.TOP
        )  # pack() inside Canvas works by accident — Canvas ignores geometry managers; label renders at (0,0)
        self.suggestion = tk.Label(self.canvas, text=self.text, font=("Arial", 20))
        self.suggestion.pack(side=tk.TOP)  # same

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode.

        Clears ``obs``, resets ``data_parser`` and ``scorekeeper``.
        Subclasses should call ``super().reset()`` then do their own
        post-reset work (e.g. calling ``get_humanoid()`` to populate the
        humanoid slots).

        Follows the Gymnasium 0.26+ API: returns ``(obs, info)`` and
        accepts keyword-only ``seed`` and ``options`` arguments.

        Args:
            seed: Unused — seeding is handled at DataParser/ScoreKeeper
                construction time for reproducibility. Accepted here for
                Gymnasium API compliance.
            options: Unused. Accepted for Gymnasium API compliance.

        Returns:
            Tuple of ``(obs_copy, {})`` — zeroed observation vector and an
            empty info dict. The returned obs is all zeros; subclasses must
            call ``build_obs()`` from ``game_interfaces.obs_builder`` and
            write the result into ``self.obs`` before it is valid for the agent.
        """
        self.obs = self._make_empty_obs()
        self.previous_cum_reward = 0.0
        self.data_parser.reset()
        self.scorekeeper.reset()
        return self.obs.copy(), {}

    @abstractmethod
    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one action step. Must be implemented by subclasses.

        Args:
            action_idx: Integer index into the action space
                (0=save, 1=squish, 2=skip, 3=scram).

        Returns:
            Tuple of ``(obs, reward, terminated, truncated, info)`` following
            the Gymnasium API.

        Raises:
            NotImplementedError: Always — subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement step()")
