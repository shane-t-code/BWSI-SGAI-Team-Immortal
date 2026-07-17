"""Adversarial image-disguise environment for RL training (two-channel version).

The Red Team agent applies small, budget-limited edits to a zombie ("overgrown")
image on two ORTHOGONAL channels, competing for one shared malice budget:
  * RECOLOR the green skin toward human skin tone -> fools a PERSON (the CNN, which
    ignores color, still catches it). Validated: human accuracy 100%->~50%->0% across
    recolor strength, while CNN P(healthy) stays ~0.
  * ROTATE the whole scene a few degrees -> fools the CNN (a small tilt pushes it off
    the images it was trained on) while a person barely notices.
Recolor changes COLOR and rotation changes GEOMETRY, so they don't interfere — the
agent faces a real allocation problem: spend budget fooling the human, the AI, or both.

Design ideas that hold this together:
  * Accumulator pattern: keep the ORIGINAL image + running edit totals and re-render
    from the original each step. We never stack an edit on an edit.
  * Two-layer render: hue_proxy (human channel) is measured on the recolor-only "color
    layer" where the frozen green mask stays aligned; the CNN sees the full rotated
    image. Rotation moves pixels but not colors, so this split is exact.
  * Lookahead observation: each step the agent also sees what P(healthy) WOULD be if it
    rotated +2 or -2 from here, so it can learn rotation DIRECTION (which is
    image-specific) instead of blindly guessing.
  * Constraints by masking, not penalty: illegal moves (over budget / past a cap) are
    hidden via action_mask(), never punished.
  * Delta reward: each step's reward is the improvement in a weighted disguise score, so
    episode reward telescopes to the total improvement.
"""

import os
import random
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from PIL import Image

from models.CNNClassifier import CNNClassifier
from models.image_transforms import TRANSFORM_REGISTRY, green_mask, hue_proxy


class AdversarialEnv(Env):
    """RL environment for learning budget-limited two-channel disguises."""

    # ----- action layout: the 4 discrete moves, by index --------------------
    ROTATE_CW = 0    # rotate the scene +2 degrees  (fools the CNN)
    ROTATE_CCW = 1   # rotate the scene -2 degrees  (fools the CNN)
    RECOLOR = 2      # recolor +0.25 toward skin    (fools the human)
    STOP = 3         # submit the image and end the episode
    NUM_ACTIONS = 4

    # ----- edit step sizes and hard caps ------------------------------------
    ANGLE_STEP = 2.0       # degrees added per rotate action
    MAX_ANGLE = 12.0       # |cumulative angle| cap: past ~20 deg the CNN RECOVERS.
    RECOLOR_STEP = 0.25    # recolor intensity added per recolor action (4 steps -> full)
    MAX_RECOLOR = 1.0      # recolor intensity caps at 1.0

    # ----- reward weights: tuned so both channels contribute about equally --
    W_HUMAN = 0.55         # weight on hue_proxy  (how human the color looks)
    W_AI = 0.45            # weight on P(healthy) (how much the CNN is fooled)

    # ----- budget -----------------------------------------------------------
    MAX_BUDGET = 6         # fixed training budget; also normalizes the obs slot.

    OBS_SIZE = 9  # [angle, recolor, budget, Pz, Ph, Pi, Pc, look_cw, look_ccw]

    def __init__(
        self,
        classifier: CNNClassifier,
        img_data_root: str = "data",
        max_budget: int = MAX_BUDGET,
        seed: int | None = None,
    ) -> None:
        """Set up the environment and the pool of zombie images.

        Args:
            classifier: CNNClassifier used to score P(healthy) in memory.
            img_data_root: Root folder the image filepaths are relative to.
            max_budget: Malice units the agent starts each episode with.
            seed: Optional seed for reproducible image sampling.
        """
        self.classifier = classifier
        self.img_data_root = img_data_root
        self.max_budget = max_budget
        self.rng = random.Random(seed)

        # Zombie-only pool: humans already look human, so there's no reward gradient.
        meta = pd.read_csv(os.path.join(img_data_root, "consolidated_metadata.csv"))
        self.zombie_paths: list[str] = meta[meta["Class"] == "Zombie"]["Filename"].tolist()
        if not self.zombie_paths:
            raise ValueError("No Zombie-class images found in the metadata CSV.")

        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32
        )

        # Episode state — filled by reset().
        self.original_image: Image.Image | None = None
        self.green_mask: np.ndarray | None = None
        self.current_image: Image.Image | None = None
        self.cum_angle = 0.0
        self.cum_recolor = 0.0
        self.budget_left = 0
        self.prev_score = 0.0
        self.last_probs = np.zeros(4, dtype=np.float32)
        self.last_hue = 0.0
        self.last_look_cw = 0.0   # P(healthy) if we rotate +2 from here
        self.last_look_ccw = 0.0  # P(healthy) if we rotate -2 from here

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a fresh episode on a new random zombie image."""
        if seed is not None:
            self.rng.seed(seed)

        path = self.rng.choice(self.zombie_paths)
        with Image.open(os.path.join(self.img_data_root, path)) as im:
            self.original_image = im.convert("RGB").copy()

        # Freeze the green mask on the ORIGINAL so hue_proxy measures the same pixels.
        self.green_mask = green_mask(self.original_image)

        self.cum_angle = 0.0
        self.cum_recolor = 0.0
        self.budget_left = self.max_budget

        self.prev_score = self._score_and_observe()
        return self._get_observation(), {"path": path}

    # ------------------------------------------------------------------
    # Rendering (accumulator -> image)
    # ------------------------------------------------------------------

    def _render_color(self) -> Image.Image:
        """The recolor-only 'color layer' (NO rotation), built from the original.

        hue_proxy is measured here: recolor doesn't move pixels, so the frozen green
        mask still lines up. Starts from a COPY so we never alias the pristine original.
        """
        color = self.original_image.copy()
        if self.cum_recolor > 0.0:
            color = TRANSFORM_REGISTRY["recolor"].apply(color, self.cum_recolor)
        return color

    def _rotate(self, color_img: Image.Image, angle_deg: float) -> Image.Image:
        """Rotate a color-layer image by angle_deg (0 = unchanged).

        Converts degrees into the library's intensity convention:
        RotationTransform uses angle = (intensity - 0.5) * 30.
        """
        if angle_deg == 0.0:
            return color_img
        return TRANSFORM_REGISTRY["rotation"].apply(color_img, angle_deg / 30.0 + 0.5)

    def _clamp_angle(self, angle: float) -> float:
        return max(-self.MAX_ANGLE, min(self.MAX_ANGLE, angle))

    def _score_and_observe(self) -> float:
        """Render, score the disguise, and cache everything the observation needs.

        Caches last_probs, last_hue, the two rotation lookaheads, and current_image.
        Returns the weighted disguise score. Uses 3 CNN calls: the current full image
        plus the +2/-2 rotation lookaheads (all share one color-layer render).
        """
        color_img = self._render_color()                       # recolor only
        full_img = self._rotate(color_img, self.cum_angle)     # + rotation = game image
        self.current_image = full_img

        probs = self.classifier.classify_pil_image(full_img).astype(np.float32)
        self.last_probs = probs
        self.last_hue = hue_proxy(color_img, self.green_mask)   # human channel

        # Lookahead: what the CNN would say if we rotated one more step each way.
        # Gives the agent a "which way helps" sense so it can learn rotation direction.
        cw = self._rotate(color_img, self._clamp_angle(self.cum_angle + self.ANGLE_STEP))
        ccw = self._rotate(color_img, self._clamp_angle(self.cum_angle - self.ANGLE_STEP))
        self.last_look_cw = float(self.classifier.classify_pil_image(cw)[1])
        self.last_look_ccw = float(self.classifier.classify_pil_image(ccw)[1])

        return self.W_HUMAN * self.last_hue + self.W_AI * float(probs[1])

    def _get_observation(self) -> np.ndarray:
        """Assemble the 9-number observation, every slot normalized to [0, 1]."""
        return np.array(
            [
                (self.cum_angle + self.MAX_ANGLE) / (2.0 * self.MAX_ANGLE),  # -12..12 -> 0..1
                self.cum_recolor,                     # 0..1
                self.budget_left / self.MAX_BUDGET,   # 0..1
                self.last_probs[0],                   # P(zombie)
                self.last_probs[1],                   # P(healthy)
                self.last_probs[2],                   # P(injured)
                self.last_probs[3],                   # P(corpse)
                self.last_look_cw,                    # P(healthy) if rotate +2
                self.last_look_ccw,                   # P(healthy) if rotate -2
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Legal-move mask
    # ------------------------------------------------------------------

    def action_mask(self) -> list[bool]:
        """Return a length-4 list: True where that action is currently legal."""
        has_budget = self.budget_left > 0
        return [
            has_budget and self.cum_angle < self.MAX_ANGLE,     # rotate CW
            has_budget and self.cum_angle > -self.MAX_ANGLE,    # rotate CCW
            has_budget and self.cum_recolor < self.MAX_RECOLOR, # recolor
            True,                                               # stop
        ]

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one edit (or stop); return (obs, reward, terminated, truncated, info).

        Reward is the DELTA in disguise score, so episode rewards telescope to the
        total improvement produced.
        """
        info: dict[str, Any] = {}

        if action == self.STOP:
            info["stopped"] = True
            info["p_healthy"] = float(self.last_probs[1])
            info["hue"] = float(self.last_hue)
            return self._get_observation(), 0.0, True, False, info

        if action == self.ROTATE_CW:
            self.cum_angle = min(self.cum_angle + self.ANGLE_STEP, self.MAX_ANGLE)
        elif action == self.ROTATE_CCW:
            self.cum_angle = max(self.cum_angle - self.ANGLE_STEP, -self.MAX_ANGLE)
        elif action == self.RECOLOR:
            self.cum_recolor = min(self.cum_recolor + self.RECOLOR_STEP, self.MAX_RECOLOR)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.budget_left -= 1

        new_score = self._score_and_observe()
        reward = new_score - self.prev_score
        self.prev_score = new_score

        truncated = self.budget_left <= 0
        info["score"] = new_score
        info["probs"] = self.last_probs
        info["p_healthy"] = float(self.last_probs[1])
        info["hue"] = float(self.last_hue)
        return self._get_observation(), reward, False, truncated, info

    def render(self) -> Image.Image | None:
        """Return the latest full game image (or None before the first render)."""
        return self.current_image
