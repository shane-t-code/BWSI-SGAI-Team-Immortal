"""Adversarial image-disguise environment for RL training.

The Red Team agent applies small, budget-limited edits to a zombie ("overgrown")
image, trying to make it look human on two fronts at once:
  * to a PERSON  -> fog fades the bright-green skin (the main human-visible tell)
  * to the CNN   -> a small rotation tips the classifier toward "healthy"
Both edits spend from one shared malice budget, so the agent faces a real
allocation problem: how much budget goes to fooling the human vs. the CNN.

Three design ideas hold this together (see the methods for detail):
  * Accumulator pattern: we keep the ORIGINAL image plus running edit totals and
    re-render from the original every step. We never stack an edit on an edit.
  * Delta reward: each step's reward is the *improvement* in a weighted disguise
    score, so the whole episode's reward telescopes to the total improvement.
  * Constraints by masking: illegal moves (over budget, past a cap) are hidden
    from the agent via action_mask(), never punished with a penalty.
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
    """RL environment for learning budget-limited image disguises."""

    # ----- action layout: the 5 discrete moves, by index --------------------
    ROTATE_CW = 0    # rotate the scene +2 degrees  (fools the CNN)
    ROTATE_CCW = 1   # rotate the scene -2 degrees  (fools the CNN)
    FOG = 2          # add +0.25 fog intensity      (fools the human)
    BLUR = 3         # add +0.25 blur intensity      (weak filler)
    STOP = 4         # submit the image and end the episode
    NUM_ACTIONS = 5

    # ----- edit step sizes and hard caps ------------------------------------
    ANGLE_STEP = 2.0      # degrees added per rotate action
    MAX_ANGLE = 12.0      # |cumulative angle| never exceeds this: past ~20 deg the
                          # CNN RECOVERS, so more rotation would undo our own work.
    FOG_STEP = 0.25       # fog intensity added per fog action  (4 steps -> full fog)
    BLUR_STEP = 0.25      # blur intensity added per blur action
    MAX_INTENSITY = 1.0   # fog and blur intensities cap at 1.0

    # ----- reward weights: tuned so both channels contribute about equally --
    W_HUMAN = 0.55        # weight on hue_proxy  (how human the color looks)
    W_AI = 0.45           # weight on P(healthy) (how much the CNN is fooled)

    # ----- budget -----------------------------------------------------------
    MAX_BUDGET = 6        # fixed training budget; also normalizes the obs slot.
                          # Tiers (low/med/high) are made later by snapshotting the
                          # image after 1 / 3 / 6 edits of a single budget-6 rollout.

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

        # Pool of ZOMBIE-only images. Humans have nothing to disguise (they already
        # look human and score ~1.0), so training on them would give no signal.
        meta = pd.read_csv(os.path.join(img_data_root, "consolidated_metadata.csv"))
        self.zombie_paths: list[str] = meta[meta["Class"] == "Zombie"]["Filename"].tolist()
        if not self.zombie_paths:
            raise ValueError("No Zombie-class images found in the metadata CSV.")

        # 5 discrete actions; 8-number observation, every slot already in [0, 1].
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Episode state — filled in properly by reset().
        self.original_image: Image.Image | None = None  # pristine zombie, never mutated
        self.green_mask: np.ndarray | None = None        # frozen on the original
        self.current_image: Image.Image | None = None    # latest full render (for saving)
        self.cum_angle = 0.0    # running rotation total (degrees)
        self.cum_fog = 0.0      # running fog intensity total (0..1)
        self.cum_blur = 0.0     # running blur intensity total (0..1)
        self.budget_left = 0    # malice units remaining this episode
        self.prev_score = 0.0   # last step's disguise score (for the delta reward)
        self.last_probs = np.zeros(4, dtype=np.float32)  # last CNN output (for the obs)
        self.last_hue = 0.0     # last hue_proxy value (for logging the human channel)

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a fresh episode on a new random zombie image.

        Returns:
            (observation, info) where info carries the chosen image path.
        """
        if seed is not None:
            self.rng.seed(seed)

        # Load a pristine copy of a random zombie. We never edit this copy in place —
        # every render starts from it, which is the whole point of the accumulator idea.
        path = self.rng.choice(self.zombie_paths)
        with Image.open(os.path.join(self.img_data_root, path)) as im:
            self.original_image = im.convert("RGB").copy()

        # Freeze the green mask on the ORIGINAL. hue_proxy needs a fixed set of pixel
        # locations so the agent can't "hide" green by making the mask move (see hue_proxy).
        self.green_mask = green_mask(self.original_image)

        # Reset accumulators and hand out the full budget.
        self.cum_angle = 0.0
        self.cum_fog = 0.0
        self.cum_blur = 0.0
        self.budget_left = self.max_budget

        # Baseline score of the untouched zombie. The first step's delta reward
        # measures improvement against this starting point.
        self.prev_score = self._score_and_observe()

        return self._get_observation(), {"path": path}

    # ------------------------------------------------------------------
    # Rendering and scoring
    # ------------------------------------------------------------------

    def _render(self) -> tuple[Image.Image, Image.Image]:
        """Rebuild the current image from the ORIGINAL + all accumulators, applied once.

        Returns two layers:
            color_img: original + fog + blur, but NOT rotated. hue_proxy is measured
                here because fog/blur don't move pixels, so the frozen green mask still
                lines up with the character.
            full_img: color_img + rotation. This is the real game image and the CNN's
                input. Rotation moves pixels (so the frozen mask would NOT line up on
                this one) but rotation doesn't change color, so greenness measured on
                color_img is exactly right for full_img too.
        """
        # Start from a COPY of the pristine original so the returned images can never
        # alias original_image. Without this, a zero-edit render (baseline, or a STOP
        # with no edits) would hand back the original object itself — and any future
        # in-place edit to current_image would silently corrupt the pristine base that
        # every step re-renders from. The copy is negligible next to the CNN forward pass.
        color = self.original_image.copy()
        # Fog first, then blur — both recolor/average pixels in place (no movement).
        if self.cum_fog > 0.0:
            color = TRANSFORM_REGISTRY["fog"].apply(color, self.cum_fog)
        if self.cum_blur > 0.0:
            color = TRANSFORM_REGISTRY["blur"].apply(color, self.cum_blur)

        # Rotation last. Convert our angle (degrees) into the library's intensity
        # convention: RotationTransform uses angle = (intensity - 0.5) * 30.
        full = color
        if self.cum_angle != 0.0:
            intensity = self.cum_angle / 30.0 + 0.5
            full = TRANSFORM_REGISTRY["rotation"].apply(color, intensity)

        return color, full

    def _score_and_observe(self) -> float:
        """Render, score the disguise, and cache the CNN probs + current image.

        Caches self.last_probs (for the observation) and self.current_image (so the
        Friday generation script can snapshot it). Returns the weighted disguise score.
        """
        color_img, full_img = self._render()
        self.current_image = full_img

        # CNN sees the full (rotated) game image. probs = [zombie, healthy, injured, corpse].
        probs = self.classifier.classify_pil_image(full_img).astype(np.float32)
        self.last_probs = probs

        # Human channel: how non-green the character looks, measured on the color layer.
        human_look = hue_proxy(color_img, self.green_mask)
        self.last_hue = human_look  # cache for logging (and for the STOP info below)
        # AI channel: how much the CNN believes "healthy".
        fools_cnn = float(probs[1])

        return self.W_HUMAN * human_look + self.W_AI * fools_cnn

    def _get_observation(self) -> np.ndarray:
        """Assemble the 8-number observation, every slot normalized to [0, 1]."""
        return np.array(
            [
                # angle runs -12..+12; shift+scale into 0..1 (0.5 = no rotation).
                (self.cum_angle + self.MAX_ANGLE) / (2.0 * self.MAX_ANGLE),
                self.cum_fog,                              # already 0..1
                self.cum_blur,                             # already 0..1
                self.budget_left / self.MAX_BUDGET,        # 0..1
                self.last_probs[0],                        # P(zombie)
                self.last_probs[1],                        # P(healthy)
                self.last_probs[2],                        # P(injured)
                self.last_probs[3],                        # P(corpse)
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Legal-move mask
    # ------------------------------------------------------------------

    def action_mask(self) -> list[bool]:
        """Return a length-5 list: True where that action is currently legal.

        Passed to agent.select_action(obs, available=...), which sets illegal
        actions to -inf before choosing — so the agent can never spend budget it
        doesn't have or push an edit past its cap. 'stop' is always legal.
        """
        has_budget = self.budget_left > 0
        return [
            has_budget and self.cum_angle < self.MAX_ANGLE,     # rotate CW: room to go up
            has_budget and self.cum_angle > -self.MAX_ANGLE,    # rotate CCW: room to go down
            has_budget and self.cum_fog < self.MAX_INTENSITY,   # fog: not maxed
            has_budget and self.cum_blur < self.MAX_INTENSITY,  # blur: not maxed
            True,                                               # stop: always allowed
        ]

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one edit (or stop), and return (obs, reward, terminated, truncated, info).

        Reward is the DELTA in disguise score: how much better this one move made the
        image. Summed over the episode, the deltas telescope to (final - starting) score.
        """
        info: dict[str, Any] = {}

        # STOP: submit as-is. No image change, so no reward. Episode ends (terminated).
        # last_probs / last_hue still reflect the current image, so logging stays correct.
        if action == self.STOP:
            info["stopped"] = True
            info["p_healthy"] = float(self.last_probs[1])
            info["hue"] = float(self.last_hue)
            return self._get_observation(), 0.0, True, False, info

        # Otherwise apply the chosen edit to its accumulator, clamped at the cap.
        if action == self.ROTATE_CW:
            self.cum_angle = min(self.cum_angle + self.ANGLE_STEP, self.MAX_ANGLE)
        elif action == self.ROTATE_CCW:
            self.cum_angle = max(self.cum_angle - self.ANGLE_STEP, -self.MAX_ANGLE)
        elif action == self.FOG:
            self.cum_fog = min(self.cum_fog + self.FOG_STEP, self.MAX_INTENSITY)
        elif action == self.BLUR:
            self.cum_blur = min(self.cum_blur + self.BLUR_STEP, self.MAX_INTENSITY)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Every edit costs one malice unit.
        self.budget_left -= 1

        # Re-render from the original, re-score, reward = improvement over last step.
        new_score = self._score_and_observe()
        reward = new_score - self.prev_score
        self.prev_score = new_score

        # Out of budget -> episode ends. We call this "truncated" (a time/limit stop)
        # rather than "terminated" (agent chose to finish) to match Gymnasium's meaning.
        truncated = self.budget_left <= 0

        info["score"] = new_score
        info["probs"] = self.last_probs
        info["p_healthy"] = float(self.last_probs[1])
        info["hue"] = float(self.last_hue)
        return self._get_observation(), reward, False, truncated, info

    def render(self) -> Image.Image | None:
        """Return the latest full game image (or None before the first render)."""
        return self.current_image
