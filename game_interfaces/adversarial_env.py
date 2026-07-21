"""Adversarial image-disguise environment for RL training (two-channel version).

The Red Team agent applies small, budget-limited edits to a zombie ("overgrown")
image on two ORTHOGONAL channels, competing for one shared malice budget:
  * RECOLOR the green skin toward human skin tone -> fools a PERSON (the CNN, which
    ignores color, still catches it). Validated: human accuracy 100%->~50%->0% across
    recolor strength, while CNN P(healthy) stays ~0.
  * ROTATE the whole scene a few degrees -> walks the CNN toward its DECISION BOUNDARY
    (a small tilt pushes it off the images it was trained on) while a person barely
    notices. The goal is not to flip the CNN to "healthy" -- it is to park it at the edge
    of its verdict, still saying "zombie" but only ~55% sure. See the tent reward below.
Recolor changes COLOR and rotation changes GEOMETRY, so they don't interfere. The
disguise is TWO-PHASE (enforced by action_mask): fully recolor first — the human MUST be
fooled or the AI's opinion is moot — then tilt just enough to sit the CNN at its decision
boundary. What the agent actually learns is the rotation DIRECTION (image-specific) and
WHEN TO STOP (right at the boundary, not past it).

Design ideas that hold this together:
  * Accumulator pattern: keep the ORIGINAL image + running edit totals and re-render
    from the original each step. We never stack an edit on an edit.
  * Two-layer render: hue_proxy (human channel) is measured on the recolor-only "color
    layer" where the frozen green mask stays aligned; the CNN sees the full rotated
    image. Rotation moves pixels but not colors, so this split is exact.
  * Lookahead observation: each step the agent also sees what P(healthy) WOULD be if it
    rotated one step each way, so it can learn rotation DIRECTION (which is
    image-specific) AND sense when the next step would overshoot the boundary.
  * Constraints by masking, not penalty: illegal moves (over budget / past a cap) are
    hidden via action_mask(), never punished.
  * TENT reward on P(healthy), not maximize. The AI-channel score peaks when P(healthy)
    == AI_TARGET (~the CNN's decision boundary) and DECREASES on both sides. Maximizing
    P(healthy) used to drive the agent PAST the boundary, so the top tier was 88%
    "confidently healthy" -- the CNN agreeing with a fooled human, a useless cell. The
    tent parks the CNN at "zombie, but unsure" and the agent LEARNS TO STOP there (any
    further edit lowers the score). (Two alternatives — rewarding raw uncertainty, and
    aiming at P(zombie) with a wider budget — were tried and lost, the latter by letting
    the agent overshoot into dead cells. A tight budget-10 cap is the guardrail.)
  * Delta reward: each step's reward is the change in the weighted score, so episode
    reward telescopes to the total. Works for the tent too (it's still a potential).
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
    ROTATE_CW = 0    # rotate the scene +0.5 degrees (walk the CNN toward its boundary)
    ROTATE_CCW = 1   # rotate the scene -0.5 degrees (walk the CNN toward its boundary)
    RECOLOR = 2      # recolor +0.25 toward skin     (fools the human)
    STOP = 3         # submit the image and end the episode
    NUM_ACTIONS = 4

    # ----- edit step sizes and hard caps ------------------------------------
    # 0.5-deg steps: fine enough to walk up to each image's decision boundary, coarse
    # enough that a budget-10 cap stops rotation BEFORE it can overshoot the boundary.
    # (Finer 0.25-deg steps + a wider budget were tried and made it WORSE — the extra
    # rotation room let the agent sail PAST the boundary into "confidently healthy" dead
    # cells.) Cap at 6 deg: beyond it every image has long since flipped.
    ANGLE_STEP = 0.5
    MAX_ANGLE = 6.0
    RECOLOR_STEP = 0.25    # recolor intensity added per recolor action (4 steps -> full)
    MAX_RECOLOR = 1.0      # recolor intensity caps at 1.0

    # ----- reward weights: tuned so both channels contribute about equally --
    W_HUMAN = 0.55         # weight on hue_proxy  (how human the color looks -> maximize)
    W_AI = 0.45            # weight on the tent   (how close the CNN is to its boundary)

    # ----- tent target: where we want to park the CNN -----------------------
    # AI_TARGET is the P(healthy) the agent aims the CNN at. 0.45 sits just under the 0.5
    # verdict boundary, so the CNN still says "zombie" but only ~55% sure — the dilemma we
    # want. TENT_SLOPE sets how sharply overshooting is punished. Aiming at P(healthy)
    # beat both raw-uncertainty and P(zombie) targets: cleanest learning signal, fewest
    # dead cells. ~40% of figures tilt toward 'corpse' and stay confidently non-human at
    # the top tier — that is AI-correct (helps the player); converting it only added dead
    # cells, so we leave it.
    AI_TARGET = 0.45
    TENT_SLOPE = 2.0

    # ----- budget -----------------------------------------------------------
    # Malice budget is the study's independent variable: generation runs the agent at a
    # LOW / MID / HIGH budget (2 / 8 / 10) for the three tiers. 10 is the ceiling AND a
    # guardrail — ~4 edits fully recolor the figure, leaving 6 rotation edits: enough to
    # reach the boundary but NOT enough to overshoot it. Raising it to 16 let the agent
    # over-rotate into dead cells, so 10 stays. With the tent reward the agent spends only
    # what it needs and STOPS, so higher budget = "allowed more malice", not "forced".
    MAX_BUDGET = 10        # training budget; also normalizes the obs slot.

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
        self.last_look_cw = 0.0   # P(healthy) if we rotate one step CW from here
        self.last_look_ccw = 0.0  # P(healthy) if we rotate one step CCW from here

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

    def _ai_score(self, probs: np.ndarray) -> float:
        """Tent on P(healthy): peaks (1.0) at AI_TARGET, falls off both sides.

        The agent MAXIMIZES this. At P(healthy)=AI_TARGET (just under the 0.5 verdict
        line) the CNN is at the edge of calling the figure human — it still says "zombie"
        but only ~55% sure, the dilemma we want. Moving P(healthy) lower (obvious zombie)
        OR higher (fooled into "healthy") lowers the score, so overshooting the boundary
        is a REWARD LOSS and the agent stops at the edge.

        We aim at P(healthy) rather than raw uncertainty (1 - max_prob) or P(zombie):
        those alternatives were tried and lost. Raw uncertainty gave too flat a learning
        signal far from the torn point, so the agent under-rotated. P(zombie) with a wider
        budget let the agent overshoot into "confidently healthy" (dead cells). Aiming at
        the healthy boundary with a tight budget was cleanest and lowest-dead-cell.

        Args:
            probs: CNN 4-class probability vector [zombie, healthy, injured, corpse].

        Returns:
            Score in [0, 1]; 1.0 exactly at the target boundary.
        """
        return max(0.0, 1.0 - self.TENT_SLOPE * abs(float(probs[1]) - self.AI_TARGET))

    def _score_and_observe(self) -> float:
        """Render, score the disguise, and cache everything the observation needs.

        Caches last_probs, last_hue, the two rotation lookaheads, and current_image.
        Returns the weighted disguise score. Uses 3 CNN calls: the current full image
        plus the two rotation lookaheads (all share one color-layer render).
        """
        color_img = self._render_color()                       # recolor only
        full_img = self._rotate(color_img, self.cum_angle)     # + rotation = game image
        self.current_image = full_img

        probs = self.classifier.classify_pil_image(full_img).astype(np.float32)
        self.last_probs = probs
        self.last_hue = hue_proxy(color_img, self.green_mask)   # human channel

        # Lookahead: what P(healthy) would be if we rotated one more step each way.
        # Gives the agent a "which way helps" sense so it can learn rotation direction.
        cw = self._rotate(color_img, self._clamp_angle(self.cum_angle + self.ANGLE_STEP))
        ccw = self._rotate(color_img, self._clamp_angle(self.cum_angle - self.ANGLE_STEP))
        self.last_look_cw = float(self.classifier.classify_pil_image(cw)[1])
        self.last_look_ccw = float(self.classifier.classify_pil_image(ccw)[1])

        # Both channels MAXIMIZE: recolor toward skin (fool the human) and CNN
        # uncertainty (make the AI unsure). Both in [0,1], weighted to a [0,1] score.
        return self.W_HUMAN * self.last_hue + self.W_AI * self._ai_score(probs)

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
                self.last_look_cw,                    # P(healthy) if we rotate one step CW
                self.last_look_ccw,                   # P(healthy) if we rotate one step CCW
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Legal-move mask
    # ------------------------------------------------------------------

    def action_mask(self) -> list[bool]:
        """Return a length-4 list: True where that action is currently legal.

        TWO-PHASE disguise: rotation is illegal until recolor is FULL. The disguise is
        "make it look human first (always fully), then tilt just enough to sit the CNN at
        its boundary". This is a masking constraint, not a reward penalty — the agent
        never even sees rotation as an option while green remains.

        Why: with free allocation the agent under-recolored ~1/3 of figures (spent budget
        tilting toward the CNN's boundary while leaving hands/feet green), so the HUMAN
        could still catch them and the AI's opinion never mattered. Forcing full recolor
        guarantees the human is fooled, which is the whole premise of the top tier. It
        also cuts dead cells: for images that flip in one rotation step, the agent simply
        declines to rotate (stays confident-zombie) instead of overshooting into
        confidently-healthy.
        """
        has_budget = self.budget_left > 0
        recolor_done = self.cum_recolor >= self.MAX_RECOLOR
        return [
            has_budget and recolor_done and self.cum_angle < self.MAX_ANGLE,   # rotate CW
            has_budget and recolor_done and self.cum_angle > -self.MAX_ANGLE,  # rotate CCW
            has_budget and self.cum_recolor < self.MAX_RECOLOR,                # recolor
            True,                                                              # stop
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
            # ai_score = CNN uncertainty (higher = less sure = better). Logged so
            # teammates read "how unsure the AI is" instead of misreading raw P(healthy).
            info["ai_score"] = self._ai_score(self.last_probs)
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
        # ai_score = torn-ness (1.0 = CNN maximally torn about "zombie"). Logged so
        # teammates read "how unsure the AI is" instead of misreading raw P(healthy).
        info["ai_score"] = self._ai_score(self.last_probs)
        return self._get_observation(), reward, False, truncated, info

    def render(self) -> Image.Image | None:
        """Return the latest full game image (or None before the first render)."""
        return self.current_image
