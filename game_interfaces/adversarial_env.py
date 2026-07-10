"""Adversarial image generation environment for RL training.

This environment allows an RL agent to learn to generate harder-to-classify
images by applying transformations that target the CNN's decision boundaries
while following the 5 rules:
1. Must remain theoretically solvable
2. Character must still be present and recognizable
3. Target specific features the CNN was trained on
4. Difficulty should be tunable
5. Stay within airport/human-plausible setting
"""

import math
from typing import Any
import numpy as np
import torch
from gymnasium import Env, spaces
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms

from gameplay.humanoid import Humanoid
from models.CNNClassifier import CNNClassifier


class AdversarialEnv(Env):
    """RL environment for generating adversarial humanoid images.

    The agent selects transformation parameters to make images harder to classify
    while keeping them solvable and realistic. The reward is based on classification
    difficulty metrics (entropy, margin, confidence).

    Observation space: Current image features + transformation parameters
    Action space: Discrete actions for different transformation types
    """

    def __init__(
        self,
        classifier: CNNClassifier,
        img_data_root: str = "data",
        max_difficulty: float = 0.8,
    ) -> None:
        """Initialize the adversarial image generation environment.

        Args:
            classifier: CNNClassifier to evaluate image difficulty.
            img_data_root: Root directory for image files.
            max_difficulty: Maximum allowed difficulty (0-1). Images exceeding
                this threshold receive penalties to maintain solvability (Rule 1).
        """
        self.classifier = classifier
        self.img_data_root = img_data_root
        self.max_difficulty = max_difficulty

        # Transformation types (Rule 4: tunable difficulty)
        self.transform_types = [
            "none",           # No transformation
            "blur",           # Gaussian blur (Rule 3: target texture features)
            "brightness",     # Brightness adjustment (Rule 3: target coloring)
            "contrast",       # Contrast adjustment (Rule 3: target visibility)
            "rotation",       # Small rotation (Rule 3: target posture cues)
            "crop",           # Slight crop (Rule 2: keep character visible)
            "noise",          # Subtle noise (Rule 3: target texture)
        ]

        # Action space: select transformation type + intensity
        # Actions 0-6: transformation type
        # Action 7: apply current transformation and move to next image
        self.action_space = spaces.Discrete(len(self.transform_types) + 1)

        # Observation space: [current_transform_idx, intensity, difficulty_metrics]
        # difficulty_metrics: [entropy, margin, max_prob, predicted_class]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),  # [transform_idx, intensity, entropy, margin, max_prob, class]
            dtype=np.float32,
        )

        # Current state
        self.current_humanoid: Humanoid | None = None
        self.current_image: Image.Image | None = None
        self.current_transform_idx = 0
        self.current_intensity = 0.0
        self.current_features: dict[str, Any] = {}
        self.original_features: dict[str, Any] = {}

        # Load a random humanoid to start
        self._load_random_humanoid()

    def _load_random_humanoid(self) -> None:
        """Load a random humanoid image from the dataset."""
        # For now, use a fixed approach - in production, use DataParser
        import os
        import random

        image_dir = os.path.join(self.img_data_root, "consolidated_dataset")
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                img_path = random.choice(images)
                self.current_humanoid = Humanoid(
                    fp=f"consolidated_dataset/{img_path}",
                    state="unknown",  # Will be determined by classifier
                    value=0.5
                )
                with Image.open(os.path.join(self.img_data_root, self.current_humanoid.fp)) as img:
                    self.current_image = img.convert("RGB").copy()
            else:
                raise ValueError(f"No images found in {image_dir}")
        else:
            raise ValueError(f"Image directory not found: {image_dir}")

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment with a new random image.

        Returns:
            Tuple of (observation, info_dict).
        """
        if seed is not None:
            np.random.seed(seed)

        self._load_random_humanoid()
        self.current_transform_idx = 0
        self.current_intensity = 0.0

        # Get original classification features
        self.original_features = self.classifier.classify_with_features(self.current_humanoid)
        self.current_features = self.original_features.copy()

        obs = self._get_observation()
        return obs, {"original_features": self.original_features}

    def _get_observation(self) -> np.ndarray:
        """Build the current observation vector.

        Returns:
            Float32 array of shape (6,) with:
                [transform_idx, intensity, entropy, margin, max_prob, predicted_class]
        """
        return np.array([
            self.current_transform_idx / len(self.transform_types),
            self.current_intensity,
            self.current_features.get("entropy", 0.0) / 2.0,  # Normalize entropy (max ~2.0 for 4 classes)
            self.current_features.get("margin", 0.0),
            self.current_features.get("max_prob", 0.0),
            self.current_features.get("predicted_class", 0) / 4.0,
        ], dtype=np.float32)

    def _apply_transformation(self, transform_idx: int, intensity: float) -> Image.Image:
        """Apply a transformation to the current image.

        Args:
            transform_idx: Index of transformation type.
            intensity: Intensity of transformation (0.0 to 1.0).

        Returns:
            Transformed PIL Image.
        """
        if self.current_image is None:
            raise RuntimeError("No image loaded. Call reset() first.")

        transform_type = self.transform_types[transform_idx]
        img = self.current_image.copy()

        # Rule 4: Tunable difficulty via intensity parameter
        # Rule 5: Keep transformations realistic and human-plausible

        if transform_type == "none":
            return img

        elif transform_type == "blur":
            # Gaussian blur - target texture features (Rule 3)
            # Intensity maps to radius (0 to 3 pixels)
            radius = intensity * 3.0
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif transform_type == "brightness":
            # Brightness adjustment - target coloring (Rule 3)
            # Intensity maps to factor (0.5 to 1.5)
            factor = 0.5 + intensity
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)

        elif transform_type == "contrast":
            # Contrast adjustment - target visibility (Rule 3)
            # Intensity maps to factor (0.5 to 1.5)
            factor = 0.5 + intensity
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(factor)

        elif transform_type == "rotation":
            # Small rotation - target posture cues (Rule 3)
            # Intensity maps to angle (-15 to +15 degrees)
            # Rule 2: Keep character recognizable (small angles only)
            angle = (intensity - 0.5) * 30.0
            return img.rotate(angle, expand=False, fillcolor=(128, 128, 128))

        elif transform_type == "crop":
            # Slight crop - Rule 2: keep character visible
            # Intensity maps to crop percentage (0% to 20%)
            crop_pct = intensity * 0.2
            w, h = img.size
            crop_w = int(w * crop_pct)
            crop_h = int(h * crop_pct)
            return img.crop((crop_w, crop_h, w - crop_w, h - crop_h)).resize((w, h), Image.LANCZOS)

        elif transform_type == "noise":
            # Subtle noise - target texture (Rule 3)
            # Intensity maps to noise magnitude (0 to 30)
            noise_mag = int(intensity * 30)
            img_array = np.array(img)
            noise = np.random.randint(-noise_mag, noise_mag, img_array.shape).astype(np.int16)
            noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_array)

        else:
            return img

    def _compute_reward(self, features: dict[str, Any]) -> float:
        """Compute reward based on classification difficulty.

        Reward function encourages:
        - Higher entropy (more uncertainty)
        - Lower margin (more ambiguous between top classes)
        - Lower max_prob (less confident predictions)
        - But not exceeding max_difficulty (Rule 1: must remain solvable)

        Args:
            features: Classification features from classify_with_features.

        Returns:
            Reward value (higher is better for the agent).
        """
        entropy = features.get("entropy", 0.0)
        margin = features.get("margin", 0.0)
        max_prob = features.get("max_prob", 1.0)

        # Base reward: encourage difficulty
        reward = entropy * 2.0  # Higher entropy = more uncertain = better
        reward -= margin * 2.0  # Lower margin = more ambiguous = better
        reward -= max_prob * 2.0  # Lower confidence = harder = better

        # Penalty for exceeding max_difficulty (Rule 1: must remain solvable)
        if max_prob < (1.0 - self.max_difficulty):
            # If confidence is too low (too difficult), apply penalty
            penalty = (1.0 - self.max_difficulty - max_prob) * 10.0
            reward -= penalty

        # Bonus for maintaining some recognizability (Rule 2)
        # If entropy is reasonable but not chaotic, give bonus
        if 0.5 < entropy < 1.5:
            reward += 1.0

        return reward

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Action index (0-6: select transform, 7: apply and finalize).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.current_humanoid is None or self.current_image is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if action < len(self.transform_types):
            # Select transformation type
            self.current_transform_idx = action
            # Reset intensity when changing transform
            self.current_intensity = 0.0
            info["action_type"] = "select_transform"

        elif action == len(self.transform_types):
            # Apply current transformation with current intensity
            transformed_img = self._apply_transformation(
                self.current_transform_idx, self.current_intensity
            )

            # Temporarily save transformed image for classification
            temp_path = f"temp_transformed.png"
            transformed_img.save(temp_path)

            # Create temporary humanoid with transformed image
            temp_humanoid = Humanoid(
                fp=temp_path,
                state=self.current_humanoid.state,
                value=self.current_humanoid.value
            )

            # Get classification features for transformed image
            self.current_features = self.classifier.classify_with_features(temp_humanoid)

            # Compute reward based on difficulty
            reward = self._compute_reward(self.current_features)

            # Episode terminates after applying a transformation
            terminated = True
            info["action_type"] = "apply_transform"
            info["transform_type"] = self.transform_types[self.current_transform_idx]
            info["intensity"] = self.current_intensity
            info["features"] = self.current_features

            # Cleanup temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

        else:
            raise ValueError(f"Invalid action: {action}")

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Any:
        """Render the environment (optional, for debugging).

        Args:
            mode: Rendering mode (not implemented in this version).

        Returns:
            None (rendering not implemented).
        """
        pass
