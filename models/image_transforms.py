# hello

"""Image transformation functions for adversarial generation.

All transformations follow the 5 rules:
1. Must remain theoretically solvable — not impossible
2. Character must still be present and roughly recognizable as a person  
3. Disguise/degradation should target the specific features the CNN was trained on
4. Difficulty should be tunable, not all-or-nothing
5. Stays within the airport/human-plausible setting
"""

import math
from typing import Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


class ImageTransform:
    """Base class for image transformations."""

    def __init__(self, min_intensity: float = 0.0, max_intensity: float = 1.0):
        """Initialize transformation with intensity bounds.

        Args:
            min_intensity: Minimum intensity (0.0 = no effect).
            max_intensity: Maximum intensity (1.0 = strongest allowed effect).
        """
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply transformation to image.

        Args:
            image: Input PIL Image.
            intensity: Intensity in [0, 1] range.

        Returns:
            Transformed PIL Image.
        """
        raise NotImplementedError("Subclasses must implement apply()")


class BlurTransform(ImageTransform):
    """Gaussian blur to target skin texture features (Rule 3).

    Rule 5: Simulates motion blur or camera focus issues - realistic.
    Rule 4: Tunable via blur radius.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply Gaussian blur.

        Args:
            image: Input PIL Image.
            intensity: Blur intensity (0 = no blur, 1 = max blur radius).

        Returns:
            Blurred image.
        """
        # Rule 4: Tunable - radius from 0 to 3 pixels
        radius = intensity * 3.0
        if radius < 0.1:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class BrightnessTransform(ImageTransform):
    """Brightness adjustment to target coloring features (Rule 3).

    Rule 5: Simulates lighting changes - realistic in airport setting.
    Rule 4: Tunable via brightness factor.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply brightness adjustment.

        Args:
            image: Input PIL Image.
            intensity: Brightness change (0 = darker, 0.5 = no change, 1 = brighter).

        Returns:
            Brightness-adjusted image.
        """
        # Rule 4: Tunable - factor from 0.5 to 1.5
        factor = 0.5 + intensity
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)


class ContrastTransform(ImageTransform):
    """Contrast adjustment to target visibility features (Rule 3).

    Rule 5: Simulates poor lighting or display issues - realistic.
    Rule 4: Tunable via contrast factor.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply contrast adjustment.

        Args:
            image: Input PIL Image.
            intensity: Contrast change (0 = lower contrast, 0.5 = no change, 1 = higher contrast).

        Returns:
            Contrast-adjusted image.
        """
        # Rule 4: Tunable - factor from 0.5 to 1.5
        factor = 0.5 + intensity
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)


class RotationTransform(ImageTransform):
    """Small rotation to target posture cues (Rule 3).

    Rule 2: Keep character recognizable - limit to small angles.
    Rule 5: Simulates camera angle or person leaning - realistic.
    Rule 4: Tunable via rotation angle.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply small rotation.

        Args:
            image: Input PIL Image.
            intensity: Rotation amount (0 = -15°, 0.5 = 0°, 1 = +15°).

        Returns:
            Rotated image.
        """
        # Rule 2 & 4: Small angles only (-15 to +15 degrees) to keep character visible
        angle = (intensity - 0.5) * 30.0
        if abs(angle) < 0.1:
            return image
        # Fill the corners that open up with MIRRORED scene content, not flat grey.
        # Flat grey wedges make the frame look tampered-with rather than "shot at an
        # angle" (Rule 5), and they advertise that an edit happened.
        # Verified the CNN reacts to the TILT, not the fill: grey wedges painted on an
        # upright image move P(healthy) 0.093 -> 0.089 (i.e. nothing), while a
        # wedge-free tilt reproduces the whole effect (0.639 vs 0.627 grey-filled).
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]
        padded = np.pad(arr, ((h // 2, h // 2), (w // 2, w // 2), (0, 0)), mode="reflect")
        rotated = Image.fromarray(padded).rotate(angle, expand=False)
        big_w, big_h = rotated.size
        left, top = (big_w - w) // 2, (big_h - h) // 2
        return rotated.crop((left, top, left + w, top + h))


class CropTransform(ImageTransform):
    """Slight crop to reduce visible information (Rule 3).

    Rule 2: Keep character present - limit crop amount.
    Rule 5: Simulates zoom or partial obstruction - realistic.
    Rule 4: Tunable via crop percentage.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply slight crop.

        Args:
            image: Input PIL Image.
            intensity: Crop amount (0 = no crop, 1 = max crop).

        Returns:
            Cropped and resized image.
        """
        # Rule 2 & 4: Limit crop to 20% to keep character visible
        crop_pct = intensity * 0.2
        if crop_pct < 0.01:
            return image

        w, h = image.size
        crop_w = int(w * crop_pct)
        crop_h = int(h * crop_pct)

        cropped = image.crop((crop_w, crop_h, w - crop_w, h - crop_h))
        return cropped.resize((w, h), Image.LANCZOS)


class NoiseTransform(ImageTransform):
    """Subtle noise to target texture features (Rule 3).

    Rule 5: Simulates low-quality camera or compression - realistic.
    Rule 4: Tunable via noise magnitude.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply subtle noise.

        Args:
            image: Input PIL Image.
            intensity: Noise amount (0 = no noise, 1 = max noise).

        Returns:
            Noisy image.
        """
        # Rule 4: Tunable - magnitude from 0 to 30
        noise_mag = int(intensity * 30)
        if noise_mag < 1:
            return image

        img_array = np.array(image)
        noise = np.random.randint(-noise_mag, noise_mag, img_array.shape).astype(np.int16)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)


class SaturationTransform(ImageTransform):
    """Saturation adjustment to target coloring features (Rule 3).

    Rule 5: Simulates lighting conditions or camera settings - realistic.
    Rule 4: Tunable via saturation factor.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply saturation adjustment.

        Args:
            image: Input PIL Image.
            intensity: Saturation change (0 = grayscale, 0.5 = no change, 1 = oversaturated).

        Returns:
            Saturation-adjusted image.
        """
        # Rule 4: Tunable - factor from 0.0 to 2.0
        factor = intensity * 2.0
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)


class ShadowTransform(ImageTransform):
    """Add shadow to simulate lighting/obstruction (Rule 5).

    Rule 3: Targets visibility features.
    Rule 2: Keep character recognizable - partial shadow only.
    Rule 4: Tunable via shadow intensity and position.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply shadow overlay.

        Args:
            image: Input PIL Image.
            intensity: Shadow intensity (0 = no shadow, 1 = heavy shadow).

        Returns:
            Image with shadow overlay.
        """
        if intensity < 0.1:
            return image

        img = image.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Rule 2 & 5: Partial shadow from one side (realistic lighting)
        # Rule 4: Tunable via alpha
        shadow_alpha = int(intensity * 150)  # Max alpha 150/255
        shadow_width = int(w * 0.3)  # Shadow covers 30% of image

        # Create gradient shadow from left side
        for x in range(shadow_width):
            alpha = int(shadow_alpha * (1 - x / shadow_width))
            draw.rectangle([x, 0, x + 1, h], fill=(0, 0, 0, alpha))

        return img


class OcclusionTransform(ImageTransform):
    """Add small occlusion to simulate objects in scene (Rule 5).

    Rule 2: Keep character recognizable - small occlusions only.
    Rule 3: Targets specific features.
    Rule 4: Tunable via occlusion size.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply small occlusion.

        Args:
            image: Input PIL Image.
            intensity: Occlusion amount (0 = no occlusion, 1 = max occlusion).

        Returns:
            Image with small occlusion.
        """
        if intensity < 0.1:
            return image

        img = image.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Rule 2 & 4: Small occlusion - max 10% of image area
        occlusion_size = int(min(w, h) * 0.1 * intensity)

        # Random position but avoid center (keep face/upper body visible)
        import random
        x = random.choice([
            random.randint(0, w // 4),  # Left side
            random.randint(3 * w // 4, w - occlusion_size),  # Right side
        ])
        y = random.randint(h // 2, h - occlusion_size)  # Lower half

        # Draw simple rectangular occlusion (simulates pillar, bag, etc.)
        draw.rectangle([x, y, x + occlusion_size, y + occlusion_size], fill=(100, 100, 100))

        return img


class RecolorTransform(ImageTransform):
    """Recolor the character's green skin toward human skin tone (Rule 3).

    The bright-green skin is the single strongest human-visible tell in these scenes.
    Unlike a global edit, this changes ONLY the figure's pixels and leaves the scene
    intact, so it fools a human while the CNN — which ignores color — still catches it. Validated in a pilot: human accuracy falls 100% -> ~50% -> 0%
    across intensity 0.4 / 0.75 / 1.0, while CNN P(healthy) stays ~0.

    Rule 2: keeps the figure present (it stays a figure, just skin-colored).
    Rule 4: tunable via intensity — this is the "malice budget" dial for the human.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Blend the character's green pixels toward skin tone.

        Uses skin_weight() (see the perception section below) to decide HOW MUCH to
        repaint each pixel, rather than an in/out mask. Two reasons that matters:

          * A hard mask mottles. Neighbouring fabric pixels land on opposite sides of
            the threshold, so a khaki vest comes out speckled brown. A continuous
            weight ramps instead of cliffs, so there is nothing to speckle.
          * A hard mask leaves a rim. Anti-aliased silhouette pixels are part figure,
            part background; a weight feathers them instead of leaving a green outline.

        Args:
            image: Input PIL Image.
            intensity: Recolor amount in [0, 1] (0 = untouched green, 1 = full skin).

        Returns:
            Recolored image (scene unchanged, figure skin-toned).
        """
        arr = np.array(image.convert("RGB")).astype(np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        # Target skin tone built from each pixel's own brightness, so the figure keeps
        # its light/dark shading (folds, shadows) instead of going flat. Skin: R>G>B.
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        tr = np.clip(lum * 1.12, 0, 255)
        tg = np.clip(lum * 0.80, 0, 255)
        tb = np.clip(lum * 0.66, 0, 255)

        # Per-pixel blend factor: how confidently this is skin, times how hard we edit.
        alpha = skin_weight(image) * intensity

        out = arr.copy()
        out[..., 0] = r * (1.0 - alpha) + tr * alpha
        out[..., 1] = g * (1.0 - alpha) + tg * alpha
        out[..., 2] = b * (1.0 - alpha) + tb * alpha
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class CompositeTransform:
    """Combines multiple transformations with controlled intensity.

    Rule 4: Provides tunable difficulty by combining effects proportionally.
    """

    def __init__(self, transforms: list[Tuple[ImageTransform, float]]):
        """Initialize composite transformation.

        Args:
            transforms: List of (transform, weight) tuples. Weights determine
                how much each transform contributes to total intensity.
        """
        self.transforms = transforms
        total_weight = sum(w for _, w in transforms)
        # Normalize weights
        self.transforms = [(t, w / total_weight) for t, w in transforms]

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply all transformations with weighted intensity.

        Args:
            image: Input PIL Image.
            intensity: Overall intensity (0 to 1).

        Returns:
            Transformed image.
        """
        result = image.copy()
        for transform, weight in self.transforms:
            # Each transform gets a portion of the total intensity
            transform_intensity = intensity * weight
            result = transform.apply(result, transform_intensity)
        return result


# Registry of available transformations
TRANSFORM_REGISTRY = {
    "blur": BlurTransform(),
    "brightness": BrightnessTransform(),
    "contrast": ContrastTransform(),
    "rotation": RotationTransform(),
    "crop": CropTransform(),
    "noise": NoiseTransform(),
    "saturation": SaturationTransform(),
    "shadow": ShadowTransform(),
    "occlusion": OcclusionTransform(),
    "recolor": RecolorTransform(),  # Red Team human channel — skin-tones the green tell
}


def get_transform(name: str) -> ImageTransform:
    """Get a transform by name.

    Args:
        name: Name of transformation from TRANSFORM_REGISTRY.

    Returns:
        ImageTransform instance.

    Raises:
        ValueError: If transform name not found.
    """
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORM_REGISTRY.keys())}")
    return TRANSFORM_REGISTRY[name]


def apply_random_transform(
    image: Image.Image,
    intensity: float,
    exclude: list[str] | None = None,
) -> Tuple[Image.Image, str]:
    """Apply a random transformation to an image.

    Args:
        image: Input PIL Image.
        intensity: Transformation intensity (0 to 1).
        exclude: List of transform names to exclude.

    Returns:
        Tuple of (transformed_image, transform_name).
    """
    available = list(TRANSFORM_REGISTRY.keys())
    if exclude:
        available = [t for t in available if t not in exclude]

    if not available:
        return image, "none"

    import random
    transform_name = random.choice(available)
    transform = get_transform(transform_name)
    return transform.apply(image, intensity), transform_name


# ---------------------------------------------------------------------------
# Human-plausibility perception ("does the character still look green?")
#
# These two functions are NOT image edits — they MEASURE an image. Together they
# form the "fool the human" half of the Red Team reward: green_mask finds the
# character's green pixels on the ORIGINAL zombie, and hue_proxy scores how green
# those same pixels still look after editing. Prajwal's tier-verification script
# imports hue_proxy too — this is the single shared definition, so the reward the
# agent optimizes and the metric he reports can never silently disagree.
# ---------------------------------------------------------------------------


# --- tuning constants for skin_weight, all measured rather than guessed -------------
# (diagnostics that produced them are described inline; see git history for the scripts)
_DEG = 360.0 / 255.0        # PIL packs hue into 0-255; convert to degrees for clarity

# Path 1 -- vivid zombie skin, by ABSOLUTE green dominance (g minus the larger of r/b).
# Absolute, not a ratio: a ratio test demands a bigger green lead from bright pixels
# than from dark ones, so it caught dark olive fabric (~7 levels of lead) while missing
# bright pale skin (~18 levels). One brightness bug, two symptoms -- speckled clothing
# AND green streaks left on skin. Measured clusters: fabric 5-9, pale skin 15-18,
# vivid skin 24-159. A sweep put the best split at 6.0 (0.2% skin missed, 0% fabric hit).
_DOM_LO, _DOM_RAMP = 6.0, 12.0

# Path 2 -- pale MINT skin, which path 1 physically cannot see: on ~17 of the 90 zombies
# the skin sits at dominance 5-11, identical to olive fabric, so no dominance threshold
# separates them. Those figures do differ in hue (mint skin 150-171 deg vs olive fabric
# 70-150). Ceiling is 170 because a blue prop staff in test_00089 starts at 179 deg and
# was being recoloured to skin tone; mint skin's 99th percentile is 171, so 170 excludes
# the staff without losing a single skin pixel.
_MINT_LO, _MINT_HI, _MINT_EDGE = 140.0, 170.0, 8.0
_MINT_SAT_LO, _MINT_SAT_RAMP = 50.0, 12.0

# Saturation gate applied to BOTH paths. The pale gray-green street sits below sat 35 and
# is the single most persistent false positive in this scene — it wrecked three separate
# diagnostics during development by masquerading as figure green. This is the guard.
_STREET_LO, _STREET_RAMP = 25.0, 20.0

_MASK_MIN_WEIGHT = 0.15     # weight above which green_mask() calls a pixel "the figure"
_HUE_PROXY_SCALE = 6.67     # calibrated so an untouched zombie scores ~1.0 (see hue_proxy)


def _ramp(x: np.ndarray, lo: float, width: float) -> np.ndarray:
    """Linear 0->1 ramp: 0 at or below `lo`, 1 at or above `lo + width`."""
    return np.clip((x - lo) / width, 0.0, 1.0)


def skin_weight(image: Image.Image) -> np.ndarray:
    """How confidently each pixel is zombie SKIN, as a float in [0, 1].

    This is the single shared definition of "the character's green": RecolorTransform
    uses it to decide how hard to repaint each pixel, and green_mask() thresholds it to
    decide which pixels hue_proxy scores. Deriving both from one function is deliberate
    — when the transform and the metric each had their own idea of "green" they
    disagreed, and the reward ended up grading pixels the transform could not change.

    Two detection paths, combined with max() so a pixel only needs to satisfy one:
      1. green dominance -> vivid zombie skin
      2. mint hue band   -> pale skin that dominance is blind to
    Then a saturation gate over both, to keep the gray-green street out.

    Args:
        image: Input PIL Image.

    Returns:
        Float array of shape (height, width), 0.0 = definitely not skin, 1.0 = certainly.
    """
    arr = np.array(image.convert("RGB")).astype(np.float32)
    hsv = np.array(image.convert("HSV")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    hue, sat = hsv[..., 0] * _DEG, hsv[..., 1]

    # Path 1: how far green leads the strongest of the other two channels.
    w_dominance = _ramp(g - np.maximum(r, b), _DOM_LO, _DOM_RAMP)

    # Path 2: inside the mint hue band (soft on both edges) and saturated enough to be
    # a real colour rather than a grey that happens to lean green.
    w_mint = (
        _ramp(hue, _MINT_LO, _MINT_EDGE)
        * (1.0 - _ramp(hue, _MINT_HI, _MINT_EDGE))
        * _ramp(sat, _MINT_SAT_LO, _MINT_SAT_RAMP)
    )

    return np.maximum(w_dominance, w_mint) * _ramp(sat, _STREET_LO, _STREET_RAMP)


def green_mask(
    image: Image.Image,
    hue_low: int = 45,
    hue_high: int = 115,
    sat_min: int = 60,
) -> np.ndarray:
    """Find the zombie-green pixels in an image.

    Now a thresholded view of skin_weight(), so the pixels the reward SCORES are exactly
    the pixels the recolor transform can CHANGE. The previous version was an independent
    HSV band (~63-162 deg) which missed mint skin at 160-171 — meaning on pale figures
    the agent was being graded on pixels it had no action for.

    Args:
        image: Input PIL Image.
        hue_low: Unused. Kept so existing callers passing it keep working.
        hue_high: Unused. Kept for backward compatibility.
        sat_min: Unused. Kept for backward compatibility.

    Returns:
        A boolean array of shape (height, width): True where the pixel is
        zombie-green, False elsewhere. Use this as a mask to select those pixels.
    """
    return skin_weight(image) > _MASK_MIN_WEIGHT


def hue_proxy(image: Image.Image, mask: np.ndarray) -> float:
    """Score how human (non-green) the masked pixels look, in [0, 1].

    This is a stand-in ("proxy") for human perception: 1.0 means the character's
    original-green pixels now show no green at all (reads as a normal person),
    and low values mean they are still vividly green (reads as a zombie).

    IMPORTANT: the mask must be computed ONCE on the ORIGINAL image and then reused
    on every edited version. If you recomputed the mask on the edited image, the green
    would already be gone, the mask would be empty, and this would trivially return
    1.0 — measuring nothing. Freezing the mask forces the honest question: did the
    edit actually push the character's pixels away from green?

    Args:
        image: The (possibly edited) PIL Image to score.
        mask: Boolean array from green_mask(original), same H×W as the image.

    Returns:
        Float in [0, 1]. Higher = less green = more human-looking.
    """
    arr = np.array(image.convert("RGB")).astype(np.float32)

    # No green pixels to begin with (e.g. a human image) → nothing to hide → looks human.
    if mask.sum() == 0:
        return 1.0

    # Pull out just the character's original-green pixels (mask selects them).
    r, g, b = arr[..., 0][mask], arr[..., 1][mask], arr[..., 2][mask]

    # "Green excess": how much the green channel dominates the other two.
    #   vivid green → g >> r and g >> b → large positive number (still a zombie)
    #   skin / gray → r, g, b close together → near zero (looks human)
    #
    # The per-pixel clamp at 0 is the important part. Skin-toned pixels have NEGATIVE
    # excess (r > g), so without it a well-recoloured region cancels out a region that
    # is still green — the average hides blotches instead of reporting them. That bug
    # scored one image a perfect 1.0000 while 766 of its pixels were still visibly
    # green, and it flattened the reward above 0.75 recolor, so the agent correctly
    # concluded there was nothing left to gain and stopped early. Clamping first means
    # leftover green can only ever push the score DOWN.
    per_pixel = np.clip((g - np.maximum(r, b)) / 255.0, 0.0, None)

    # Scale so an untouched zombie lands at ~1.0. Re-derived after the clamp changed
    # the mean (the old ×4 was calibrated against the signed version).
    green_excess = np.clip(per_pixel.mean() * _HUE_PROXY_SCALE, 0.0, 1.0)

    # Flip it so the reward points the right way: high = not green = human-plausible.
    return float(1.0 - green_excess)
