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
        return image.rotate(angle, expand=False, fillcolor=(128, 128, 128))


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


class FogTransform(ImageTransform):
    """Atmospheric haze — blends the whole frame toward pale gray (Rule 5).

    This is the Red Team's "fool the human" action. The zombie's single strongest
    human-visible tell is its bright-green skin; blending toward gray fades that
    green out so the figure reads as an ordinary person in foggy weather, while the
    CNN (which ignores color) barely reacts.

    Rule 2: Even at max intensity it does NOT fully erase the character — some green
    stays faintly visible (alpha caps at 0.75, not 1.0), so the figure is still there.
    Rule 4: Tunable via a single intensity knob.
    """

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """Blend the image toward pale gray haze.

        Args:
            image: Input PIL Image.
            intensity: Haze amount in [0, 1] (0 = no fog, 1 = heaviest fog).

        Returns:
            Hazed image.
        """
        # Work in float so the blend math doesn't overflow the 0-255 pixel range mid-way.
        arr = np.array(image.convert("RGB")).astype(np.float32)

        # The "fog color": a pale, slightly blue-gray. Every pixel gets pulled toward this.
        haze = np.array([200, 205, 210], dtype=np.float32)

        # alpha = how much of the final pixel is fog vs. original (0 = none, 1 = all fog).
        # Capped at 0.75 (not 1.0) on purpose: at full intensity the image is 75% fog,
        # 25% original — enough to gray out the green but leave the figure present (Rule 2).
        # NOTE: the 0.75 constant is load-bearing — the reward weights (0.55/0.45) were
        # tuned against fog's measured strength at this exact value. Do not change it
        # without re-running the swing analysis and re-balancing the weights.
        alpha = 0.75 * intensity

        # Alpha blend: each output pixel is a weighted average of original and fog.
        blended = arr * (1.0 - alpha) + haze * alpha

        # Clip back into the valid 0-255 range and convert to uint8 to make a real image.
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


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
    "fog": FogTransform(),  # Red Team "fool the human" action — fades the green tell
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


def green_mask(
    image: Image.Image,
    hue_low: int = 45,
    hue_high: int = 115,
    sat_min: int = 60,
) -> np.ndarray:
    """Find the zombie-green pixels in an image.

    "Green" is easiest to detect in HSV color space, where a color is described by
    Hue (which color on the rainbow), Saturation (how vivid vs. gray), and Value
    (how bright) — instead of RGB. A pixel counts as zombie-green if its hue is in
    the green band AND it is saturated enough to be real green (not a washed-out
    gray that merely leans green).

    Args:
        image: Input PIL Image.
        hue_low: Lower edge of the green hue band (PIL's 0-255 hue scale).
        hue_high: Upper edge of the green hue band.
        sat_min: Minimum saturation to count as vivid green (filters out gray).

    Returns:
        A boolean array of shape (height, width): True where the pixel is
        zombie-green, False elsewhere. Use this as a mask to select those pixels.
    """
    # Convert to HSV and split into the three channels. int16 (not uint8) so later
    # arithmetic in hue_proxy can't overflow or wrap around.
    hsv = np.array(image.convert("HSV")).astype(np.int16)
    h, s = hsv[..., 0], hsv[..., 1]

    # Green pixel = hue inside the green band AND saturated enough to be real green.
    return (h > hue_low) & (h < hue_high) & (s > sat_min)


def hue_proxy(image: Image.Image, mask: np.ndarray) -> float:
    """Score how human (non-green) the masked pixels look, in [0, 1].

    This is a stand-in ("proxy") for human perception: 1.0 means the character's
    original-green pixels now show no green at all (reads as a normal person),
    and low values mean they are still vividly green (reads as a zombie).

    IMPORTANT: the mask must be computed ONCE on the ORIGINAL image and then reused
    on every edited version. If you recomputed the mask on a fogged image, the green
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
    # Divide by 255 to land in ~[0, 1], average over all masked pixels, and multiply
    # by 4 so a fully-green character scores ~1.0 (the ×4 was calibrated on real data).
    green_excess = np.clip(((g - np.maximum(r, b)) / 255.0).mean() * 4.0, 0.0, 1.0)

    # Flip it so the reward points the right way: high = not green = human-plausible.
    return float(1.0 - green_excess)
