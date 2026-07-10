"""CNNClassifier — HumanoidClassifier backed by the trained DefaultCNN."""

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gameplay.constants import CLASSIFIER_MODEL_PATH
from gameplay.humanoid import Humanoid
from models.DefaultCNN import DefaultCNN
from models.HumanoidClassifier import HumanoidClassifier


class CNNClassifier(HumanoidClassifier):
    """Classifies humanoids using the trained DefaultCNN image classifier.

    Loads model weights from disk at construction time. Raises
    ``FileNotFoundError`` immediately if the weights file is missing — fail
    fast and see a clear error rather than silent uniform-probability
    classification.

    Attributes:
        num_classes: Number of output classes (default 4: zombie, healthy, injured, corpse).
        net: Loaded DefaultCNN model.
        device: Torch device used for inference (CUDA if available, else CPU).
        image_transforms: Torchvision preprocessing pipeline applied before inference.
        img_data_root: Root directory that image ``.fp`` paths are relative to.
    """

    num_classes: int
    net: DefaultCNN
    device: torch.device
    image_transforms: transforms.Compose
    img_data_root: str

    def __init__(
        self,
        num_classes: int = 4,
        model_file: str = CLASSIFIER_MODEL_PATH,
        img_data_root: str = "data",
    ) -> None:
        """Load CNN weights and set up the inference pipeline.

        Args:
            num_classes: Number of output classes.
            model_file: Path to the ``.pth`` weights file.
            img_data_root: Root directory that image ``.fp`` paths are
                relative to.

        Raises:
            FileNotFoundError: If ``model_file`` does not exist.
        """
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"CNN weights not found at '{model_file}'. "
                "Train the model first ('python main.py train --type cnn') or update "
                "CLASSIFIER_MODEL_PATH in gameplay/constants.py."
            )
        self.num_classes = num_classes
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.image_transforms = transforms.Compose(
            [
                # Resize to 512×512 — DefaultCNN's FC layer expects 128×128×24 features
                # (two MaxPool halvings of kernel_size=2: 512→256→128).
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    # ImageNet RGB channel statistics (mean and std per channel),
                    # computed over ~1.2M training images. Using these values normalizes
                    # inputs to approximately zero mean and unit variance — standard
                    # practice for CNN training even on non-ImageNet datasets.
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.img_data_root = img_data_root
        self._load_model(model_file)

    def _load_model(self, weights_path: str) -> None:
        """Load DefaultCNN weights from disk.

        Args:
            weights_path: Path to the ``.pth`` weights file.
        """
        self.net = DefaultCNN(self.num_classes)
        load_kwargs: dict = {"map_location": self.device}
        if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (
            2,
            0,
        ):
            load_kwargs["weights_only"] = True
        self.net.load_state_dict(torch.load(weights_path, **load_kwargs))  # nosec B614
        self.net.to(self.device)
        self.net.eval()  # disable Dropout2d during inference — must be called after load

    def classify(self, humanoid: Humanoid) -> np.ndarray:
        """Return CNN probability distribution over humanoid classes.

        Note:
            Designed for single-image inference only. The internal tensor is
            created with ``unsqueeze(0)`` (batch size 1) and the result is
            extracted with ``[0]``. Passing a batch is not supported and will
            silently return probabilities for only the first image.

        Args:
            humanoid: Humanoid instance with a .fp filepath attribute.

        Returns:
            Float32 array of shape ``(4,)`` with softmax probabilities in
            class order: ``[zombie, healthy, injured, corpse]``.
        """
        with Image.open(os.path.join(self.img_data_root, humanoid.fp)) as image_:
            tensor = (
                self.image_transforms(image_.convert("RGB"))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
        with torch.no_grad():
            outputs = self.net(tensor)
            return torch.nn.functional.softmax(outputs, 1)[0].cpu().numpy()

    def classify_with_features(
        self, humanoid: Humanoid
    ) -> dict[str, np.ndarray | float]:
        """Return classification with detailed features for RL training.

        Provides additional metrics beyond raw probabilities to help an RL agent
        learn how to generate harder-to-classify images.

        Args:
            humanoid: Humanoid instance with a .fp filepath attribute.

        Returns:
            Dictionary with keys:
                - ``probs``: Float32 array of shape ``(4,)`` with softmax probabilities.
                - ``max_prob``: Maximum probability (confidence in predicted class).
                - ``entropy``: Shannon entropy of the probability distribution.
                - ``margin``: Difference between top-2 probabilities (lower = more ambiguous).
                - ``predicted_class``: Index of the predicted class (0-3).
                - ``logits``: Raw pre-softmax outputs (logits).
        """
        with Image.open(os.path.join(self.img_data_root, humanoid.fp)) as image_:
            tensor = (
                self.image_transforms(image_.convert("RGB"))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
        with torch.no_grad():
            logits = self.net(tensor)
            probs = torch.nn.functional.softmax(logits, 1)[0].cpu().numpy()
            logits_np = logits[0].cpu().numpy()

        # Calculate additional metrics
        max_prob = float(np.max(probs))
        predicted_class = int(np.argmax(probs))

        # Shannon entropy: -sum(p * log(p))
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        # Margin: difference between top-2 probabilities
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0

        return {
            "probs": probs,
            "max_prob": max_prob,
            "entropy": entropy,
            "margin": margin,
            "predicted_class": predicted_class,
            "logits": logits_np,
        }

    def get_feature_activations(
        self, humanoid: Humanoid, layer: str = "conv2"
    ) -> np.ndarray:
        """Extract intermediate layer activations for analysis.

        Useful for understanding what features the CNN is using to classify
        an image, which can guide adversarial image generation.

        Args:
            humanoid: Humanoid instance with a .fp filepath attribute.
            layer: Which layer to extract activations from. Options: ``"conv1"``,
                ``"conv2"`` (default). ``"conv2"`` is recommended as it captures
                higher-level features.

        Returns:
            Float32 array of feature map activations. Shape depends on layer:
                - ``conv1``: ``(12, 256, 256)`` (12 channels, 256x256 spatial)
                - ``conv2``: ``(24, 128, 128)`` (24 channels, 128x128 spatial)
        """
        with Image.open(os.path.join(self.img_data_root, humanoid.fp)) as image_:
            tensor = (
                self.image_transforms(image_.convert("RGB"))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

        # Hook to capture intermediate activations
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # Register hook on requested layer
        if layer == "conv1":
            handle = self.net.conv1.register_forward_hook(get_activation("conv1"))
        elif layer == "conv2":
            handle = self.net.conv2.register_forward_hook(get_activation("conv2"))
        else:
            raise ValueError(f"Unknown layer: {layer}. Use 'conv1' or 'conv2'.")

        # Forward pass
        with torch.no_grad():
            _ = self.net(tensor)

        # Remove hook
        handle.remove()

        # Return activations as numpy array (remove batch dimension)
        return activations[layer][0].cpu().numpy()

    def get_gradients(
        self, humanoid: Humanoid, target_class: int | None = None
    ) -> np.ndarray:
        """Compute gradients w.r.t. input image for adversarial analysis.

        Shows which pixels most influence the classification, useful for
        identifying vulnerable features to attack.

        Args:
            humanoid: Humanoid instance with a .fp filepath attribute.
            target_class: If provided, compute gradients for maximizing this class.
                If None, use the predicted class.

        Returns:
            Float32 array of shape ``(3, 512, 512)`` with gradients for each RGB channel.
        """
        with Image.open(os.path.join(self.img_data_root, humanoid.fp)) as image_:
            tensor = (
                self.image_transforms(image_.convert("RGB"))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
        tensor.requires_grad = True

        # Forward pass
        outputs = self.net(tensor)

        # Use predicted class if target not specified
        if target_class is None:
            target_class = int(outputs.argmax(dim=1))

        # Compute gradient of target class score w.r.t. input
        target_score = outputs[0, target_class]
        target_score.backward()

        # Get gradients
        gradients = tensor.grad.detach()[0].cpu().numpy()
        tensor.requires_grad = False

        return gradients
