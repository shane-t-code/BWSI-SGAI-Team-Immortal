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
