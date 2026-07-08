"""DefaultCNN — 4-class CNN classifier for humanoid images.

Classifies 512×512 RGB images into four categories: zombie, healthy, injured,
corpse. Based on the PyTorch CIFAR-10 tutorial:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultCNN(nn.Module):
    """Two-layer convolutional network for humanoid image classification.

    Architecture:
        conv1 (3→12 channels, 2×2 kernel) → ReLU → MaxPool(2)
        conv2 (12→24 channels, 3×3 kernel) → ReLU → MaxPool(2)
        Dropout2d(0.2) → Flatten → Linear(128×128×24 → num_classes)

    Input: 512×512 RGB images. conv1 (k=2, p=1) expands spatial dims to 513×513;
    two MaxPool halvings then reduce to 128×128. FC: 128 × 128 × 24 = 393,216.
    See the trace in __init__ for the full derivation — 513 is intentional.

    Attributes:
        conv1: First convolutional layer (3 → 12 channels).
        conv2: Second convolutional layer (12 → 24 channels).
        pool: Max pooling layer shared across both conv stages.
        drop: Dropout layer (p=0.2) applied before the FC layer.
        fc: Fully-connected output layer mapping features → num_classes.
    """

    def __init__(self, num_classes_: int = 4) -> None:
        """Initialize the network layers.

        Args:
            num_classes_: Number of output classes (default 4).
        """
        super().__init__()

        # Our images are RGB, so we have input channels = 3.
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=2, stride=1, padding=1
        )

        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1
        )

        # We in the end apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # A drop layer to help prevent overfitting.
        # Note: nn.Dropout2d drops entire *channels* (all spatial values for a channel),
        # not individual scalar features. At p=0.2, ~20% of the 24 feature maps are
        # zeroed each forward pass during training. This is different from nn.Dropout,
        # which zeros individual values independently.
        self.drop = nn.Dropout2d(p=0.2)

        # ── How conv and pool layers change spatial dimensions ───────────────────
        #
        # A conv layer slides a small filter (the "kernel") across the input.
        # A pool layer shrinks the input by taking the max in each non-overlapping tile.
        #
        # After every conv or pool, the new spatial size follows this formula:
        #
        #   out = floor( (W + 2*p - k) / s ) + 1
        #
        #   W = current width (same for height -- our images are square)
        #   k = kernel size -- how wide the filter is
        #   p = padding -- extra zeros added around the border before the filter runs
        #   s = stride -- how many pixels the filter advances between placements
        #
        # Why does this work? (W + 2*p) is the total padded width. The filter needs k
        # pixels to fit, so there are (W + 2*p - k) valid positions where it can
        # land. Dividing by s accounts for stride > 1. Floor handles any remainder.
        # MaxPool uses the same formula with p=0 and s=kernel_size. The final +1
        # accounts for the first kernel placement at the starting edge of the image.
        #
        # Full trace through this network:
        #
        #   Layer               W     k   p  s    Formula                          Output
        #   ------------------- ----  ---  -  --  -------------------------------- ---------
        #   Input               512                                               512 x 512
        #   conv1 (k=2, p=1)    512   2    1  1   floor((512 + 2 - 2) / 1) + 1    513 x 513
        #   pool  (k=2, s=2)    513   2    0  2   floor((513 + 0 - 2) / 2) + 1    256 x 256
        #   conv2 (k=3, p=1)    256   3    1  1   floor((256 + 2 - 3) / 1) + 1    256 x 256
        #   pool  (k=2, s=2)    256   2    0  2   floor((256 + 0 - 2) / 2) + 1    128 x 128
        #
        # Note: conv1 outputs 513 x 513, not 512 x 512 -- this is NOT a bug.
        # padding=1 adds one zero pixel on each side (512 padded to 514), and a
        # kernel=2 filter has 514 - 2 + 1 = 513 valid landing positions. You might
        # expect 512; the formula says 513 and the formula is right.
        #
        # FC input size: 128 x 128 spatial x 24 feature maps = 393,216 values.
        #
        # WARNING: if you change kernel_size, padding, or add/remove conv layers,
        # rerun this trace -- the FC size changes. A mismatch raises:
        #   RuntimeError: mat1 and mat2 shapes cannot be multiplied
        #
        # Further reading: https://cs231n.github.io/convolutional-networks/#conv
        self.fc = nn.Linear(in_features=128 * 128 * 24, out_features=num_classes_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In the forward function, pass the data through the layers we defined in the init function
        if x.shape[1:] != (3, 512, 512):
            raise ValueError(
                f"DefaultCNN expects input shape (N, 3, 512, 512), got {tuple(x.shape)}"
            )

        # conv1 → ReLU → MaxPool.
        # These two orderings are mathematically equivalent for max pooling:
        # max(ReLU(x)) == ReLU(max(x)). Either order is correct.
        x = self.pool(F.relu(self.conv1(x)))

        # conv2 → ReLU → MaxPool (same reasoning as above).
        x = self.pool(F.relu(self.conv2(x)))

        # Select some features to drop to prevent overfitting (only drop during training)
        x = self.drop(x)

        # Flatten
        x = x.view(-1, 128 * 128 * 24)
        # Feed to fully-connected layer to predict class

        x = self.fc(x)
        return x
