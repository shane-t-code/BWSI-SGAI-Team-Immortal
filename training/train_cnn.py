"""CNN training script for the BWSI SGAI project.

Trains DefaultCNN on the consolidated dataset and saves weights to
models/weights/baselineCNN.pth. Designed as a fast baseline — extend it
for your own experiments (more epochs, augmentation, fine-tuning, etc.).

Train via:
    python main.py train --type cnn
    python main.py train --type cnn -e 20 -s 7
    python main.py train --help

Hyperparameters to tune:
    NUM_EPOCHS      — total training epochs (default: 10)
    BATCH_SIZE      — images per gradient step (default: 8)
    LEARNING_RATE   — Adam optimizer step size (default: 0.001)
    VAL_SPLIT       — fraction of data held out for validation (default: 0.2)

Accuracy improvements to try (with a small dataset like this one):
    - Data augmentation: there's a ColorJitter transform commented out in
      train() — uncomment it and see what happens to your training curves.
      What do you expect? Also try adjusting rotation degrees or flip probability.
    - Regularization: pass weight_decay=1e-4 to torch.optim.Adam, or add
      Dropout layers to DefaultCNN — reduces overfitting on small datasets
    - Early stopping: track best val accuracy and save weights at the peak
      instead of the final epoch — peak epoch varies by dataset and seed
"""

import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# When run directly as a script (python training/train_cnn.py), Python adds
# training/ to sys.path — not the project root. This block ensures the root
# is on sys.path so local packages (game_interfaces, gameplay, models) import.
# Has no effect when imported as a module (main.py adds the root itself).
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from game_interfaces.data_parser import datarow_to_state  # noqa: E402
from gameplay.constants import CLASSIFIER_MODEL_PATH, DATA_DIR  # noqa: E402
from gameplay.enums import State  # noqa: E402
from gameplay.humanoid import MAP_CLASS_STR_TO_INT  # noqa: E402
from models.DefaultCNN import DefaultCNN  # noqa: E402

# ── Hyperparameters ────────────────────────────────────────────────────────────
NUM_EPOCHS: int = 10
BATCH_SIZE: int = 8
LEARNING_RATE: float = 0.001
VAL_SPLIT: float = 0.2

SEED: int | None = 42

NUM_CLASSES: int = len(State)

CNN_LOG_DIR: str = "training/CNN-logs"
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _datarow_to_label(row: pd.Series) -> int:
    """Map a metadata CSV row to an integer class label.

    Delegates to :func:`~game_interfaces.data_parser.datarow_to_state` so
    the CNN trains on exactly the same class definitions the game uses at
    inference time. Customize state mapping there, not here.

    Args:
        row: A pandas Series (one row of the metadata DataFrame).

    Returns:
        Integer class index matching :data:`~gameplay.humanoid.MAP_CLASS_STR_TO_INT`.
    """
    return MAP_CLASS_STR_TO_INT[datarow_to_state(row)]


class HumanoidDataset(Dataset):
    """PyTorch Dataset wrapping the consolidated humanoid image dataset.

    Attributes:
        img_data_root: Root directory that image paths are relative to.
        transform: Optional torchvision transform pipeline applied to each image.
        image_paths: List of relative image paths from the metadata CSV.
        labels: Integer class label for each image.
    """

    img_data_root: str
    transform: transforms.Compose | None
    image_paths: list[str]
    labels: list[int]

    def __init__(
        self,
        metadata_path: str,
        img_data_root: str,
        transform: "transforms.Compose | None" = None,
    ) -> None:
        """Load the metadata CSV and build the image/label lists.

        Args:
            metadata_path: Path to ``consolidated_metadata.csv``.
            img_data_root: Root directory that image paths are relative to.
            transform: Optional torchvision transform pipeline to apply to
                each image when loading.
        """
        self.img_data_root = img_data_root
        self.transform = transform

        df = pd.read_csv(metadata_path)
        self.image_paths = df["Filename"].tolist()
        self.labels = [_datarow_to_label(row) for _, row in df.iterrows()]

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, int]:
        """Load and return a single (image, label) pair.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of ``(image_tensor, label)`` where ``image_tensor`` is the
            result of applying ``self.transform`` (or a raw PIL Image if no
            transform is set), and ``label`` is the integer class index.
        """
        img_path = os.path.join(self.img_data_root, self.image_paths[idx])
        with Image.open(img_path) as image_:
            image_ = image_.convert("RGB")  # type: ignore[assignment]
            if self.transform:
                image_ = self.transform(image_)
        return image_, self.labels[idx]


def _save_cnn_plots(
    history: dict,
    model: DefaultCNN,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    log_dir: str,
) -> None:
    """Save training curve and confusion matrix plots to log_dir.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc`` — one value per epoch.
        model: Trained model (used for the confusion matrix eval pass).
        val_loader: DataLoader for the validation set.
        device: Device the model is on.
        class_names: Ordered list of class name strings (index matches label int).
        log_dir: Directory to write plot files into.
    """
    os.makedirs(log_dir, exist_ok=True)
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # ── Training curves — loss and accuracy side by side ──────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, history["train_loss"], marker="o", label="Train")
    ax1.plot(epochs_range, history["val_loss"], marker="o", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["train_acc"], marker="o", label="Train")
    ax2.plot(epochs_range, history["val_acc"], marker="o", label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CNN Training Curves", fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(log_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {curves_path}")

    # ── Confusion matrix — eval pass over validation set ─────────────────
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Validation Set)")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.tight_layout()
    cm_path = os.path.join(log_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")


def train(
    data_dir: str = DATA_DIR,
    epochs: int = NUM_EPOCHS,
    seed: int | None = SEED,
    save_path: str = CLASSIFIER_MODEL_PATH,
    log_dir: str = CNN_LOG_DIR,
) -> None:
    """Train DefaultCNN on the consolidated dataset and save weights.

    Loads the dataset from ``data_dir/consolidated_metadata.csv``, splits it
    into train/val sets, runs ``epochs`` of training with Adam +
    CrossEntropyLoss, saves the final weights to ``save_path``, and writes
    training curve and confusion matrix plots to ``log_dir``.

    Args:
        data_dir: Root directory containing images and
            ``consolidated_metadata.csv``.
        epochs: Total number of training epochs.
        seed: Random seed for reproducibility. Pass None to disable seeding.
        save_path: Path to write the trained model weights (.pth).
        log_dir: Directory to write training plots into (default: CNN-logs/).
    """
    if seed is not None:
        set_seed(seed)

    metadata_path = os.path.join(data_dir, "consolidated_metadata.csv")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # Normalization statistics must match CNNClassifier.classify() exactly —
    # the same mean/std applied at training time must be applied at inference
    # time, or the model sees a shifted input distribution and performs poorly.
    _normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],  # ImageNet channel statistics
    )

    # Training transform: augmentation + normalize.
    # Augmentation shows the model slightly different versions of each image
    # every epoch, which helps it learn generalizable features instead of
    # memorizing exact pixels. This reduces overfitting on small datasets —
    # but there's a hard limit: no matter how many ways you transform 160
    # images, you can't manufacture information that isn't there. Augmentation
    # buys you something, but it's not a substitute for more training data.
    #
    # RandomHorizontalFlip is the one augmentation active in this baseline.
    # Mirroring left↔right preserves both class-distinguishing features in
    # this dataset (color and upright/prone orientation), so it's always safe.
    # It's also the only one that consistently improves validation accuracy
    # without slowing or breaking convergence on a dataset this small.
    #
    # The commented-out transforms below are experiments for you to try.
    # Each one changes what the model sees during training in a different way —
    # run them and watch what happens to the training curves.
    #
    # Augmentation is applied ONLY to training images — validation images stay
    # clean so accuracy numbers reflect real generalisation, not augmented luck.
    train_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            # ^ ColorJitter randomly shifts brightness, contrast, saturation, and hue.
            # Think about which visual feature most separates zombies from humans
            # in this dataset, then predict what ColorJitter does to that signal.
            # transforms.RandomRotation(degrees=15),
            # ^ RandomRotation tilts images up to ±15°. Two of the four classes
            # differ by orientation (upright vs. prone). What happens to that
            # signal when a lying-down image gets rotated 90°?
            transforms.ToTensor(),
            _normalize,
        ]
    )

    # Validation transform: resize + normalize only — no augmentation.
    val_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            _normalize,
        ]
    )

    # Build two dataset instances over the same files — one per transform.
    # We then assign the same split indices to both so train and val images
    # never overlap, while each split gets its own transform pipeline.
    train_dataset = HumanoidDataset(metadata_path, data_dir, transform=train_transforms)
    val_dataset = HumanoidDataset(metadata_path, data_dir, transform=val_transforms)

    n_total = len(train_dataset)
    val_size = max(1, int(n_total * VAL_SPLIT))
    train_size = n_total - val_size
    if train_size < 1:
        raise ValueError(
            f"Not enough data for training — dataset has {n_total} sample(s) but "
            f"VAL_SPLIT={VAL_SPLIT} reserves {val_size} for validation, leaving 0 for training. "
            "Reduce VAL_SPLIT or add more data."
        )
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_set = torch.utils.data.Subset(train_dataset, indices[:train_size])
    val_set = torch.utils.data.Subset(val_dataset, indices[train_size:])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {train_size} samples | Val: {val_size} samples")

    model = DefaultCNN(num_classes_=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = running_loss / train_size
        train_acc = correct / train_size

        # ── Validation pass ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss /= val_size
        val_acc = val_correct / val_size

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:>2}/{epochs} | "
            f"train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
            f"val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
        )

    # NOTE: With augmentation enabled, this baseline overfits less than before —
    # train accuracy still climbs faster than val accuracy, but the gap should
    # be smaller and val accuracy should peak higher than the ~50% seen without
    # augmentation. Some overfitting is still expected: the dataset has ~160
    # training images across 4 classes, so augmentation expands variety per
    # epoch but does not add new ground-truth examples. This is still the floor
    # the floor you're meant to beat. See the module docstring for further
    # improvements (regularization, early stopping).
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nWeights saved to {save_path}")

    class_names = [s.value for s in State]
    _save_cnn_plots(history, model, val_loader, device, class_names, log_dir)


if __name__ == "__main__":
    # Allow running this script directly from the training/ directory:
    #   python train_cnn.py
    # Sets cwd to project root so relative paths (DATA_DIR="data") resolve
    # correctly. sys.path is already fixed at import time (see top of file).
    # To pass flags (-e, -s, -d, -o), use the main CLI instead:
    #   python main.py train --type cnn
    os.chdir(_project_root)
    train()
