"""
Utility script to fine-tune the transfer classifier with an 80/20 train/validation
split and visualise evaluation metrics (confusion matrix sans gas_alarm and
accuracy-over-epochs curve).

Usage example:
    python Model/train_validate_eval.py --epochs 10 --batch-size 32
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Model.base_model_panns import (
    AudioEmbeddingDataset,
    PANNsCNN10,
    TransferClassifier,
    get_device,
    get_label_dict,
)


@dataclass
class TrainConfig:
    dataset_root: Path
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    split_ratio: float
    seed: int
    output_dir: Path


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]


EXCLUDED_LABELS = {"gas_alarm"}


def parse_args() -> TrainConfig:
    """Parse CLI arguments and return a dataclass for convenience."""

    parser = argparse.ArgumentParser(description="Train and evaluate the transfer classifier.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset") / "Dataset_copy",
        help="Root directory containing class-labelled subfolders.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularisation strength for Adam.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train ratio; remainder is used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Image") / "training_plots",
        help="Directory where evaluation figures will be saved.",
    )

    args = parser.parse_args()
    return TrainConfig(
        dataset_root=args.dataset_root.expanduser().resolve(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        split_ratio=args.split_ratio,
        seed=args.seed,
        output_dir=args.output_dir.expanduser().resolve(),
    )


def set_random_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch for deterministic behaviour."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(
    cfg: TrainConfig, embedding_model: PANNsCNN10
) -> tuple[DataLoader, DataLoader]:
    """
    Create train/validation splits (80/20 by default) for the audio embedding dataset
    and return their corresponding dataloaders.
    """

    dataset = AudioEmbeddingDataset(str(cfg.dataset_root), embedding_model)
    total_samples = len(dataset)
    if total_samples < 2:
        raise ValueError("Dataset needs at least two samples to create a train/validation split.")

    train_len = max(1, int(total_samples * cfg.split_ratio))
    val_len = total_samples - train_len
    if val_len == 0:
        raise ValueError(
            f"Validation set would be empty with split_ratio={cfg.split_ratio:.2f}. "
            "Please decrease the ratio."
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def infer_input_dim(sample_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Inspect one batch to determine the embedding dimensionality."""

    for features, _ in sample_loader:
        return features.shape[-1]
    raise RuntimeError("Failed to infer input dimension from the dataloader.")


def train_classifier_model(
    classifier: TransferClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[TransferClassifier, TrainingHistory]:
    """Train the classifier, track history, and keep the best validation checkpoint."""

    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = 0.0
    history = TrainingHistory(train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    for epoch in range(1, cfg.epochs + 1):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc = evaluate_classifier_model(classifier, val_loader, device)
        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        print(
            f"[Epoch {epoch:02d}/{cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier, history


def evaluate_classifier_model(
    classifier: TransferClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate classifier on a dataloader.

    Returns
    -------
    loss : float
        Average cross-entropy loss across the loader.
    accuracy : float
        Classification accuracy.
    """

    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = classifier(features)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)

    return avg_loss, acc


def collect_predictions(
    classifier: TransferClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather predictions and labels for plotting."""

    classifier.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = classifier(features)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def plot_accuracy_history(history: TrainingHistory, output_path: Path) -> None:
    """Plot training and validation accuracy over epochs."""

    epochs = range(1, len(history.train_acc) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history.train_acc, marker="o", label="Train Accuracy")
    ax.plot(epochs, history.val_acc, marker="s", label="Validation Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.set_xlim(1, len(history.train_acc))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_path: Path,
) -> None:
    """Create and save a confusion matrix heatmap with per-class accuracy values."""

    if len(label_names) == 0 or y_true.size == 0:
        print("No data available to plot confusion matrix without excluded classes.")
        return

    labels = list(range(len(label_names)))
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f", include_values=True)
    ax.set_title("Confusion Matrix (Per-Class Accuracy)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_precision_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    output_path: Path,
) -> None:
    """Create and save precision-recall curves for each class and the micro-average."""

    num_classes = len(label_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(12, 10))

    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_bin.ravel(), y_score.ravel()
    )
    ap_micro = average_precision_score(y_true_bin, y_score, average="micro")
    ax.step(
        recall_micro,
        precision_micro,
        where="post",
        label=f"micro-average (AP = {ap_micro:.2f})",
        color="black",
        linewidth=2.0,
    )

    for idx, name in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_score[:, idx])
        ap = average_precision_score(y_true_bin[:, idx], y_score[:, idx])
        ax.step(recall, precision, where="post", label=f"{name} (AP = {ap:.2f})", alpha=0.6)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def ensure_output_dir(path: Path) -> None:
    """Create the output directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = parse_args()
    set_random_seed(cfg.seed)
    ensure_output_dir(cfg.output_dir)

    device = get_device()
    print(f"Using device: {device}")

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {cfg.dataset_root}")

    panns_model = PANNsCNN10()
    # For feature extraction we keep the model on CPU; batching keeps memory in check.
    panns_model.eval()

    train_loader, val_loader = prepare_dataloaders(cfg, panns_model)

    label_dict = get_label_dict(str(cfg.dataset_root))
    label_items = sorted(label_dict.items(), key=lambda kv: kv[1])
    filtered_items = [item for item in label_items if item[0] not in EXCLUDED_LABELS]
    if not filtered_items:
        raise ValueError(
            "No classes available for evaluation after excluding: "
            f"{', '.join(sorted(EXCLUDED_LABELS))}"
        )
    label_names = [name for name, _ in filtered_items]
    num_classes = len(label_items)

    input_dim = infer_input_dim(train_loader)
    classifier = TransferClassifier(input_dim=input_dim, num_classes=num_classes)
    classifier, history = train_classifier_model(classifier, train_loader, val_loader, device, cfg)

    y_true_all, y_pred_all = collect_predictions(classifier, val_loader, device)

    included_indices = [idx for _, idx in filtered_items]
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(included_indices)}
    mask = np.isin(y_true_all, included_indices) & np.isin(y_pred_all, included_indices)

    y_true_filtered = y_true_all[mask]
    y_pred_filtered = y_pred_all[mask]

    y_true_mapped = np.array([index_mapping[idx] for idx in y_true_filtered])
    y_pred_mapped = np.array([index_mapping[idx] for idx in y_pred_filtered])

    cm_path = cfg.output_dir / "confusion_matrix.png"
    acc_curve_path = cfg.output_dir / "accuracy_curve.png"

    plot_confusion_matrix(y_true_mapped, y_pred_mapped, label_names, cm_path)
    plot_accuracy_history(history, acc_curve_path)

    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved accuracy curve to {acc_curve_path}")


if __name__ == "__main__":
    main()
