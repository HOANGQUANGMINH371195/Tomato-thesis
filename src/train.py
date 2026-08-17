"""One-command reproducible training for classification and segmentation.

Examples:
    uv run python -m src.train --task classification
    uv run python -m src.train --task segmentation
    uv run python -m src.train --task all

Every run writes its config, environment, epoch metrics, checkpoints, plots,
and a human-readable report under artifacts/runs/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from .data_gate import validate_root
from .experiment import ExperimentRun, mark_active_failure
from .mcunet_unet import MCUNetSegmentation
from .report_artifacts import (profile_model, resource_snapshot, save_architecture_figure,
                               save_classification_ablation,
                               save_dataset_distribution, save_dataset_examples,
                               save_segmentation_overlay_grid, save_workflow_figure,
                               save_yield_sensitivity)
from .yield_model import yield_loss_interval

CLASS_NAMES = ["healthy", "early_blight", "late_blight", "bacterial_spot"]
SEGMENTATION_NAMES = ["background", "early_blight", "late_blight", "bacterial_spot"]
SEGMENTATION_LABEL_TO_ID = {name: i for i, name in enumerate(SEGMENTATION_NAMES)}


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value); stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams: stream.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_info(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
    return str(device)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:
        if requested == "cuda":
            raise RuntimeError(f"CUDA was explicitly requested but initialization failed: {error}") from error
        print(f"Warning: CUDA unavailable ({error}); using CPU")
        return torch.device("cpu")
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was explicitly requested but torch.cuda.is_available() is false; check NVIDIA driver and PyTorch CUDA wheel")
    if available:
        try:
            torch.cuda.init()
            return torch.device("cuda")
        except Exception as error:
            if requested == "cuda":
                raise RuntimeError(f"CUDA driver/toolkit mismatch: {error}") from error
            print(f"Warning: CUDA initialization failed ({error}); using CPU")
    return torch.device("cpu")


class ClassificationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        augmentation = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ] if train else [transforms.Resize((image_size, image_size))]
        self.transform = transforms.Compose(
            augmentation
            + [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        return self.transform(image), CLASS_NAMES.index(row.label)


class SegmentationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
        with Image.open(row.mask_path) as mask:
            mask = mask.convert("L")
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )(image)
        return tensor, torch.from_numpy(np.asarray(mask, dtype=np.int64))


class ResidualAttentionHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(x + self.block(x), inplace=True)
        return residual * self.gate(residual) + residual


class Classifier(nn.Module):
    def __init__(self, num_classes: int, attention: bool = True, pretrained: bool = True,
                 allow_random_init: bool = False) -> None:
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        try:
            backbone = models.densenet121(weights=weights)
        except Exception as error:
            if not allow_random_init:
                raise RuntimeError("ImageNet weights could not be loaded. Use --allow-random-init only for a deliberate baseline.") from error
            print(f"Warning: using random initialization by explicit request ({error})")
            backbone = models.densenet121(weights=None)
        self.features = backbone.features
        self.attention = ResidualAttentionHead(1024) if attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.features(x), inplace=True)
        x = self.attention(x)
        return self.classifier(torch.flatten(self.pool(x), 1))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(self.pool(x))


class ResidualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.attention = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.attention(self.block(x)) + self.skip(x), inplace=True)


class AttentionResidualUNet(nn.Module):
    def __init__(self, num_classes: int = 4, base: int = 32) -> None:
        super().__init__()
        self.down1 = ResidualConv(3, base)
        self.down2 = ResidualConv(base, base * 2)
        self.down3 = ResidualConv(base * 2, base * 4)
        self.down4 = ResidualConv(base * 4, base * 8)
        self.bottom = ResidualConv(base * 8, base * 16)
        self.up4 = ResidualConv(base * 16 + base * 8, base * 8)
        self.up3 = ResidualConv(base * 8 + base * 4, base * 4)
        self.up2 = ResidualConv(base * 4 + base * 2, base * 2)
        self.up1 = ResidualConv(base * 2 + base, base)
        self.head = nn.Conv2d(base, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def up(self, x: torch.Tensor, skip: torch.Tensor, block: nn.Module) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return block(torch.cat([x, skip], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x = self.bottom(self.pool(x4))
        x = self.up(x, x4, self.up4)
        x = self.up(x, x3, self.up3)
        x = self.up(x, x2, self.up2)
        x = self.up(x, x1, self.up1)
        return self.head(x)


# Backwards-compatible import name for old local notebooks.
UNet = AttentionResidualUNet


def dice_loss(logits: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, classes).permute(0, 3, 1, 2).float()
    intersection = (probability * target_one_hot).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice[1:].mean()


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, classes: int) -> dict[str, float]:
    prediction = logits.argmax(1)
    counts: dict[str, float] = {}
    for class_id in range(classes):
        pred = prediction == class_id
        true = target == class_id
        counts[f"{SEGMENTATION_NAMES[class_id]}_tp"] = float((pred & true).sum().item())
        counts[f"{SEGMENTATION_NAMES[class_id]}_fp"] = float((pred & ~true).sum().item())
        counts[f"{SEGMENTATION_NAMES[class_id]}_fn"] = float((~pred & true).sum().item())
        counts[f"{SEGMENTATION_NAMES[class_id]}_tn"] = float((~pred & ~true).sum().item())
    return metrics_from_segmentation_counts(counts, classes)


def metrics_from_segmentation_counts(counts: dict[str, float], classes: int) -> dict[str, float]:
    output: dict[str, float] = {}
    dices: list[float] = []; ious: list[float] = []; precisions: list[float] = []; recalls: list[float] = []
    all_ious: list[float] = []; all_recalls: list[float] = []; weighted_iou = 0.0
    disease_tp = 0.0; disease_fp = 0.0; disease_fn = 0.0
    total_correct = 0.0; total_pixels = 0.0
    for class_id in range(classes):
        name = SEGMENTATION_NAMES[class_id]
        tp = counts.get(f"{name}_tp", 0.0); fp = counts.get(f"{name}_fp", 0.0)
        fn = counts.get(f"{name}_fn", 0.0); tn = counts.get(f"{name}_tn", 0.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        npv = tn / max(tn + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        iou = tp / max(tp + fp + fn, 1.0)
        mcc_denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
        mcc = (tp * tn - fp * fn) / mcc_denominator
        support = tp + fn
        for suffix, value in {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "support": support,
                              "precision": precision, "recall": recall, "sensitivity": recall,
                              "specificity": specificity, "npv": npv, "f1": f1, "dice": f1, "iou": iou}.items():
            output[f"{name}_{suffix}"] = float(value)
        output[f"{name}_balanced_accuracy"] = float((recall + specificity) / 2)
        output[f"{name}_mcc"] = float(mcc)
        # Human-friendly aliases used by figures and report tables.
        output[f"precision_{name}"] = float(precision)
        output[f"recall_{name}"] = float(recall)
        output[f"specificity_{name}"] = float(specificity)
        output[f"f1_{name}"] = float(f1)
        output[f"dice_{name}"] = float(f1)
        output[f"iou_{name}"] = float(iou)
        if class_id > 0:
            dices.append(f1); ious.append(iou); precisions.append(precision); recalls.append(recall)
            disease_tp += tp; disease_fp += fp; disease_fn += fn
        all_ious.append(iou); all_recalls.append(recall)
        total_correct += tp; total_pixels += tp + fn
        weighted_iou += support * iou
    output["dice_macro"] = float(np.mean(dices))
    output["iou_macro"] = float(np.mean(ious))
    output["precision_macro"] = float(np.mean(precisions))
    output["recall_macro"] = float(np.mean(recalls))
    output["pixel_accuracy"] = float(total_correct / max(total_pixels, 1.0))
    output["mean_iou_all_classes"] = float(np.mean(all_ious))
    output["mean_class_accuracy_all_classes"] = float(np.mean(all_recalls))
    output["frequency_weighted_iou"] = float(weighted_iou / max(total_pixels, 1.0))
    output["dice_micro_diseases"] = float(2 * disease_tp / max(2 * disease_tp + disease_fp + disease_fn, 1.0))
    output["iou_micro_diseases"] = float(disease_tp / max(disease_tp + disease_fp + disease_fn, 1.0))
    return output


def segmentation_boundary_metrics(logits: torch.Tensor, target: torch.Tensor, classes: int) -> dict[str, float]:
    """Boundary IoU and symmetric Hausdorff distance for report-time evaluation."""
    predictions = logits.argmax(1).detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    values: dict[str, list[float]] = {}
    kernel = np.ones((3, 3), np.uint8)
    diagonal = float(np.hypot(target.shape[-2], target.shape[-1]))
    for prediction, truth in zip(predictions, targets):
        for class_id, name in enumerate(SEGMENTATION_NAMES):
            pred = (prediction == class_id).astype(np.uint8); true = (truth == class_id).astype(np.uint8)
            pred_boundary = pred ^ cv2.erode(pred, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            true_boundary = true ^ cv2.erode(true, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            pred_tolerant = cv2.dilate(pred_boundary, kernel, iterations=1).astype(bool)
            true_tolerant = cv2.dilate(true_boundary, kernel, iterations=1).astype(bool)
            intersection = np.count_nonzero((pred_boundary.astype(bool) & true_tolerant) | (true_boundary.astype(bool) & pred_tolerant))
            union = np.count_nonzero(pred_boundary | true_boundary)
            if union == 0:
                continue
            boundary_iou = float(intersection / union)
            pred_points = np.argwhere(pred_boundary); true_points = np.argwhere(true_boundary)
            if not len(pred_points) and not len(true_points):
                hausdorff = 0.0
            elif not len(pred_points) or not len(true_points):
                hausdorff = diagonal
            else:
                distance_to_true = cv2.distanceTransform((~true_boundary.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
                distance_to_pred = cv2.distanceTransform((~pred_boundary.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
                hausdorff = float(max(distance_to_true[pred_boundary.astype(bool)].max(initial=0),
                                      distance_to_pred[true_boundary.astype(bool)].max(initial=0)))
            values.setdefault(f"{name}_boundary_iou", []).append(boundary_iou)
            values.setdefault(f"{name}_hausdorff_pixels", []).append(hausdorff)
    output = {key: float(np.mean(items)) for key, items in values.items()}
    boundary_scores = [output[f"{name}_boundary_iou"] for name in SEGMENTATION_NAMES[1:] if f"{name}_boundary_iou" in output]
    hausdorff_scores = [output[f"{name}_hausdorff_pixels"] for name in SEGMENTATION_NAMES[1:] if f"{name}_hausdorff_pixels" in output]
    output["boundary_iou_macro"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    output["hausdorff_pixels_macro"] = float(np.mean(hausdorff_scores)) if hausdorff_scores else 0.0
    return output


def classification_metrics(true: list[int], predicted: list[int], probabilities: list[list[float]] | None = None) -> dict[str, Any]:
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 classification_report, cohen_kappa_score, confusion_matrix,
                                 log_loss, matthews_corrcoef, roc_auc_score)

    report = classification_report(
        true, predicted, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(true, predicted, labels=list(range(len(CLASS_NAMES))))
    total = int(cm.sum())
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i]); fn = int(cm[i, :].sum() - tp); fp = int(cm[:, i].sum() - tp); tn = int(total - tp - fn - fp)
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1); npv = tn / max(tn + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "support": tp + fn,
                           "precision": precision, "recall": recall, "sensitivity": recall,
                           "specificity": specificity, "npv": npv, "f1": f1,
                           "balanced_accuracy": (recall + specificity) / 2}
    output: dict[str, Any] = {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "matthews_correlation_coefficient": float(matthews_corrcoef(true, predicted)),
        "cohen_kappa": float(cohen_kappa_score(true, predicted)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    if probabilities:
        probability = np.asarray(probabilities, dtype=np.float64)
        probability /= np.clip(probability.sum(axis=1, keepdims=True), 1e-12, None)
        labels_one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[np.asarray(true)]
        confidence = probability.max(axis=1); correctness = np.asarray(predicted) == np.asarray(true)
        bins = np.linspace(0.0, 1.0, 11); ece = 0.0
        for lower, upper in zip(bins[:-1], bins[1:]):
            selected = (confidence > lower) & (confidence <= upper)
            if selected.any():
                ece += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
        output.update({
            "roc_auc_ovr_macro": float(roc_auc_score(labels_one_hot, probability, multi_class="ovr", average="macro")),
            "roc_auc_ovr_weighted": float(roc_auc_score(labels_one_hot, probability, multi_class="ovr", average="weighted")),
            "average_precision_macro": float(average_precision_score(labels_one_hot, probability, average="macro")),
            "multiclass_log_loss": float(log_loss(true, probability, labels=list(range(len(CLASS_NAMES))))),
            "multiclass_brier_score": float(np.mean(np.sum((probability - labels_one_hot) ** 2, axis=1))),
            "expected_calibration_error_10_bins": float(ece),
        })
        for class_id, name in enumerate(CLASS_NAMES):
            output["per_class"][name]["roc_auc"] = float(roc_auc_score(labels_one_hot[:, class_id], probability[:, class_id]))
            output["per_class"][name]["average_precision"] = float(average_precision_score(labels_one_hot[:, class_id], probability[:, class_id]))
    return output


def save_metrics(run: ExperimentRun, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with (run.path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (run.path / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_metrics_history(run: ExperimentRun, before_epoch: int) -> list[dict[str, Any]]:
    """Load completed epochs when resuming without duplicating a partial epoch."""
    target = run.path / "metrics.json"
    if target.exists():
        try:
            rows = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                return [row for row in rows if int(row.get("epoch", 0)) < before_epoch]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    csv_path = run.path / "metrics.csv"
    if csv_path.exists():
        try:
            rows = pd.read_csv(csv_path).to_dict(orient="records")
            return [row for row in rows if int(row.get("epoch", 0)) < before_epoch]
        except (OSError, ValueError, TypeError):
            pass
    return []


def save_classification_figures(run: ExperimentRun, rows: list[dict[str, Any]], metrics: dict[str, Any],
                                true: list[int] | None = None, probabilities: list[list[float]] | None = None) -> None:
    figures = run.path / "figures"
    epochs = [row["epoch"] for row in rows]
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
        axes[0].plot(epochs, [row["val_loss"] for row in rows], label="validation")
        axes[0].set_title("Classification loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].plot(epochs, [row.get("val_accuracy", np.nan) for row in rows], label="accuracy")
        axes[1].plot(epochs, [row.get("val_macro_f1", np.nan) for row in rows], label="macro-F1")
        axes[1].plot(epochs, [row.get("val_weighted_f1", np.nan) for row in rows], label="weighted-F1")
        axes[1].set_title("Validation metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=.25)
        axes[2].plot(epochs, [row["lr"] for row in rows], color="tab:orange")
        axes[2].set_title("Learning rate"); axes[2].set_xlabel("Epoch"); axes[2].set_yscale("log"); axes[2].grid(alpha=.25)
        fig.tight_layout(); fig.savefig(figures / "classification_training_curves.png", dpi=180); plt.close(fig)
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title="Classification confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(4):
        for j in range(4): ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout(); fig.savefig(figures / "classification_confusion_matrix.png", dpi=180); plt.close(fig)
    report = metrics["classification_report"]
    values = [report[name]["f1-score"] for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(CLASS_NAMES, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_ylim(0, 1); ax.set_ylabel("F1-score"); ax.set_title("Test F1-score by class"); plt.xticks(rotation=25, ha="right"); fig.tight_layout()
    fig.savefig(figures / "classification_test_f1_by_class.png", dpi=180); plt.close(fig)
    if true is not None and probabilities:
        from sklearn.metrics import precision_recall_curve, roc_curve
        probability = np.asarray(probabilities); labels_one_hot = np.eye(len(CLASS_NAMES))[np.asarray(true)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for class_id, name in enumerate(CLASS_NAMES):
            false_positive, true_positive, _ = roc_curve(labels_one_hot[:, class_id], probability[:, class_id])
            precision, recall, _ = precision_recall_curve(labels_one_hot[:, class_id], probability[:, class_id])
            axes[0].plot(false_positive, true_positive, label=name)
            axes[1].plot(recall, precision, label=name)
        confidence = probability.max(1); correct = probability.argmax(1) == np.asarray(true)
        bins = np.linspace(0, 1, 11); centers: list[float] = []; accuracy: list[float] = []; mean_confidence: list[float] = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            selected = (confidence > lower) & (confidence <= upper)
            if selected.any(): centers.append((lower + upper) / 2); accuracy.append(float(correct[selected].mean())); mean_confidence.append(float(confidence[selected].mean()))
        axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set(title="One-vs-rest ROC", xlabel="False-positive rate", ylabel="True-positive rate")
        axes[1].set(title="One-vs-rest precision-recall", xlabel="Recall", ylabel="Precision")
        axes[2].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration"); axes[2].plot(mean_confidence, accuracy, marker="o", label="model")
        axes[2].set(title="Confidence calibration", xlabel="Mean confidence", ylabel="Observed accuracy")
        for ax in axes: ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(figures / "classification_roc_pr_calibration.png", dpi=180); plt.close(fig)


def save_classification_predictions(run: ExperimentRun, model: nn.Module, loader: DataLoader, frame: pd.DataFrame, device: torch.device) -> None:
    records: list[dict[str, Any]] = []; offset = 0; model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device)); probabilities = torch.softmax(logits, dim=1).cpu().numpy(); predictions = probabilities.argmax(1)
            for j in range(len(labels)):
                row = frame.iloc[offset + j]
                record = {"path": row.path, "true_label": row.label, "predicted_label": CLASS_NAMES[int(predictions[j])], "correct": bool(int(labels[j]) == int(predictions[j])), "confidence": float(probabilities[j].max())}
                record.update({f"prob_{name}": float(probabilities[j, i]) for i, name in enumerate(CLASS_NAMES)})
                records.append(record)
            offset += len(labels)
    pd.DataFrame(records).to_csv(run.path / "classification_predictions.csv", index=False)
    (run.path / "classification_predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def save_classification_integrated_gradients(run: ExperimentRun, model: nn.Module, loader: DataLoader,
                                             frame: pd.DataFrame, device: torch.device, limit: int = 8) -> None:
    target_dir = run.path / "figures" / "classification_integrated_gradients"; target_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []; offset = 0; model.eval(); steps = 12
    for inputs, labels in loader:
        for index in range(len(inputs)):
            if len(records) >= limit:
                pd.DataFrame(records).to_csv(target_dir / "index.csv", index=False)
                return
            tensor = inputs[index:index + 1].to(device); baseline = torch.zeros_like(tensor)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), 1)[0]; predicted = int(probabilities.argmax())
            total_gradient = torch.zeros_like(tensor)
            for alpha in torch.linspace(1 / steps, 1, steps, device=device):
                sample = (baseline + alpha * (tensor - baseline)).detach().requires_grad_(True)
                gradient = torch.autograd.grad(model(sample)[0, predicted], sample)[0]
                total_gradient += gradient.detach()
            attribution = ((tensor - baseline) * total_gradient / steps).abs().sum(1)[0].cpu().numpy()
            attribution -= attribution.min(); attribution /= max(float(attribution.max()), 1e-12)
            image = tensor[0].detach().cpu().numpy() * np.asarray([.229, .224, .225])[:, None, None] + np.asarray([.485, .456, .406])[:, None, None]
            image = np.clip(np.transpose(image, (1, 2, 0)), 0, 1)
            row = frame.iloc[offset + index]
            fig, axes = plt.subplots(1, 2, figsize=(8, 4)); axes[0].imshow(image); axes[0].set_title(f"true={row.label}")
            axes[1].imshow(image); axes[1].imshow(attribution, cmap="inferno", alpha=.55); axes[1].set_title(f"pred={CLASS_NAMES[predicted]} ({float(probabilities[predicted]):.1%})")
            for axis in axes: axis.axis("off")
            fig.tight_layout(); filename = f"{len(records) + 1:03d}_{Path(row.path).stem}.png"; fig.savefig(target_dir / filename, dpi=180); plt.close(fig)
            records.append({"path": row.path, "true_label": row.label, "predicted_label": CLASS_NAMES[predicted],
                            "confidence": float(probabilities[predicted]), "figure": filename, "method": "Integrated Gradients", "steps": steps})
        offset += len(inputs)
    pd.DataFrame(records).to_csv(target_dir / "index.csv", index=False)


def yield_loss(disease: str, severity: float, config: dict[str, Any] | None = None) -> float:
    beta = {"early_blight": 0.6, "late_blight": 1.1, "bacterial_spot": 0.4}.get(disease, 0.0)
    if config is not None:
        return configured_yield_loss(config, disease, severity)
    return float(np.clip(beta * severity, 0.0, 100.0))


def save_segmentation_figures(run: ExperimentRun, rows: list[dict[str, Any]], metrics: dict[str, float], severity_rows: list[dict[str, Any]]) -> None:
    figures = run.path / "figures"
    epochs = [row["epoch"] for row in rows]
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
        axes[0].plot(epochs, [row["val_loss"] for row in rows], label="validation")
        axes[0].set_title("Segmentation loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].plot(epochs, [row.get("val_dice_macro", np.nan) for row in rows], label="macro-Dice")
        axes[1].plot(epochs, [row.get("val_iou_macro", np.nan) for row in rows], label="macro-IoU")
        axes[1].set_title("Validation overlap metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=.25)
        axes[2].plot(epochs, [row["lr"] for row in rows], color="tab:orange")
        axes[2].set_title("Learning rate"); axes[2].set_xlabel("Epoch"); axes[2].set_yscale("log"); axes[2].grid(alpha=.25)
        fig.tight_layout(); fig.savefig(figures / "segmentation_training_curves.png", dpi=180); plt.close(fig)
    names = SEGMENTATION_NAMES[1:]
    dice = [metrics[f"dice_{name}"] for name in names]
    iou = [metrics[f"iou_{name}"] for name in names]
    x = np.arange(3); fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(x-.18, dice, .36, label="Dice"); ax.bar(x+.18, iou, .36, label="IoU")
    ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Test segmentation scores by disease"); ax.legend(); plt.xticks(rotation=25, ha="right"); fig.tight_layout()
    fig.savefig(figures / "segmentation_test_scores_by_disease.png", dpi=180); plt.close(fig)
    if severity_rows:
        severity_frame = pd.DataFrame(severity_rows)
        severity_frame.to_csv(run.path / "severity_yield_loss.csv", index=False)
        (run.path / "severity_yield_loss.json").write_text(severity_frame.to_json(orient="records", indent=2), encoding="utf-8")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for name, group in severity_frame.groupby("true_disease"):
            axes[0].scatter(group["true_severity"], group["predicted_severity"], label=name, alpha=.75)
        axes[0].plot([0, 100], [0, 100], "k--", label="ideal")
        axes[0].set(xlabel="Ground-truth severity (%)", ylabel="Predicted severity (%)", title="Severity agreement"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
        grouped = severity_frame.groupby("predicted_disease")["predicted_yield_loss"].mean().reindex(names).fillna(0)
        axes[1].bar(names, grouped.values, color=["#f58518", "#54a24b", "#e45756"])
        axes[1].set(ylabel="Potential yield impact (%)", title="Mean test sensitivity scenario"); axes[1].tick_params(axis="x", rotation=25); axes[1].grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(figures / "severity_and_yield_loss.png", dpi=180); plt.close(fig)


def save_segmentation_overlays(run: ExperimentRun, model: nn.Module, loader: DataLoader, frame: pd.DataFrame,
                               device: torch.device, config: dict[str, Any], overlay_limit: int = 36) -> list[dict[str, Any]]:
    overlay_dir = run.path / "figures" / "segmentation_overlays"; overlay_dir.mkdir(exist_ok=True)
    mask_dir = run.path / "predictions" / "segmentation_masks"; mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []; offset = 0
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            logits = model(inputs.to(device)); predicted = logits.argmax(1).cpu().numpy(); target_np = target.numpy()
            inputs_np = inputs.numpy() * np.array([[[[0.229]], [[0.224]], [[0.225]],]]).reshape(1, 3, 1, 1) + np.array([[[[0.485]], [[0.456]], [[0.406]],]]).reshape(1, 3, 1, 1)
            for j in range(len(inputs)):
                manifest_row = frame.iloc[offset + j]
                image = np.clip(np.transpose(inputs_np[j], (1, 2, 0)), 0, 1)
                true_mask, pred_mask = target_np[j], predicted[j]
                with Image.open(manifest_row.leaf_mask_path) as leaf:
                    leaf_np = np.asarray(leaf.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.Resampling.NEAREST)) > 0
                true_disease = str(manifest_row.label)
                predicted_counts = [int(np.count_nonzero((pred_mask == class_id) & leaf_np)) for class_id in range(1, len(SEGMENTATION_NAMES))]
                predicted_class_id = int(np.argmax(predicted_counts)) + 1 if max(predicted_counts, default=0) > 0 else 0
                predicted_disease = SEGMENTATION_NAMES[predicted_class_id] if predicted_class_id else "healthy"
                true_severity = 100 * np.count_nonzero((true_mask > 0) & leaf_np) / max(np.count_nonzero(leaf_np), 1)
                pred_severity = 100 * predicted_counts[predicted_class_id - 1] / max(np.count_nonzero(leaf_np), 1) if predicted_class_id else 0.0
                yield_interval = yield_loss_interval(predicted_disease, pred_severity)
                sample = {"sample_id": manifest_row.sample_id, "true_disease": true_disease,
                          "predicted_disease": predicted_disease, "disease_correct": predicted_disease == true_disease,
                          "true_severity": float(true_severity), "predicted_severity": float(pred_severity),
                          "severity_error": float(pred_severity - true_severity),
                          "predicted_yield_loss": yield_interval["central"],
                          "predicted_yield_loss_low": yield_interval["low"],
                          "predicted_yield_loss_high": yield_interval["high"],
                          "true_lesion_pixels": int(np.count_nonzero(true_mask > 0)),
                          "predicted_lesion_pixels": int(np.count_nonzero(pred_mask > 0))}
                for class_id, name in enumerate(SEGMENTATION_NAMES[1:], 1):
                    pred_class, true_class = pred_mask == class_id, true_mask == class_id
                    tp = int(np.count_nonzero(pred_class & true_class)); fp = int(np.count_nonzero(pred_class & ~true_class)); fn = int(np.count_nonzero(~pred_class & true_class))
                    sample.update({f"{name}_tp": tp, f"{name}_fp": fp, f"{name}_fn": fn,
                                   f"{name}_dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
                                   f"{name}_iou": float(tp / max(tp + fp + fn, 1))})
                rows.append(sample)
                Image.fromarray(pred_mask.astype(np.uint8)).save(mask_dir / f"{manifest_row.sample_id}.png")
                if len(rows) <= overlay_limit:
                    save_segmentation_overlay_grid(
                        image, true_mask, pred_mask, true_disease, predicted_disease,
                        true_severity, pred_severity, yield_interval["central"],
                        overlay_dir / f"{len(rows):03d}_{manifest_row.sample_id}.png",
                    )
            offset += len(inputs)
    return rows


def save_checkpoint(run: ExperimentRun, model: nn.Module, optimizer: Any, epoch: int, metric: float, name: str,
                    scaler: Any | None = None, scheduler: Any | None = None, *,
                    best_metric: float | None = None, best_epoch: int | None = None,
                    patience: int = 0) -> None:
    payload = {
            "epoch": epoch,
            "metric": metric,
            "best_metric": metric if best_metric is None else best_metric,
            "best_epoch": epoch if best_epoch is None else best_epoch,
            "patience": patience,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    target = run.path / "checkpoints" / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Any, device: torch.device,
                    scaler: Any | None = None, scheduler: Any | None = None, *,
                    restore_rng: bool = True) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if restore_rng:
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
    return checkpoint


def accumulation_divisor(step: int, total_steps: int, accumulation: int) -> int:
    group_start = ((step - 1) // accumulation) * accumulation + 1
    return min(group_start + accumulation - 1, total_steps) - group_start + 1


def set_backbone_trainable(model: Classifier, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def class_weights(config: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if not bool(config["classification"].get("use_class_weights", True)):
        return None
    path = Path(config["data"]["root"]) / "classification_balance.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path).set_index("label")
    return torch.tensor([float(frame.loc[name, "inverse_frequency_weight"]) for name in CLASS_NAMES], dtype=torch.float32, device=device)


def configured_yield_loss(config: dict[str, Any], disease: str, severity: float) -> float:
    beta_key = f"{disease}_beta"
    beta = float(config["yield_loss"].get(beta_key, 0.0))
    return float(np.clip(beta * severity, float(config["yield_loss"].get("clamp_min", 0.0)), float(config["yield_loss"].get("clamp_max", 100.0))))


def split_label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        f"{split}/{label}": int(count)
        for (split, label), count in frame.groupby(["split", "label"]).size().items()
    }


def loader_pair(frame: pd.DataFrame, kind: str, size: int, batch: int, workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cls = ClassificationDataset if kind == "classification" else SegmentationDataset
    train = dataset_cls(frame[frame.split == "train"], size, True)
    val = dataset_cls(frame[frame.split == "val"], size, False)
    test = dataset_cls(frame[frame.split == "test"], size, False)
    kwargs = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train, batch_size=batch, shuffle=True, **kwargs),
        DataLoader(val, batch_size=batch, shuffle=False, **kwargs),
        DataLoader(test, batch_size=batch, shuffle=False, **kwargs),
    )


def run_classification(config: dict[str, Any], args: argparse.Namespace) -> Path:
    seed = int(config["project"]["seed"]); seed_everything(seed)
    task_name = str(config.get("runtime", {}).get("task", "classification"))
    frame = pd.read_csv(config["data"]["classification_manifest"])
    device = resolve_device(args.device)
    resume_path = Path(args.resume) if args.resume else None
    existing_path = resume_path.parent.parent if resume_path else getattr(args, "existing_run_path", None)
    run = ExperimentRun(Path(config["logging"]["output_root"]), task_name, config, existing_path)
    run.log(f"START task={task_name} device={device_info(device)} seed={seed}")
    run.write_json("dataset_snapshot.json", {"manifest": config["data"]["classification_manifest"], "rows": len(frame), "counts": split_label_counts(frame), "device": device_info(device)})
    train_loader, val_loader, test_loader = loader_pair(frame, "classification", int(config["data"]["image_size_classification"]), int(config["classification"]["batch_size"]), args.workers)
    model = Classifier(4, attention="attention" in config["classification"]["model"], pretrained=args.pretrained, allow_random_init=args.allow_random_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["classification"]["learning_rate"]), weight_decay=float(config["classification"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=config["classification"]["amp"] and device.type == "cuda")
    weights = class_weights(config, device)
    epochs = int(config["classification"]["epochs"] if args.epochs is None else args.epochs)
    accumulation = max(1, math.ceil(int(config["classification"].get("effective_batch_size", config["classification"]["batch_size"])) / int(config["classification"]["batch_size"])))
    rows: list[dict[str, Any]] = []; best = -math.inf; best_epoch = 0; patience = 0; start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), model, optimizer, device, scaler, scheduler)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best = float(checkpoint.get("best_metric", checkpoint.get("metric", -math.inf)))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        patience = int(checkpoint.get("patience", 0))
        rows = load_metrics_history(run, start_epoch)
        run.log(f"RESUME checkpoint={args.resume} start_epoch={start_epoch} best_epoch={best_epoch} patience={patience}")
        if not (run.path / "checkpoints" / "best.pt").exists():
            save_checkpoint(run, model, optimizer, start_epoch - 1, best, "best.pt", scaler, scheduler,
                            best_metric=best, best_epoch=best_epoch, patience=patience)
    def evaluate(loader: DataLoader) -> tuple[float, dict[str, Any], list[int], list[int], list[list[float]]]:
        model.eval(); total_loss = 0.0; true: list[int] = []; pred: list[int] = []; probabilities: list[list[float]] = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(inputs); loss = F.cross_entropy(logits, labels, weight=weights)
                batch_probability = torch.softmax(logits, dim=1).cpu()
                total_loss += loss.item() * len(labels); true.extend(labels.cpu().tolist()); pred.extend(batch_probability.argmax(1).tolist()); probabilities.extend(batch_probability.tolist())
        return total_loss / len(loader.dataset), classification_metrics(true, pred, probabilities), true, pred, probabilities
    for epoch in range(start_epoch, epochs + 1):
        started = time.time(); backbone_trainable = epoch > int(config["classification"].get("freeze_backbone_epochs", 0)); set_backbone_trainable(model, backbone_trainable); model.train()
        if not backbone_trainable:
            model.features.eval()
        train_loss = 0.0; train_true: list[int] = []; train_pred: list[int] = []
        optimizer.zero_grad(set_to_none=True)
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs); loss = F.cross_entropy(logits, labels, weight=weights)
            scaler.scale(loss / accumulation_divisor(step, len(train_loader), accumulation)).backward()
            if step % accumulation == 0 or step == len(train_loader): scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * len(labels); train_true.extend(labels.detach().cpu().tolist()); train_pred.extend(logits.detach().argmax(1).cpu().tolist())
        train_metrics = classification_metrics(train_true, train_pred); val_loss, val_metrics, _, _, _ = evaluate(val_loader); scheduler.step(val_metrics["macro_f1"])
        row = {"epoch": epoch, "train_loss": train_loss / len(train_loader.dataset), "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started, "accumulation_steps": accumulation, "backbone_trainable": backbone_trainable, **resource_snapshot(device), **{f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))}, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
        rows.append(row); save_metrics(run, rows); run.log(f"epoch={epoch} train_loss={row['train_loss']:.8f} train_f1={train_metrics['macro_f1']:.8f} val_loss={val_loss:.8f} val_f1={val_metrics['macro_f1']:.8f}")
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]; best_epoch = epoch; patience = 0
            save_checkpoint(run, model, optimizer, epoch, best, "best.pt", scaler, scheduler,
                            best_metric=best, best_epoch=best_epoch, patience=patience)
            run.write_json("best_validation.json", val_metrics)
        else: patience += 1
        save_checkpoint(run, model, optimizer, epoch, val_metrics["macro_f1"], "last.pt", scaler, scheduler,
                        best_metric=best, best_epoch=best_epoch, patience=patience)
        print(f"[classification] epoch {epoch}/{epochs} val_macro_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}")
        if patience >= int(config["classification"]["early_stopping_patience"]): break
    if not (run.path / "checkpoints" / "best.pt").exists(): raise RuntimeError("No best checkpoint was produced; epochs must be >= 1")
    load_checkpoint(run.path / "checkpoints" / "best.pt", model, optimizer, device, scaler, scheduler, restore_rng=False)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    test_loss, test_metrics, test_true, _, test_probabilities = evaluate(test_loader); test_metrics["loss"] = test_loss
    run.write_json("test_metrics.json", test_metrics); pd.DataFrame.from_dict(test_metrics["per_class"], orient="index").rename_axis("class").reset_index().to_csv(run.path / "classification_per_class_metrics.csv", index=False)
    save_classification_predictions(run, model, test_loader, frame[frame.split == "test"].reset_index(drop=True), device); save_classification_figures(run, rows, test_metrics, test_true, test_probabilities)
    if task_name == "classification":
        save_classification_integrated_gradients(run, model, test_loader, frame[frame.split == "test"].reset_index(drop=True), device)
    model_profile = profile_model(model, (1, 3, int(config["data"]["image_size_classification"]), int(config["data"]["image_size_classification"])), device, run.path / "checkpoints" / "best.pt")
    run.write_json("model_profile.json", model_profile)
    summary = {"task": task_name, "model": config["classification"]["model"], "best_val_macro_f1": best, "best_epoch": best_epoch, "test": test_metrics, "model_profile": model_profile, "epochs_completed": len(rows), "device": device_info(device)}; run.write_json("summary.json", summary); run.log(f"END task={task_name} test_accuracy={test_metrics['accuracy']:.8f} test_macro_f1={test_metrics['macro_f1']:.8f}"); run.complete(summary)
    run.save_text("REPORT.md", f"# Classification run\n\nBest validation macro-F1: **{best:.4f}**\n\nTest accuracy: **{test_metrics['accuracy']:.4f}**\n\nTest macro-F1: **{test_metrics['macro_f1']:.4f}**\n\nClass weights, TP/FP/TN/FN and all per-class metrics are in `classification_per_class_metrics.csv` and `test_metrics.json`.\n")
    return run.path


def run_segmentation(config: dict[str, Any], args: argparse.Namespace) -> Path:
    seed = int(config["project"]["seed"]); seed_everything(seed); frame = pd.read_csv(config["data"]["segmentation_manifest"]); device = resolve_device(args.device)
    task_name = str(config.get("runtime", {}).get("task", "segmentation"))
    resume_path = Path(args.resume) if args.resume else None
    existing_path = resume_path.parent.parent if resume_path else getattr(args, "existing_run_path", None)
    run = ExperimentRun(Path(config["logging"]["output_root"]), task_name, config, existing_path); run.log(f"START task={task_name} device={device_info(device)} seed={seed}")
    run.write_json("dataset_snapshot.json", {"manifest": config["data"]["segmentation_manifest"], "rows": len(frame), "counts": split_label_counts(frame), "device": device_info(device)})
    train_loader, val_loader, test_loader = loader_pair(frame, "segmentation", int(config["data"]["image_size_segmentation"]), int(config["segmentation"]["batch_size"]), args.workers)
    if config["segmentation"].get("model") == "mcunet_segmentation":
        width_mult = args.mcunet_width if args.mcunet_width is not None else config["segmentation"].get("width_mult", 0.5)
        model = MCUNetSegmentation(num_classes=4, width_mult=float(width_mult)).to(device)
    else:
        model = AttentionResidualUNet(num_classes=4, base=args.unet_base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["segmentation"]["learning_rate"]), weight_decay=float(config["segmentation"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.3); scaler = torch.amp.GradScaler("cuda", enabled=config["segmentation"]["amp"] and device.type == "cuda")
    epochs = int(config["segmentation"]["epochs"] if args.epochs is None else args.epochs); accumulation = max(1, math.ceil(int(config["segmentation"].get("effective_batch_size", config["segmentation"]["batch_size"])) / int(config["segmentation"]["batch_size"])))
    ce_weight = None; rows: list[dict[str, Any]] = []; best = -math.inf; best_epoch = 0; patience = 0; start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), model, optimizer, device, scaler, scheduler)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best = float(checkpoint.get("best_metric", checkpoint.get("metric", -math.inf)))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        patience = int(checkpoint.get("patience", 0))
        rows = load_metrics_history(run, start_epoch)
        run.log(f"RESUME checkpoint={args.resume} start_epoch={start_epoch} best_epoch={best_epoch} patience={patience}")
        if not (run.path / "checkpoints" / "best.pt").exists():
            save_checkpoint(run, model, optimizer, start_epoch - 1, best, "best.pt", scaler, scheduler,
                            best_metric=best, best_epoch=best_epoch, patience=patience)
    ce_ratio = float(config["segmentation"].get("cross_entropy_weight", 0.5)); dice_ratio = float(config["segmentation"].get("dice_weight", 0.5))
    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ce_ratio * F.cross_entropy(logits, target, weight=ce_weight) + dice_ratio * dice_loss(logits, target, 4)
    def aggregate_counts(target_metrics: dict[str, float], aggregate: dict[str, float]) -> None:
        for class_name in SEGMENTATION_NAMES:
            for suffix in ("tp", "fp", "tn", "fn"):
                key = f"{class_name}_{suffix}"; aggregate[key] = aggregate.get(key, 0.0) + target_metrics[key]
    def evaluate(loader: DataLoader, advanced: bool = False) -> tuple[float, dict[str, float]]:
        model.eval(); total_loss = 0.0; aggregate: dict[str, float] = {}; advanced_values: dict[str, list[float]] = {}
        with torch.no_grad():
            for inputs, target in loader:
                inputs, target = inputs.to(device), target.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(inputs); loss = loss_fn(logits, target)
                total_loss += loss.item() * len(inputs); aggregate_counts(segmentation_metrics(logits, target, 4), aggregate)
                if advanced:
                    for key, value in segmentation_boundary_metrics(logits, target, 4).items():
                        advanced_values.setdefault(key, []).extend([value] * len(inputs))
        metrics = metrics_from_segmentation_counts(aggregate, 4)
        metrics.update({key: float(np.mean(values)) for key, values in advanced_values.items()})
        return total_loss / len(loader.dataset), metrics
    for epoch in range(start_epoch, epochs + 1):
        started = time.time(); model.train(); train_loss = 0.0; aggregate: dict[str, float] = {}; optimizer.zero_grad(set_to_none=True)
        for step, (inputs, target) in enumerate(train_loader, 1):
            inputs, target = inputs.to(device), target.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs); loss = loss_fn(logits, target)
            scaler.scale(loss / accumulation_divisor(step, len(train_loader), accumulation)).backward()
            if step % accumulation == 0 or step == len(train_loader): scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * len(inputs); aggregate_counts(segmentation_metrics(logits.detach(), target, 4), aggregate)
        train_metrics = metrics_from_segmentation_counts(aggregate, 4); val_loss, val_metrics = evaluate(val_loader); scheduler.step(val_metrics["dice_macro"])
        row = {"epoch": epoch, "train_loss": train_loss / len(train_loader.dataset), "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started, "accumulation_steps": accumulation, **resource_snapshot(device), **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row); save_metrics(run, rows); run.log(f"epoch={epoch} train_loss={row['train_loss']:.8f} train_dice={train_metrics['dice_macro']:.8f} val_loss={val_loss:.8f} val_dice={val_metrics['dice_macro']:.8f}")
        if val_metrics["dice_macro"] > best:
            best = val_metrics["dice_macro"]; best_epoch = epoch; patience = 0
            save_checkpoint(run, model, optimizer, epoch, best, "best.pt", scaler, scheduler,
                            best_metric=best, best_epoch=best_epoch, patience=patience)
            run.write_json("best_validation.json", val_metrics)
        else: patience += 1
        save_checkpoint(run, model, optimizer, epoch, val_metrics["dice_macro"], "last.pt", scaler, scheduler,
                        best_metric=best, best_epoch=best_epoch, patience=patience)
        print(f"[segmentation] epoch {epoch}/{epochs} val_dice={val_metrics['dice_macro']:.4f} val_iou={val_metrics['iou_macro']:.4f}")
        if patience >= int(config["segmentation"]["early_stopping_patience"]): break
    if not (run.path / "checkpoints" / "best.pt").exists(): raise RuntimeError("No best checkpoint was produced; epochs must be >= 1")
    load_checkpoint(run.path / "checkpoints" / "best.pt", model, optimizer, device, scaler, scheduler, restore_rng=False)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    test_loss, test_metrics = evaluate(test_loader, advanced=True); test_metrics["loss"] = test_loss; run.write_json("test_metrics.json", test_metrics)
    segmentation_table = [{"class": name, **{key: test_metrics[f"{name}_{key}"] for key in ("tp", "fp", "tn", "fn", "support", "precision", "recall", "specificity", "npv", "balanced_accuracy", "mcc", "f1", "dice", "iou")}} for name in SEGMENTATION_NAMES]
    pd.DataFrame(segmentation_table).to_csv(run.path / "segmentation_per_class_metrics.csv", index=False)
    severity_rows = save_segmentation_overlays(run, model, test_loader, frame[frame.split == "test"].reset_index(drop=True), device, config); save_segmentation_figures(run, rows, test_metrics, severity_rows)
    save_yield_sensitivity(run.path / "yield_sensitivity.csv", run.path / "figures" / "yield_beta_sensitivity.png")
    model_profile = profile_model(model, (1, 3, int(config["data"]["image_size_segmentation"]), int(config["data"]["image_size_segmentation"])), device, run.path / "checkpoints" / "best.pt")
    run.write_json("model_profile.json", model_profile)
    summary = {"task": task_name, "model": config["segmentation"]["model"], "best_val_dice": best, "best_epoch": best_epoch, "test": test_metrics, "model_profile": model_profile, "epochs_completed": len(rows), "device": device_info(device), "severity_rows": len(severity_rows)}; run.write_json("summary.json", summary); run.log(f"END task={task_name} test_dice={test_metrics['dice_macro']:.8f} test_iou={test_metrics['iou_macro']:.8f} samples={len(severity_rows)}"); run.complete(summary)
    model_label = str(config["segmentation"].get("model", "segmentation"))
    run.save_text("REPORT.md", f"# Segmentation run\n\nModel: **{model_label}**\n\nBest validation macro-Dice: **{best:.4f}**\n\nTest macro-Dice: **{test_metrics['dice_macro']:.4f}**\n\nAll test samples have per-sample severity, yield-loss, pixel counts and predicted masks in `severity_yield_loss.csv` and `predictions/segmentation_masks/`.\n")
    return run.path


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def incomplete_checkpoint(root: Path, task: str) -> Path | None:
    candidates: list[Path] = []
    for status_path in root.rglob("status.json"):
        if status_path.name == "latest.json":
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_path = status_path.parent
        recorded_task = str(status.get("task", run_path.name))
        if recorded_task != task or status.get("status") not in {"running", "failed"}:
            continue
        checkpoint = run_path / "checkpoints" / "last.pt"
        if checkpoint.exists():
            candidates.append(checkpoint)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_group_for_checkpoint(root: Path, checkpoint: Path | None) -> str | None:
    if checkpoint is None:
        return None
    run_path = checkpoint.parent.parent
    if run_path.parent != root and run_path.parent.parent == root:
        return run_path.parent.name
    return None


def successful_task_in_group(root: Path, group: str | None, task: str) -> bool:
    if not group:
        return False
    status_path = root / group / task / "status.json"
    if not status_path.exists():
        return False
    try:
        return json.loads(status_path.read_text(encoding="utf-8")).get("status") == "success"
    except (OSError, json.JSONDecodeError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def session_path(root: Path, group: str) -> Path:
    return root / group / "session.json"


def ensure_session(root: Path, group: str, tasks: list[str]) -> dict[str, Any]:
    path = session_path(root, group)
    session = _read_json(path)
    task_states = session.setdefault("tasks", {})
    for task in tasks:
        task_states.setdefault(task, {"status": "pending"})
    session.update({"run_group": group, "status": "running", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _write_json_atomic(path, session)
    return session


def set_session_task(root: Path, group: str, task: str, status: str,
                     run_path: Path | None = None, error: BaseException | None = None) -> None:
    path = session_path(root, group)
    session = _read_json(path) or {"run_group": group, "tasks": {}}
    task_payload: dict[str, Any] = {"status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if run_path is not None:
        task_payload["path"] = str(run_path)
    if error is not None:
        task_payload["error"] = repr(error)
    session.setdefault("tasks", {})[task] = task_payload
    states = [entry.get("status") for entry in session["tasks"].values()]
    session["status"] = "finalizing" if states and all(state == "success" for state in states) else "running"
    if status == "failed":
        session["status"] = "failed"
    session["updated_at"] = task_payload["updated_at"]
    _write_json_atomic(path, session)


def latest_incomplete_session(root: Path, tasks: list[str]) -> str | None:
    candidates: list[Path] = []
    if not root.exists():
        return None
    for path in root.glob("overnight_*/session.json"):
        session = _read_json(path)
        if session.get("status") == "success":
            continue
        task_states = session.get("tasks", {})
        if session.get("status") == "finalizing" or any(task_states.get(task, {}).get("status") != "success" for task in tasks):
            candidates.append(path)
    return max(candidates, key=lambda item: item.stat().st_mtime).parent.name if candidates else None


def checkpoint_in_group(root: Path, group: str, task: str) -> Path | None:
    run_path = root / group / task
    if successful_task_in_group(root, group, task):
        return None
    checkpoint = run_path / "checkpoints" / "last.pt"
    return checkpoint if checkpoint.exists() else None


def save_session_summary(root: Path, group: str) -> None:
    group_path = root / group
    summaries: dict[str, Any] = {}
    for task_path in group_path.iterdir():
        summary_path = task_path / "summary.json" if task_path.is_dir() else None
        if summary_path and summary_path.exists():
            summaries[task_path.name] = _read_json(summary_path)
    _write_json_atomic(group_path / "session_summary.json", {"run_group": group, "tasks": summaries})
    def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict): result.update(flatten(item, name))
            elif isinstance(item, (str, int, float, bool)) or item is None: result[name] = item
        return result
    pd.DataFrame([{"task": task, **flatten(summary)} for task, summary in summaries.items()]).to_csv(group_path / "all_metrics_flat.csv", index=False)
    baseline = summaries.get("classification_baseline"); attention = summaries.get("classification")
    if baseline and attention:
        rows = []
        for name, summary in [("DenseNet-121 baseline", baseline), ("DenseNet-121 + residual attention", attention)]:
            rows.append({"model": name, "accuracy": summary["test"]["accuracy"], "macro_f1": summary["test"]["macro_f1"],
                         "balanced_accuracy": summary["test"]["balanced_accuracy"], "mcc": summary["test"]["matthews_correlation_coefficient"],
                         "parameters": summary["model_profile"]["parameters_total"], "latency_ms": summary["model_profile"]["latency_ms_per_batch"]})
        comparison = pd.DataFrame(rows); comparison.to_csv(group_path / "classification_ablation.csv", index=False)
        save_classification_ablation(comparison, group_path / "classification_ablation.png")
    segmentation_baseline = summaries.get("segmentation_baseline"); segmentation_main = summaries.get("segmentation")
    if segmentation_baseline and segmentation_main:
        rows = []
        for name, summary in [("Attention-Residual U-Net", segmentation_baseline), ("MCUNet-Seg", segmentation_main)]:
            rows.append({"model": name, "dice_macro": summary["test"]["dice_macro"], "iou_macro": summary["test"]["iou_macro"],
                         "boundary_iou_macro": summary["test"]["boundary_iou_macro"], "parameters": summary["model_profile"]["parameters_total"],
                         "latency_ms": summary["model_profile"]["latency_ms_per_batch"]})
        comparison = pd.DataFrame(rows); comparison.to_csv(group_path / "segmentation_ablation.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(comparison)); width = .25
        for index, key in enumerate(["dice_macro", "iou_macro", "boundary_iou_macro"]):
            ax.bar(x + (index - 1) * width, comparison[key], width, label=key)
        ax.set_xticks(x, comparison["model"]); ax.set_ylim(0, 1); ax.set_title("Segmentation ablation on the held-out test split"); ax.legend(); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(group_path / "segmentation_ablation.png", dpi=200); plt.close(fig)
    lines = [f"# Overnight session {group}", "", "## Quick result table", "",
             "| Task | Model | Headline metric | Accuracy/IoU | AUC/Boundary IoU | Parameters | Latency (ms) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task, summary in summaries.items():
        test = summary.get("test", {})
        headline = test.get("macro_f1", test.get("dice_macro", "n/a"))
        secondary = test.get("accuracy", test.get("iou_macro", "n/a"))
        tertiary = test.get("roc_auc_ovr_macro", test.get("boundary_iou_macro", "n/a"))
        profile = summary.get("model_profile", {})
        lines.append(f"| {task} | {summary.get('model', '')} | {headline} | {secondary} | {tertiary} | {profile.get('parameters_total', '')} | {profile.get('latency_ms_per_batch', '')} |")
    lines.extend(["", "## Report-ready outputs", "", "- `all_metrics_flat.csv`: every scalar from every task in one table.",
                  "- `classification_ablation.csv/.png` and `segmentation_ablation.csv/.png`: fair held-out comparisons.",
                  "- `figures/`: workflow, model architecture, dataset distribution and examples.",
                  "- Each task directory: checkpoints, epoch traces, detailed metrics, predictions and figures.", "",
                  "## Scientific interpretation", "", "Yield values are low/central/high potential-impact sensitivity scenarios. They are not field-calibrated farm-yield forecasts.", ""])
    (group_path / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classification", "segmentation", "all"], default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override for smoke tests or short runs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--unet-base", type=int, default=16, help="16 fits comfortably in 6 GB VRAM")
    parser.add_argument("--mcunet-width", type=float, default=None, help="Override MCUNet encoder width multiplier from config")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-random-init", action="store_true", help="Explicitly allow random DenseNet initialization if ImageNet weights cannot load")
    parser.add_argument("--resume", type=Path, default=None, help="Resume one task from a checkpoint; use with --task classification or segmentation")
    parser.add_argument("--no-auto-resume", action="store_true", help="Start a new run even when an interrupted run is detected")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.resume and args.task == "all":
        raise SystemExit("--resume must be used with one task, not --task all")
    config = copy.deepcopy(load_config(args.config))
    config["runtime"] = {
        "cli_task": args.task, "cli_epochs": args.epochs, "cli_workers": args.workers,
        "cli_unet_base": args.unet_base, "cli_mcunet_width": args.mcunet_width, "cli_pretrained": args.pretrained,
        "cli_allow_random_init": args.allow_random_init, "cli_device": args.device,
        "cli_resume": str(args.resume) if args.resume else None,
        "auto_resume": not args.no_auto_resume,
    }
    seed_everything(int(config["project"]["seed"]))
    gate_errors = validate_root(Path(config["data"]["root"]))
    if gate_errors:
        raise SystemExit("DATA GATE: FAIL\n" + "\n".join(f"- {error}" for error in gate_errors))
    print("DATA GATE: PASS (automatic preflight)")
    output_root = Path(config["logging"]["output_root"])
    if args.task == "all":
        tasks = []
        if bool(config.get("experiments", {}).get("run_classification_baseline", True)):
            tasks.append("classification_baseline")
        tasks.append("classification")
        if bool(config.get("experiments", {}).get("run_segmentation_baseline", True)):
            tasks.append("segmentation_baseline")
        tasks.append("segmentation")
    else:
        tasks = [args.task]
    legacy_checkpoints = {
        task: (args.resume if task == args.task and args.resume else incomplete_checkpoint(output_root, task))
        for task in tasks
    } if not args.no_auto_resume or args.resume else {task: None for task in tasks}
    if args.resume:
        run_group = run_group_for_checkpoint(output_root, args.resume)
    else:
        candidates = [checkpoint for checkpoint in legacy_checkpoints.values() if checkpoint is not None]
        newest = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
        checkpoint_group = run_group_for_checkpoint(output_root, newest)
        pending_group = latest_incomplete_session(output_root, tasks) if not args.no_auto_resume else None
        pending_path = session_path(output_root, pending_group) if pending_group else None
        if pending_path is not None and (newest is None or pending_path.stat().st_mtime >= newest.stat().st_mtime):
            run_group = pending_group
        else:
            run_group = checkpoint_group
    resumed_session = run_group is not None
    run_group = run_group or f"overnight_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    config["runtime"]["run_group"] = run_group
    ensure_session(output_root, run_group, tasks)
    group_path = output_root / run_group
    (group_path / "figures").mkdir(exist_ok=True)
    if not (group_path / "figures" / "workflow.png").exists():
        save_workflow_figure(group_path / "figures" / "workflow.png")
        save_dataset_distribution(Path(config["data"]["classification_manifest"]), Path(config["data"]["segmentation_manifest"]), group_path / "figures" / "dataset_distribution.png")
        save_architecture_figure(group_path / "figures" / "model_architectures.png")
        save_dataset_examples(Path(config["data"]["classification_manifest"]), Path(config["data"]["segmentation_manifest"]), group_path / "figures" / "dataset_examples.png")
    checkpoints = {
        task: (args.resume if args.resume and task == args.task else checkpoint_in_group(output_root, run_group, task))
        for task in tasks
    }
    if resumed_session:
        print(f"AUTO-RESUME: detected incomplete session {run_group}")
    console_handle = (group_path / "console.log").open("a", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(original_stdout, console_handle)
    sys.stderr = TeeStream(original_stderr, console_handle)
    try:
        for task in tasks:
            session = _read_json(session_path(output_root, run_group))
            session_success = session.get("tasks", {}).get(task, {}).get("status") == "success"
            if (session_success or successful_task_in_group(output_root, run_group, task)) and checkpoints.get(task) is None:
                set_session_task(output_root, run_group, task, "success", output_root / run_group / task)
                print(f"AUTO-RESUME: {task} already successful; skipping")
                continue
            task_args = copy.copy(args)
            task_args.resume = checkpoints.get(task)
            task_args.existing_run_path = output_root / run_group / task if (output_root / run_group / task).exists() else None
            task_config = copy.deepcopy(config)
            task_config["runtime"]["task"] = task
            if task == "classification_baseline":
                task_config["classification"]["model"] = "densenet121_baseline"
            if task == "segmentation_baseline":
                task_config["segmentation"]["model"] = "attention_residual_unet"
            task_config["runtime"]["cli_resume"] = str(task_args.resume) if task_args.resume else None
            runner = run_classification if task.startswith("classification") else run_segmentation
            set_session_task(output_root, run_group, task, "running", output_root / run_group / task)
            try:
                completed_path = runner(task_config, task_args)
            except BaseException as error:
                set_session_task(output_root, run_group, task, "failed", output_root / run_group / task, error)
                raise
            set_session_task(output_root, run_group, task, "success", completed_path)
            print(f"Saved {task} run to {completed_path}")
        session = _read_json(session_path(output_root, run_group)); session["status"] = "finalizing"
        session["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json_atomic(session_path(output_root, run_group), session)
        save_session_summary(output_root, run_group)
        session["status"] = "success"; session["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json_atomic(session_path(output_root, run_group), session)
    except BaseException as error:
        mark_active_failure(error)
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        console_handle.close()


if __name__ == "__main__":
    main()
