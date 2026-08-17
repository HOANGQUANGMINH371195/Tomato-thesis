"""Reusable figures and model/resource profiles for report-ready runs."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from .yield_model import YIELD_PROFILES, calculate_yield_loss, metadata


def save_workflow_figure(target: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Tomato disease analysis workflow", fontsize=18, fontweight="bold", pad=16)

    boxes = {
        "input": (.09, .50, .14, .15, "Leaf image", "#d8f3dc"),
        "classifier": (.31, .72, .18, .18, "DenseNet-121\nclassifier", "#bde0fe"),
        "disease": (.54, .72, .18, .18, "Disease class\n+ confidence", "#caf0f8"),
        "segmenter": (.31, .28, .18, .18, "Segmentation\nmodel", "#ffd6a5"),
        "mask": (.54, .28, .18, .18, "Lesion mask", "#ffcad4"),
        "severity": (.72, .28, .10, .16, "Severity\n(%)", "#ffe5a5"),
        "impact": (.90, .50, .18, .20, "Low / central / high\npotential yield impact", "#e4c1f9"),
    }
    patches: dict[str, FancyBboxPatch] = {}
    for name, (x, y, width, height, label, color) in boxes.items():
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.012", facecolor=color, edgecolor="#334155",
            linewidth=1.6, transform=ax.transAxes,
        )
        ax.add_patch(patch)
        patches[name] = patch
        ax.text(x, y, label, ha="center", va="center", fontsize=11.5,
                fontweight="bold", transform=ax.transAxes)

    connections = [
        ("input", "classifier"), ("input", "segmenter"),
        ("classifier", "disease"), ("segmenter", "mask"),
        ("mask", "severity"), ("disease", "impact"),
        ("severity", "impact"),
    ]
    for source, destination in connections:
        start = boxes[source][:2]
        end = boxes[destination][:2]
        ax.add_patch(FancyArrowPatch(
            start, end, patchA=patches[source], patchB=patches[destination],
            arrowstyle="-|>", mutation_scale=14, linewidth=2.0,
            color="#475569", shrinkA=2, shrinkB=2,
            connectionstyle="arc3,rad=0", transform=ax.transAxes,
        ))

    ax.text(.42, .90, "Classification branch", ha="center", fontsize=12, color="#1d4ed8")
    ax.text(.48, .08, "Segmentation and severity branch", ha="center", fontsize=12,
            color="#b45309")
    ax.text(.88, .72, "Sensitivity calculation", ha="center", fontsize=12,
            color="#7e22ce")
    fig.tight_layout(); fig.savefig(target, dpi=240, bbox_inches="tight", facecolor="white"); plt.close(fig)


def save_classification_ablation(comparison: pd.DataFrame, target: Path) -> None:
    """Show the attention model's score change relative to the plain baseline."""
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
               ("balanced_accuracy", "Balanced accuracy")]
    baseline = comparison.iloc[0]
    attention = comparison.iloc[1]
    changes = [100.0 * (float(attention[key]) - float(baseline[key])) for key, _ in metrics]
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.barh(y, changes, color="#dc2626", height=.52)
    ax.set_yticks(y, [label for _, label in metrics], fontsize=12)
    ax.set_xlim(-1.20, .15)
    ax.set_xlabel("Change from DenseNet-121 baseline (percentage points)\nPositive = improvement; negative = decrease",
                  fontsize=12)
    ax.set_title("Effect of adding the residual-attention head", fontsize=16,
                 fontweight="bold")
    ax.axvline(0.0, color="#334155", linewidth=1.5)
    ax.grid(axis="x", alpha=.25)
    for bar, value in zip(bars, changes):
        ax.text(value + .03, bar.get_y() + bar.get_height() / 2, f"{value:+.2f} pp",
                ha="left", va="center", color="white", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(target, dpi=240, bbox_inches="tight", facecolor="white"); plt.close(fig)


def save_segmentation_overlay_grid(
    image: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    true_disease: str,
    predicted_disease: str,
    true_severity: float,
    pred_severity: float,
    central_yield_impact: float,
    target: Path,
) -> None:
    """Render a readable 2-by-2 qualitative segmentation comparison."""
    cmap = ListedColormap(["black", "#f58518", "#54a24b", "#e45756"])
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.8), constrained_layout=True)
    panels = axes.ravel()
    true_name = true_disease.replace("_", " ").title()
    predicted_name = predicted_disease.replace("_", " ").title()
    panels[0].imshow(image)
    panels[0].set_title(f"Input image\nTrue class: {true_name}", fontsize=12)
    panels[1].imshow(true_mask, cmap=cmap, vmin=0, vmax=3)
    panels[1].set_title(f"Ground-truth mask\nSeverity: {true_severity:.1f}%", fontsize=12)
    panels[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=3)
    panels[2].set_title(f"Predicted mask\nSeverity: {pred_severity:.1f}%", fontsize=12)
    panels[3].imshow(image)
    panels[3].imshow(pred_mask > 0, cmap="Reds", alpha=.45)
    panels[3].set_title(
        f"Prediction overlay\nClass: {predicted_name}\nImpact scenario: {central_yield_impact:.1f}%",
        fontsize=12,
    )
    for axis in panels:
        axis.axis("off")
    fig.savefig(target, dpi=200, facecolor="white")
    plt.close(fig)


def save_dataset_distribution(classification_manifest: Path, segmentation_manifest: Path, target: Path) -> None:
    classification = pd.read_csv(classification_manifest)
    segmentation = pd.read_csv(segmentation_manifest)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, frame, title in [(axes[0], classification, "Classification split distribution"),
                             (axes[1], segmentation, "Segmentation split distribution")]:
        table = frame.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "val", "test"])
        table.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
        ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("Samples"); ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(target, dpi=200); plt.close(fig)


def save_architecture_figure(target: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8))
    stages = [
        (axes[0], ["224×224\nRGB image", "DenseNet-121\nfeatures", "Residual\nattention",
                   "Global\npooling", "Four-class\nsoftmax"], "Classification architecture", "#dbeafe"),
        (axes[1], ["256×256\nRGB image", "Inverted-residual\nencoder", "Multi-scale\nskip features",
                   "U-Net\ndecoder", "Four-class\nmask"], "Segmentation architecture", "#ffedd5"),
    ]
    centers = np.linspace(.10, .90, 5)
    box_width = .15
    box_height = .34
    for ax, labels, title, color in stages:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=8)
        patches: list[FancyBboxPatch] = []
        for x, label in zip(centers, labels):
            patch = FancyBboxPatch(
                (x - box_width / 2, .48 - box_height / 2), box_width, box_height,
                boxstyle="round,pad=0.01", facecolor=color, edgecolor="#334155",
                linewidth=1.6, transform=ax.transAxes,
            )
            ax.add_patch(patch)
            patches.append(patch)
            ax.text(x, .48, label, ha="center", va="center", fontsize=11.5,
                    fontweight="bold", transform=ax.transAxes)
        for index in range(len(centers) - 1):
            ax.add_patch(FancyArrowPatch(
                (centers[index], .48), (centers[index + 1], .48),
                patchA=patches[index], patchB=patches[index + 1],
                arrowstyle="-|>", mutation_scale=14, linewidth=2.0,
                color="#475569", shrinkA=2, shrinkB=2,
                connectionstyle="arc3,rad=0", transform=ax.transAxes,
            ))
    fig.tight_layout(pad=1.5)
    fig.savefig(target, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dataset_examples(classification_manifest: Path, segmentation_manifest: Path, target: Path) -> None:
    segmentation = pd.read_csv(segmentation_manifest)
    rows = segmentation.groupby("label", sort=True).first().reset_index()
    preferred_order = ["bacterial_spot", "early_blight", "late_blight"]
    rows = rows.set_index("label").loc[preferred_order].reset_index()
    display_names = {
        "bacterial_spot": "Bacterial Spot",
        "early_blight": "Early Blight",
        "late_blight": "Late Blight",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    for column, row in rows.iterrows():
        with Image.open(row.image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"))
        with Image.open(row.mask_path) as mask_file:
            lesion = np.asarray(mask_file) > 0
        axes[0, column].imshow(image)
        axes[0, column].set_title(display_names[str(row.label)], fontsize=15,
                                  fontweight="bold", pad=10)
        axes[1, column].imshow(image)
        axes[1, column].imshow(np.ma.masked_where(~lesion, lesion), cmap="Reds",
                               alpha=.62, vmin=0, vmax=1)
        for axis in axes[:, column]:
            axis.axis("off")

    axes[0, 0].text(-.08, .5, "Input image", transform=axes[0, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=14,
                    fontweight="bold", color="#334155")
    axes[1, 0].text(-.08, .5, "Lesion annotation", transform=axes[1, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=14,
                    fontweight="bold", color="#334155")
    fig.suptitle("Representative segmentation samples", fontsize=19,
                 fontweight="bold")
    fig.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_yield_sensitivity(target_csv: Path, target_figure: Path) -> None:
    rows: list[dict[str, Any]] = []
    severity = np.linspace(0, 100, 101)
    fig, ax = plt.subplots(figsize=(10, 6))
    for disease, profile in YIELD_PROFILES.items():
        central = [calculate_yield_loss(disease, value, profile["beta"]) for value in severity]
        low = [calculate_yield_loss(disease, value, profile["beta_min"]) for value in severity]
        high = [calculate_yield_loss(disease, value, profile["beta_max"]) for value in severity]
        ax.plot(severity, central, label=f"{profile['display_name']} β={profile['beta']}")
        ax.fill_between(severity, low, high, alpha=.14)
        rows.extend({"disease": disease, "severity": float(s), "beta_min_loss": float(lo),
                     "beta_central_loss": float(mid), "beta_max_loss": float(hi)}
                    for s, lo, mid, hi in zip(severity, low, central, high))
    ax.set(xlabel="Severity proxy (%)", ylabel="Potential yield impact (%)",
           title="Disease-specific β sensitivity scenarios", xlim=(0, 100), ylim=(0, 100))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(target_figure, dpi=200); plt.close(fig)
    pd.DataFrame(rows).to_csv(target_csv, index=False)
    target_csv.with_suffix(".metadata.json").write_text(json.dumps(metadata(), indent=2), encoding="utf-8")


def resource_snapshot(device: torch.device) -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    values = {"process_max_rss_kb": float(usage.ru_maxrss), "system_load_1m": float(os.getloadavg()[0])}
    if device.type == "cuda":
        values.update({
            "cuda_allocated_bytes": float(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": float(torch.cuda.memory_reserved(device)),
            "cuda_peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
        })
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits", "--id=0"], text=True, stderr=subprocess.DEVNULL,
            ).strip().split(",")
            if len(output) == 4:
                values.update({"gpu_utilization_percent": float(output[0]), "gpu_temperature_c": float(output[1]),
                               "gpu_memory_used_mib": float(output[2]), "gpu_power_w": float(output[3])})
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    try:
        memory = {line.split(":", 1)[0]: float(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
        values.update({"system_memory_total_kb": memory.get("MemTotal", 0.0), "system_memory_available_kb": memory.get("MemAvailable", 0.0)})
    except (OSError, ValueError, IndexError):
        pass
    return values


def profile_model(model: nn.Module, shape: tuple[int, ...], device: torch.device,
                  checkpoint: Path | None = None) -> dict[str, Any]:
    macs = 0
    hooks: list[Any] = []
    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += int(output.numel() * module.kernel_size[0] * module.kernel_size[1] * module.in_channels / module.groups)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    sample = torch.zeros(shape, device=device); model.eval()
    with torch.no_grad():
        model(sample)
    for item in hooks: item.remove()
    if device.type == "cuda": torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3): model(sample)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        iterations = 10
        for _ in range(iterations): model(sample)
        if device.type == "cuda": torch.cuda.synchronize()
    latency_ms = 1000 * (time.perf_counter() - started) / iterations
    return {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "approximate_macs": macs,
        "approximate_flops": 2 * macs,
        "input_shape": shape,
        "latency_ms_per_batch": latency_ms,
        "throughput_images_per_second": shape[0] * 1000 / max(latency_ms, 1e-9),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint and checkpoint.exists() else None,
        **resource_snapshot(device),
    }
