"""One-command end-to-end inference for a new tomato leaf image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .data_audit import heuristic_leaf_mask
from .mcunet_unet import MCUNetSegmentation
from .train import CLASS_NAMES, SEGMENTATION_LABEL_TO_ID, Classifier, load_config, resolve_device
from .yield_model import metadata, yield_loss_interval


def _successful_checkpoint(root: Path, task: str) -> Path:
    candidates: list[Path] = []
    for status_path in root.glob(f"overnight_*/{task}/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = status_path.parent / "checkpoints" / "best.pt"
        if status.get("status") == "success" and checkpoint.exists():
            candidates.append(checkpoint)
    if not candidates:
        raise FileNotFoundError(f"No successful {task} best checkpoint found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_state(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    return payload


def integrated_gradients(model: Classifier, tensor: torch.Tensor, target: int, steps: int = 16) -> np.ndarray:
    baseline = torch.zeros_like(tensor); total_gradient = torch.zeros_like(tensor)
    for alpha in torch.linspace(1 / steps, 1, steps, device=tensor.device):
        sample = (baseline + alpha * (tensor - baseline)).detach().requires_grad_(True)
        score = model(sample)[0, target]
        gradient = torch.autograd.grad(score, sample)[0]
        total_gradient += gradient.detach()
    attribution = ((tensor - baseline) * total_gradient / steps).abs().sum(1)[0].detach().cpu().numpy()
    attribution -= attribution.min(); attribution /= max(float(attribution.max()), 1e-12)
    return attribution


def predict(image_path: Path, classification_checkpoint: Path, segmentation_checkpoint: Path,
            config_path: Path, output: Path, device_name: str = "auto") -> Path:
    config = load_config(config_path); device = resolve_device(device_name)
    output.mkdir(parents=True, exist_ok=False)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    classification_size = int(config["data"]["image_size_classification"])
    segmentation_size = int(config["data"]["image_size_segmentation"])
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    classifier_input = normalize(transforms.ToTensor()(image.resize((classification_size, classification_size), Image.Resampling.BILINEAR))).unsqueeze(0).to(device)
    classifier = Classifier(4, attention=True, pretrained=False).to(device)
    classification_payload = _load_state(classifier, classification_checkpoint, device)
    with torch.no_grad():
        class_probability = torch.softmax(classifier(classifier_input), 1)[0]
    class_id = int(class_probability.argmax()); disease = CLASS_NAMES[class_id]
    attribution = integrated_gradients(classifier, classifier_input, class_id)

    resized = image.resize((segmentation_size, segmentation_size), Image.Resampling.BILINEAR)
    segmentation_input = normalize(transforms.ToTensor()(resized)).unsqueeze(0).to(device)
    segmenter = MCUNetSegmentation(4, float(config["segmentation"].get("width_mult", .5))).to(device)
    segmentation_payload = _load_state(segmenter, segmentation_checkpoint, device)
    with torch.no_grad():
        semantic_mask = segmenter(segmentation_input).argmax(1)[0].cpu().numpy().astype(np.uint8)
    leaf_mask = heuristic_leaf_mask(np.asarray(resized)) > 0
    disease_id = SEGMENTATION_LABEL_TO_ID.get(disease, 0)
    conditional_mask = (semantic_mask == disease_id) & leaf_mask if disease_id else np.zeros_like(leaf_mask)
    severity = 100 * np.count_nonzero(conditional_mask) / max(np.count_nonzero(leaf_mask), 1)
    impact = yield_loss_interval(disease, severity)

    Image.fromarray(semantic_mask).save(output / "semantic_mask.png")
    Image.fromarray((conditional_mask.astype(np.uint8) * 255)).save(output / "disease_mask.png")
    Image.fromarray((leaf_mask.astype(np.uint8) * 255)).save(output / "leaf_mask_heuristic.png")
    image.save(output / "input.png")
    probabilities = {name: float(class_probability[index]) for index, name in enumerate(CLASS_NAMES)}
    result = {
        "input": str(image_path), "device": str(device), "predicted_disease": disease,
        "confidence": float(class_probability[class_id]), "probabilities": probabilities,
        "severity_proxy_percent": float(severity), "potential_yield_impact_percent": impact,
        "leaf_pixels_heuristic": int(np.count_nonzero(leaf_mask)),
        "disease_pixels": int(np.count_nonzero(conditional_mask)),
        "classification_checkpoint": str(classification_checkpoint),
        "segmentation_checkpoint": str(segmentation_checkpoint),
        "classification_checkpoint_epoch": int(classification_payload.get("epoch", 0)),
        "segmentation_checkpoint_epoch": int(segmentation_payload.get("epoch", 0)),
        "yield_model": metadata(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "RESULT.md").write_text(
        f"# Prediction result\n\nDisease: **{disease}** ({result['confidence']:.2%})\n\n"
        f"Severity proxy: **{severity:.2f}%**\n\nPotential yield impact: **{impact['central']:.2f}%** "
        f"(sensitivity interval {impact['low']:.2f}–{impact['high']:.2f}%)\n\n"
        "This is a model-based scenario, not a field-calibrated yield forecast.\n", encoding="utf-8")

    display = np.asarray(resized) / 255.0
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    axes[0].imshow(display); axes[0].set_title("Input")
    axes[1].imshow(display); axes[1].imshow(np.asarray(Image.fromarray((attribution * 255).astype(np.uint8)).resize((segmentation_size, segmentation_size))) / 255, cmap="inferno", alpha=.55); axes[1].set_title("Integrated Gradients")
    axes[2].imshow(semantic_mask, vmin=0, vmax=3, cmap="viridis"); axes[2].set_title("Semantic lesion mask")
    axes[3].imshow(display); axes[3].imshow(conditional_mask, cmap="Reds", alpha=.5); axes[3].set_title(f"{disease}\nS={severity:.1f}% | YL={impact['central']:.1f}%")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(output / "prediction_workflow.png", dpi=200); plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--classification-checkpoint", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    classification = args.classification_checkpoint or _successful_checkpoint(args.runs_root, "classification")
    segmentation = args.segmentation_checkpoint or _successful_checkpoint(args.runs_root, "segmentation")
    output = args.output or Path("artifacts/inference") / time.strftime("prediction_%Y%m%d_%H%M%S", time.gmtime())
    print(f"Saved prediction to {predict(args.image, classification, segmentation, args.config, output, args.device)}")


if __name__ == "__main__":
    main()
