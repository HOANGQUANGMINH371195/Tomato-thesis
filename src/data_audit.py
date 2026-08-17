"""Build reproducible manifests, masks, splits and data-quality reports.

This script is intentionally independent from model training. It makes the
dataset a versioned, inspectable input to both the classifier and MC-UNet.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

SUPPORTED = {
    "Tomato_healthy": "healthy",
    "Tomato_Early_blight": "early_blight",
    "Tomato_Late_blight": "late_blight",
    "Tomato_Bacterial_spot": "bacterial_spot",
}
SEGMENTATION_LABELS = {
    "Tomato_Early_blight": 1,
    "Tomato_Late_blight": 2,
    "Tomato_Bacterial_spot": 3,
}
IGNORED_LABELS = {"Tomato_Leaf_Mold"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_id(name: str) -> str:
    """Keep original UUID together with flip/rotation variants."""
    return name.split("___", 1)[0]


def canonical_label(directory_name: str) -> str | None:
    normalized = directory_name.replace("Tomato___", "Tomato_").replace("Tomato__", "Tomato_")
    aliases = {
        "Tomato_healthy": "healthy",
        "Tomato_Early_blight": "early_blight",
        "Tomato_Late_blight": "late_blight",
        "Tomato_Bacterial_spot": "bacterial_spot",
    }
    return aliases.get(normalized)


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def make_classification_manifest(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        (data_root / "PlantVillage", "plantvillage"),
        (data_root / "archive" / "Tomato Leaf Disease" / "train", "archive_train"),
        (data_root / "archive" / "Tomato Leaf Disease" / "test", "archive_test"),
    ]
    for source_root, source_name in sources:
        if not source_root.exists():
            continue
        for path in image_files(source_root):
            label_dir = path.parent.name
            label = canonical_label(label_dir)
            if label is None:
                continue
            width, height = read_image_size(path)
            rows.append(
                {
                    "path": str(path),
                    "source": source_name,
                    "label": label,
                    "raw_label": label_dir,
                    "group_id": group_id(path.name),
                    "sha256": sha256(path),
                    "width": width,
                    "height": height,
                    "file_bytes": path.stat().st_size,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Exact duplicate pixels across sources must not cross a split. Keep one row
    # per content hash; source provenance remains in duplicate_count/duplicates.
    counts = frame.groupby("sha256").size().rename("duplicate_count")
    frame = frame.join(counts, on="sha256")
    priority = {"plantvillage": 0, "archive_train": 1, "archive_test": 2}
    frame["_priority"] = frame["source"].map(priority).fillna(99)
    frame = (
        frame.sort_values(["sha256", "_priority", "path"])
        .drop_duplicates("sha256", keep="first")
        .drop(columns="_priority")
        .sort_values(["label", "path"])
        .reset_index(drop=True)
    )
    return frame


def materialize_classification(frame: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Copy deduplicated classification images into the portable artifact tree."""
    if frame.empty:
        return frame
    portable_root = output_root / "classification" / "images"
    portable_root.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    portable_paths: list[str] = []
    for row in result.itertuples():
        source = Path(row.path)
        target = portable_root / row.label / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        portable_paths.append(str(target))
    result["path"] = portable_paths
    return result


def image_lookup(root: Path) -> dict[str, Path]:
    return {p.name.lower(): p for p in image_files(root)}


def decode_embedded_image(data: str, target: Path) -> None:
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    target.write_bytes(base64.b64decode(data))


def normalized_json_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def polygon_mask(annotation: dict[str, Any]) -> tuple[np.ndarray, str, set[str]]:
    width = int(annotation["imageWidth"])
    height = int(annotation["imageHeight"])
    mask = np.zeros((height, width), dtype=np.uint8)
    seen: set[str] = set()
    image = Image.fromarray(mask)
    draw = ImageDraw.Draw(image)
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label", ""))
        seen.add(label)
        class_id = SEGMENTATION_LABELS.get(label)
        if class_id is None or shape.get("shape_type") not in {"polygon", "linestrip"}:
            continue
        points = [tuple(map(float, point)) for point in shape.get("points", [])]
        if len(points) < 3:
            continue
        clipped = [
            (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
            for x, y in points
        ]
        draw.polygon(clipped, fill=class_id)
    return np.asarray(image, dtype=np.uint8), annotation.get("imagePath", ""), seen


def heuristic_leaf_mask(image: np.ndarray) -> np.ndarray:
    """Create a documented fallback leaf mask; it is not ground truth."""
    # MC-UNet images are mostly leaf-on-light-background. A foreground mask is
    # preferable to a green-only mask because brown/necrotic lesions belong to
    # the leaf area as well. This remains a fallback, not reviewed ground truth.
    nonwhite = np.any(image < 245, axis=2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(nonwhite, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if components > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def make_segmentation_manifest(data_root: Path, output_root: Path) -> pd.DataFrame:
    dataset_root = data_root / "MC-UNet" / "Dataset"
    labels_root = dataset_root / "Labels"
    images_root = dataset_root / "Images"
    image_map = image_lookup(images_root)
    image_out = output_root / "mc_unet" / "images"
    mask_out = output_root / "mc_unet" / "masks"
    leaf_out = output_root / "mc_unet" / "leaf_masks_heuristic"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    leaf_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for json_path in sorted(labels_root.glob("*.json")):
        annotation = json.loads(json_path.read_text(encoding="utf-8"))
        mask, referenced_name, seen = polygon_mask(annotation)
        supported_seen = sorted(set(seen) & set(SEGMENTATION_LABELS))
        if not supported_seen:
            continue  # Leaf Mold and any unknown-only annotation are out of scope.
        image_name = Path(referenced_name).name
        image_path = image_map.get(image_name.lower())
        stem = json_path.stem
        canonical_image = image_out / f"{stem}.jpg"
        if image_path is not None:
            canonical_image.write_bytes(image_path.read_bytes())
        elif annotation.get("imageData"):
            decode_embedded_image(annotation["imageData"], canonical_image)
        else:
            rows.append({"json_path": str(json_path), "status": "missing_image_data"})
            continue
        with Image.open(canonical_image) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
        mask_path = mask_out / f"{stem}.png"
        leaf_path = leaf_out / f"{stem}.png"
        Image.fromarray(mask).save(mask_path)
        leaf_mask = np.maximum(heuristic_leaf_mask(rgb), (mask > 0).astype(np.uint8))
        Image.fromarray((leaf_mask * 255).astype(np.uint8)).save(leaf_path)
        lesion_pixels = int(np.count_nonzero(mask))
        leaf_pixels = int(np.count_nonzero(leaf_mask))
        rows.append(
            {
                "sample_id": stem,
                "group_id": group_id(stem),
                "annotation_id": json_path.stem,
                "image_path": str(canonical_image),
                "mask_path": str(mask_path),
                "leaf_mask_path": str(leaf_path),
                "image_materialization": "source_jpg" if image_path else "embedded_imageData",
                "label": canonical_label(supported_seen[0]),
                "labels_in_json": ";".join(sorted(seen)),
                "image_width": int(annotation["imageWidth"]),
                "image_height": int(annotation["imageHeight"]),
                "lesion_pixels": lesion_pixels,
                "leaf_pixels_heuristic": leaf_pixels,
                "lesion_fraction_image": lesion_pixels / float(mask.size),
                "lesion_fraction_heuristic_leaf": lesion_pixels / float(max(leaf_pixels, 1)),
                "early_blight_pixels": int(np.count_nonzero(mask == 1)),
                "late_blight_pixels": int(np.count_nonzero(mask == 2)),
                "bacterial_spot_pixels": int(np.count_nonzero(mask == 3)),
                "shape_count": len(annotation.get("shapes", [])),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def assign_group_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    result = frame.copy()
    result["split"] = "train"
    groups = result[["group_id", "label"]].drop_duplicates()
    _, holdout_groups = train_test_split(
        groups["group_id"],
        test_size=0.20,
        random_state=seed,
        stratify=groups["label"],
    )
    holdout = groups[groups["group_id"].isin(holdout_groups)].copy()
    val_group_ids, test_group_ids = train_test_split(
        holdout["group_id"],
        test_size=0.50,
        random_state=seed + 1,
        stratify=holdout["label"],
    )
    val_groups = set(val_group_ids)
    test_groups = set(test_group_ids)
    result.loc[result["group_id"].isin(val_groups), "split"] = "val"
    result.loc[result["group_id"].isin(test_groups), "split"] = "test"
    return result


def frame_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.groupby(columns, dropna=False).size().reset_index(name="count").to_dict("records")


def write_report(
    output_root: Path,
    classification: pd.DataFrame,
    segmentation: pd.DataFrame,
    source_counts: dict[str, Any],
) -> None:
    train_class_counts = classification[classification["split"] == "train"]["label"].value_counts()
    train_seg_counts = segmentation[segmentation["split"] == "train"]["label"].value_counts()
    class_balance = pd.DataFrame({"count": train_class_counts}).rename_axis("label").reset_index()
    class_balance["fraction"] = class_balance["count"] / class_balance["count"].sum()
    class_balance["inverse_frequency_weight"] = class_balance["count"].sum() / (
        len(class_balance) * class_balance["count"]
    )
    seg_balance = pd.DataFrame({"count": train_seg_counts}).rename_axis("label").reset_index()
    seg_balance["fraction"] = seg_balance["count"] / seg_balance["count"].sum()
    seg_balance["inverse_frequency_weight"] = seg_balance["count"].sum() / (
        len(seg_balance) * seg_balance["count"]
    )
    class_balance.to_csv(output_root / "classification_balance.csv", index=False)
    seg_balance.to_csv(output_root / "segmentation_balance.csv", index=False)
    class_imbalance = (
        float(class_balance["count"].max() / class_balance["count"].min())
        if not class_balance.empty
        else 0.0
    )
    seg_imbalance = (
        float(seg_balance["count"].max() / seg_balance["count"].min())
        if not seg_balance.empty
        else 0.0
    )
    report: list[str] = [
        "# Data audit report",
        "",
        "> Generated by `uv run python -m src.data_audit`. Do not edit manually.",
        "",
        "## Scope",
        "",
        "Supported classes: healthy, Early Blight, Late Blight and Bacterial Spot. Leaf Mold is excluded.",
        "",
        "## Classification manifest",
        "",
        f"- unique image rows: **{len(classification):,}**",
        f"- raw rows before exact-content deduplication: **{source_counts['classification_raw_rows']:,}**",
        f"- exact duplicate rows removed: **{source_counts['classification_raw_rows'] - len(classification):,}**",
        f"- duplicate content groups retained once: **{source_counts['classification_duplicate_groups']:,}**",
        "",
        "### Class × split",
        "",
        classification.groupby(["split", "label"]).size().unstack(fill_value=0).to_markdown() if not classification.empty else "No rows.",
        "",
        "### Training balance",
        "",
        f"Training class imbalance ratio (largest/smallest): **{class_imbalance:.3f}**.",
        "",
        class_balance.to_markdown(index=False) if not class_balance.empty else "No rows.",
        "",
        "## Segmentation manifest",
        "",
        f"- supported annotated samples: **{len(segmentation):,}**",
        f"- JSON/image samples generated from embedded imageData: **{sum(segmentation['image_materialization'].eq('embedded_imageData')) if not segmentation.empty else 0:,}**",
        "",
        "### Disease × split",
        "",
        segmentation.groupby(["split", "label"]).size().unstack(fill_value=0).to_markdown() if not segmentation.empty else "No rows.",
        "",
        "### Segmentation training balance",
        "",
        f"Training segmentation imbalance ratio (largest/smallest): **{seg_imbalance:.3f}**.",
        "",
        seg_balance.to_markdown(index=False) if not seg_balance.empty else "No rows.",
        "",
        "### Segmentation quality indicators",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| samples with lesion pixels | {int((segmentation['lesion_pixels'] > 0).sum()) if not segmentation.empty else 0} |",
        f"| samples with empty heuristic leaf mask | {int((segmentation['leaf_pixels_heuristic'] == 0).sum()) if not segmentation.empty else 0} |",
        f"| total supported polygons converted | {int(segmentation['shape_count'].sum()) if not segmentation.empty else 0} |",
        f"| median lesion/image fraction | {segmentation['lesion_fraction_image'].median():.6f} |" if not segmentation.empty else "| median lesion/image fraction | 0 |",
        f"| median lesion/heuristic-leaf fraction | {segmentation['lesion_fraction_heuristic_leaf'].median():.6f} |" if not segmentation.empty else "| median lesion/heuristic-leaf fraction | 0 |",
        "",
        "## Gates to review before training",
        "",
        "- Confirm every manifest path exists.",
        "- Inspect random image/mask/leaf-mask overlays.",
        "- Confirm group IDs do not cross train/val/test.",
        "- Confirm the heuristic leaf mask is acceptable or replace it with a reviewed leaf mask.",
        "- Use only train rows for balancing and augmentation.",
    ]
    (output_root / "data_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    classification = make_classification_manifest(args.data_root)
    if not classification.empty:
        classification = materialize_classification(classification, args.output_root)
        classification = assign_group_splits(classification, args.seed)
        classification.to_csv(args.output_root / "classification_manifest.csv", index=False)

    segmentation = make_segmentation_manifest(args.data_root, args.output_root)
    if not segmentation.empty:
        segmentation = assign_group_splits(segmentation, args.seed)
        segmentation.to_csv(args.output_root / "segmentation_manifest.csv", index=False)
        segmentation.to_csv(args.output_root / "segmentation_quality.csv", index=False)

    raw_classification = sum(
        1
        for source in [
            args.data_root / "PlantVillage",
            args.data_root / "archive" / "Tomato Leaf Disease" / "train",
            args.data_root / "archive" / "Tomato Leaf Disease" / "test",
        ]
        for path in image_files(source)
        if canonical_label(path.parent.name) is not None
    )
    duplicate_groups = int((classification["duplicate_count"] > 1).sum()) if not classification.empty else 0
    source_counts = {
        "classification_raw_rows": raw_classification,
        "classification_duplicate_groups": duplicate_groups,
    }
    write_report(args.output_root, classification, segmentation, source_counts)
    summary = {
        "seed": args.seed,
        "classification_rows": len(classification),
        "segmentation_rows": len(segmentation),
        "classification_counts": frame_counts(classification, ["split", "label"]),
        "segmentation_counts": frame_counts(segmentation, ["split", "label"]),
        "source_counts": source_counts,
        "supported_labels": sorted(SUPPORTED.values()),
        "excluded_labels": sorted(IGNORED_LABELS),
    }
    (args.output_root / "data_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
