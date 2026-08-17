"""Fail-fast checks for the train-ready manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def check_manifest(path: Path, segmentation: bool) -> list[str]:
    errors: list[str] = []
    frame = pd.read_csv(path)
    required = (
        {"group_id", "label", "split", "path"}
        if not segmentation
        else {"group_id", "label", "split", "image_path", "mask_path", "leaf_mask_path"}
    )
    missing_columns = required - set(frame.columns)
    if missing_columns:
        return [f"{path}: missing columns {sorted(missing_columns)}"]
    if frame.empty:
        return [f"{path}: manifest is empty"]
    allowed_splits = {"train", "val", "test"}
    if not set(frame["split"].dropna()).issubset(allowed_splits):
        errors.append(f"{path}: unexpected split values {sorted(set(frame['split'].dropna()) - allowed_splits)}")
    missing_splits = allowed_splits - set(frame["split"].dropna())
    if missing_splits:
        errors.append(f"{path}: missing required splits {sorted(missing_splits)}")
    if frame["split"].isna().any() or frame["group_id"].isna().any() or frame["label"].isna().any():
        errors.append(f"{path}: null split/group/label value")
    if segmentation and not set(frame["label"]).issubset({"early_blight", "late_blight", "bacterial_spot"}):
        errors.append(f"{path}: unsupported segmentation labels")
    if not segmentation and not set(frame["label"]).issubset({"healthy", "early_blight", "late_blight", "bacterial_spot"}):
        errors.append(f"{path}: unsupported classification labels")
    for split_a, split_b in (("train", "val"), ("train", "test"), ("val", "test")):
        a = set(frame.loc[frame.split == split_a, "group_id"])
        b = set(frame.loc[frame.split == split_b, "group_id"])
        overlap = a & b
        if overlap:
            errors.append(f"{path}: group leakage {split_a}/{split_b}: {len(overlap)} groups")
    if segmentation:
        for row in frame.itertuples():
            for column in ("image_path", "mask_path", "leaf_mask_path"):
                if not Path(getattr(row, column)).exists():
                    errors.append(f"{path}: missing {column}: {getattr(row, column)}")
    else:
        for value in frame["path"]:
            if not Path(value).exists():
                errors.append(f"{path}: missing image: {value}")
    return errors


def validate_root(root: Path) -> list[str]:
    errors: list[str] = []
    classification_path = root / "classification_manifest.csv"
    segmentation_path = root / "segmentation_manifest.csv"
    quality_path = root / "segmentation_quality.csv"
    for path in (classification_path, segmentation_path, quality_path):
        if not path.exists():
            errors.append(f"missing required artifact: {path}")
    if errors:
        return errors
    errors += check_manifest(classification_path, False)
    errors += check_manifest(segmentation_path, True)
    segmentation = pd.read_csv(quality_path)
    if segmentation.empty:
        errors.append(f"{quality_path}: quality report is empty")
    else:
        if (segmentation["lesion_pixels"] <= 0).any():
            errors.append("segmentation: at least one lesion mask is empty")
        if (segmentation["leaf_pixels_heuristic"] <= 0).any():
            errors.append("segmentation: at least one leaf mask is empty")
        if (segmentation["lesion_fraction_heuristic_leaf"] > 1.0).any():
            errors.append("segmentation: lesion fraction exceeds 100%")
    values = set()
    for path in (root / "mc_unet" / "masks").glob("*.png"):
        values.update(np.unique(np.asarray(Image.open(path))).tolist())
    if not values or not values.issubset({0, 1, 2, 3}):
        errors.append(f"segmentation: unexpected or missing mask values {sorted(values)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/data"))
    args = parser.parse_args()
    errors = validate_root(args.root)
    if errors:
        print("DATA GATE: FAIL")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("DATA GATE: PASS")
    print("- classification manifest: paths and group splits valid")
    print("- segmentation manifest: paths, masks, leaf masks and group splits valid")
    print("- supported mask values: 0, 1, 2, 3")
    print("- lesion fraction: within 0–100%")


if __name__ == "__main__":
    main()
