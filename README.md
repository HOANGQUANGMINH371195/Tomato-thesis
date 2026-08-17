# Prediction of Tomato Yield Impact Based on Leaf Disease using Deep Learning

This repository contains a reproducible deep-learning pipeline for:

1. classification of `healthy`, `early_blight`, `late_blight`, and `bacterial_spot` leaves;
2. lesion segmentation for the three supported diseases;
3. disease-severity estimation from predicted lesion area;
4. low, central, and high potential yield-impact sensitivity scenarios;
5. automatic experiment logging, recovery, evaluation, figures, and report generation.

Leaf Mold is outside the scope of the project.

## Run the complete experiment

Install [uv](https://docs.astral.sh/uv/), clone the repository, restore Git LFS objects,
and run this command from the project root:

```bash
git lfs pull
uv run python -m src.train --task all --device auto
```

The data gate runs automatically before training. The command then executes the
DenseNet-121 baseline, residual-attention classifier, Attention-Residual U-Net baseline,
and MCUNet-Seg experiment. On a compatible CUDA system, `--device auto` selects the GPU;
otherwise it uses the CPU.

If training is interrupted, run the same command again. The newest incomplete session is
detected automatically, completed tasks are skipped, and unfinished tasks resume from the
latest checkpoint.

## Data

All canonical data is stored under `artifacts/data/`; no second data directory or external
archive is required.

```text
artifacts/data/
├── classification/images/<class>/
├── classification_manifest.csv
├── classification_balance.csv
├── mc_unet/images/
├── mc_unet/masks/
├── mc_unet/leaf_masks_heuristic/
├── segmentation_manifest.csv
├── segmentation_balance.csv
├── segmentation_quality.csv
├── data_audit.md
└── data_audit.json
```

The canonical snapshot contains 6,613 deduplicated classification images and 750
segmentation samples. Group-aware train/validation/test splits prevent transformed copies
of one source image from crossing subsets.

## Models

- Classification baseline: ImageNet-pretrained DenseNet-121.
- Classification model: DenseNet-121 with a custom residual-attention head.
- Segmentation baseline: Attention-Residual U-Net.
- Segmentation model: MCUNet-Seg with a compact inverted-residual encoder and U-Net-style
  decoder.

Default hyperparameters are defined in `configs/default.yaml`. The standard configuration
uses 224×224 classification inputs, 256×256 segmentation inputs, mixed precision on CUDA,
gradient accumulation, early stopping, and best-validation checkpoint selection.

## Experiment outputs

The completed experiment and every associated artifact are stored under
`artifacts/runs/overnight_20260816_183758/`. Each task contains:

- best and latest checkpoints;
- complete epoch history and console logs;
- configuration and environment snapshots;
- test predictions and predicted segmentation masks;
- aggregate and per-class metrics;
- TP, FP, TN, FN, precision, recall, specificity, F1, Dice, IoU, ROC-AUC, calibration,
  boundary, and distance metrics where applicable;
- training curves, confusion matrices, ROC/PR plots, attribution maps, segmentation
  overlays, severity plots, and comparison figures.

Large checkpoint files are versioned with Git LFS. Install Git LFS before cloning or run
`git lfs pull` afterward to restore their full contents.

## Prediction on a new image

```bash
uv run python -m src.predict --image /path/to/tomato_leaf.jpg --device auto
```

The prediction stage saves disease probabilities, lesion and leaf masks, severity,
potential yield-impact scenarios, Integrated Gradients, and a visual workflow result.

## Validation

```bash
uv run pytest -q
uv run ruff check --select F src tests
```

For a short CPU smoke test that must not be reported as a thesis result:

```bash
uv run python -m src.train --task all --epochs 1 --workers 0 --device cpu --no-pretrained --no-auto-resume
```

## Thesis report

The academic report is available as `report/main.pdf`; its LaTeX source and bibliography
are in `report/main.tex` and `report/references.bib`. To rebuild it:

```bash
cd report
tectonic main.tex
```

The potential yield-impact values are transparent sensitivity scenarios, not coefficients
fitted from paired field-yield observations. Field measurements are required before they
can be interpreted as validated yield-loss predictions.
