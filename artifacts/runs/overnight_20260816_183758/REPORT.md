# Overnight session overnight_20260816_183758

## Quick result table

| Task | Model | Headline metric | Accuracy/IoU | AUC/Boundary IoU | Parameters | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| classification_baseline | densenet121_baseline | 1.0 | 1.0 | 1.0 | 6957956 | 24.329763300193008 |
| segmentation | mcunet_segmentation | 0.5243806910106416 | 0.3682704702829495 | 0.09170366947283108 | 731108 | 6.845519199850969 |
| segmentation_baseline | attention_residual_unet | 0.7345699431849613 | 0.585506586917829 | 0.24209066543868046 | 2101220 | 6.4602358997944975 |
| classification | densenet121_custom_residual_attention | 0.9894838867334975 | 0.9909365558912386 | 0.9999699286668932 | 26099716 | 33.71075579998433 |

## Report-ready outputs

- `all_metrics_flat.csv`: every scalar from every task in one table.
- `classification_ablation.csv/.png` and `segmentation_ablation.csv/.png`: fair held-out comparisons.
- `figures/`: workflow, model architecture, dataset distribution and examples.
- Each task directory: checkpoints, epoch traces, detailed metrics, predictions and figures.

## Scientific interpretation

Yield values are low/central/high potential-impact sensitivity scenarios. They are not field-calibrated farm-yield forecasts.
