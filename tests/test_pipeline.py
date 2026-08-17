from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import random

import numpy as np
import torch

from src.mcunet_unet import MCUNetSegmentation
from src.train import (accumulation_divisor, classification_metrics, ensure_session,
                       latest_incomplete_session, load_checkpoint, save_checkpoint,
                       segmentation_boundary_metrics, set_session_task)
from src.yield_model import calculate_yield_loss, yield_loss_interval


class PipelineTests(unittest.TestCase):
    def test_mcunet_output_shape(self) -> None:
        model = MCUNetSegmentation(4, .5).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 4, 256, 256))

    def test_accumulation_final_partial_group(self) -> None:
        self.assertEqual([accumulation_divisor(i, 10, 4) for i in range(1, 11)], [4] * 8 + [2] * 2)

    def test_advanced_classification_metrics(self) -> None:
        true = [0, 1, 2, 3] * 2
        probabilities = np.asarray([[.8, .1, .05, .05], [.1, .7, .1, .1], [.1, .1, .7, .1], [.05, .05, .1, .8]] * 2)
        metrics = classification_metrics(true, probabilities.argmax(1).tolist(), probabilities.tolist())
        for key in ("roc_auc_ovr_macro", "average_precision_macro", "matthews_correlation_coefficient",
                    "cohen_kappa", "multiclass_brier_score", "expected_calibration_error_10_bins"):
            self.assertIn(key, metrics)

    def test_boundary_metrics_identical(self) -> None:
        logits = torch.zeros(2, 4, 16, 16); logits[:, 0] = 1
        target = torch.zeros(2, 16, 16, dtype=torch.long)
        metrics = segmentation_boundary_metrics(logits, target, 4)
        self.assertEqual(metrics["background_boundary_iou"], 1.0)
        self.assertEqual(metrics["background_hausdorff_pixels"], 0.0)

    def test_yield_scenarios(self) -> None:
        self.assertEqual(calculate_yield_loss("Late Blight", 100), 100.0)
        self.assertEqual(yield_loss_interval("bacterial_spot", 50), {"low": 15.0, "central": 20.0, "high": 22.5})
        self.assertEqual(calculate_yield_loss("healthy", 100), 0.0)

    def test_session_power_gap(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary); group = "overnight_test"
            ensure_session(root, group, ["classification", "segmentation"])
            set_session_task(root, group, "classification", "success")
            self.assertEqual(latest_incomplete_session(root, ["classification", "segmentation"]), group)
            set_session_task(root, group, "segmentation", "success")
            self.assertEqual(latest_incomplete_session(root, ["classification", "segmentation"]), group)
            session_path = root / group / "session.json"
            payload = json.loads(session_path.read_text()); payload["status"] = "success"
            session_path.write_text(json.dumps(payload))
            self.assertIsNone(latest_incomplete_session(root, ["classification", "segmentation"]))

    def test_checkpoint_restores_training_and_rng_state(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary); (root / "checkpoints").mkdir()
            run = SimpleNamespace(path=root); model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            random.seed(17); np.random.seed(17); torch.manual_seed(17)
            save_checkpoint(run, model, optimizer, 4, .6, "last.pt", best_metric=.8, best_epoch=2, patience=2)
            expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
            random.random(); np.random.random(); torch.rand(1)
            payload = load_checkpoint(root / "checkpoints" / "last.pt", model, optimizer, torch.device("cpu"))
            actual = (random.random(), float(np.random.random()), float(torch.rand(1)))
            self.assertEqual(actual, expected)
            self.assertEqual((payload["epoch"], payload["best_epoch"], payload["patience"]), (4, 2, 2))


if __name__ == "__main__":
    unittest.main()
