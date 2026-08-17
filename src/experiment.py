"""Experiment bookkeeping helpers used by future train/evaluate commands."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_RUN: "ExperimentRun | None" = None


def utc_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "uv_lock_hash": _command(["sha256sum", "uv.lock"]),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
    }
    try:
        import torch
        cuda_error = None
        cuda_available = False
        try:
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                # Force lazy initialization so an incompatible driver is
                # captured as metadata rather than crashing a run.
                torch.cuda.init()
        except Exception as error:  # driver/toolkit mismatch is reportable state
            cuda_error = repr(error)
        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda_available": cuda_available,
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count() if cuda_available else 0,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else [],
                "cuda_error": cuda_error,
            }
        )
        if cuda_available:
            snapshot["gpu_properties"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        snapshot["torch"] = None
    return snapshot


class ExperimentRun:
    """Create a run directory and persist all configuration/metric artifacts."""

    def __init__(self, root: Path, prefix: str, config: dict[str, Any], existing_path: Path | None = None) -> None:
        global ACTIVE_RUN
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        run_group = config.get("runtime", {}).get("run_group")
        self.path = existing_path or (root / run_group / prefix if run_group else root / utc_run_id(prefix))
        self.path.mkdir(parents=True, exist_ok=existing_path is not None)
        (self.path / "checkpoints").mkdir(exist_ok=True)
        (self.path / "predictions").mkdir(exist_ok=True)
        (self.path / "figures").mkdir(exist_ok=True)
        snapshot = environment_snapshot()
        if existing_path is not None and (self.path / "config.json").exists():
            history = self.path / "resume_history"
            history.mkdir(exist_ok=True)
            resumed_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            (history / f"config_{resumed_at}.json").write_text(
                json.dumps(config, indent=2, default=str), encoding="utf-8"
            )
            (history / f"environment_{resumed_at}.json").write_text(
                json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
            )
        else:
            self.write_json("config.json", config)
            self.write_json("environment.json", snapshot)
        self.write_json("status.json", {"status": "running", "task": prefix, "run_group": run_group})
        self._update_latest("running")
        ACTIVE_RUN = self
        if existing_path is not None:
            self.log("RESUME_SESSION metadata_saved=resume_history")

    def write_json(self, name: str, value: Any) -> None:
        (self.path / name).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

    def append_metrics(self, metrics: dict[str, Any], filename: str = "metrics.csv") -> None:
        target = self.path / filename
        exists = target.exists()
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            if not exists:
                writer.writeheader()
            writer.writerow(metrics)

    def save_text(self, name: str, text: str) -> None:
        (self.path / name).write_text(text, encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with (self.path / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{message}\n")

    def complete(self, summary: dict[str, Any] | None = None) -> None:
        payload = {"status": "success", "task": self.path.name, "run_group": self.path.parent.name}
        if summary:
            payload.update(summary)
        self.write_json("status.json", payload)
        self.log("STATUS success")
        self._update_latest("success", payload)

    def fail(self, error: BaseException) -> None:
        payload = {"status": "failed", "error": repr(error)}
        self.write_json("status.json", payload)
        self.save_text("error_traceback.txt", traceback.format_exc())
        self.log(f"STATUS failed error={error!r}")
        self._update_latest("failed", payload)

    def _update_latest(self, status: str, payload: dict[str, Any] | None = None) -> None:
        latest_path = self.root / "latest.json"
        current: dict[str, Any] = {}
        if latest_path.exists():
            try:
                current = json.loads(latest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        task = self.path.name
        current[task] = {"status": status, "path": str(self.path), "run_group": self.path.parent.name, **(payload or {})}
        latest_path.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")


def mark_active_failure(error: BaseException) -> None:
    if ACTIVE_RUN is not None:
        ACTIVE_RUN.fail(error)
