import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _timestamp() -> str:
    """Simple timestamp (local time) for experiment directories."""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)


@dataclass
class ExperimentLogger:
    """
    Lightweight experiment run logger.

    Creates a directory structure:
      experiments/<ts>__<experiment>__<method>__diag-<mode>__obj-<flag>/
        seed_00001/
          logs/
          metrics/
          plots/

    This is purely organizational and does not change training logic.
    """

    args: object
    seeds: List[int]
    base_dir: Path = field(init=False)
    seed_dirs: Dict[int, Path] = field(init=False, default_factory=dict)
    summary_path: Path = field(init=False)

    def __post_init__(self):
        timestamp = _timestamp()
        experiment = _sanitize(str(getattr(self.args, "experiment", "exp")))
        method = _sanitize(str(getattr(self.args, "method", "method")))
        diag = getattr(self.args, "diagnosis_mode", "binary") if hasattr(self.args, "diagnosis_mode") else "binary"
        obj_flag = "obj1" if getattr(self.args, "objective_aware", False) else "obj0"

        self.base_dir = Path("experiments") / f"{timestamp}__{experiment}__{method}__diag-{diag}__{obj_flag}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.base_dir / "summary.json"
        self._write_command(self.base_dir)
        self._write_config(self.base_dir, self.args)

    def _write_command(self, directory: Path):
        command_file = directory / "command.txt"
        try:
            command_file.write_text(" ".join(sys.argv))
        except Exception:
            pass

    def _write_config(self, directory: Path, args_obj: object):
        config_file = directory / "config.json"
        try:
            config_file.write_text(json.dumps(vars(args_obj), indent=2, default=str))
        except Exception:
            pass

    def get_seed_dir(self, seed: int) -> Path:
        if seed in self.seed_dirs:
            return self.seed_dirs[seed]
        seed_dir = self.base_dir / f"seed_{seed:05d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = seed_dir / "logs"
        plots_dir = seed_dir / "plots"
        metrics_dir = seed_dir / "metrics"
        for directory in (logs_dir, plots_dir, metrics_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._write_command(seed_dir)
        # config with seed override
        seed_args = dict(vars(self.args))
        seed_args["seed"] = seed
        try:
            seed_dir.joinpath("config.json").write_text(json.dumps(seed_args, indent=2, default=str))
        except Exception:
            pass
        self.seed_dirs[seed] = seed_dir
        return seed_dir

    def seed_logs_dir(self, seed: int) -> Path:
        return self.get_seed_dir(seed) / "logs"

    def seed_metrics_dir(self, seed: int) -> Path:
        return self.get_seed_dir(seed) / "metrics"

    def seed_plots_dir(self, seed: int) -> Path:
        return self.get_seed_dir(seed) / "plots"

    def write_metrics(self, seed: int, metrics: Dict):
        metrics_path = self.seed_metrics_dir(seed) / "metrics.json"
        try:
            metrics_path.write_text(json.dumps(metrics, indent=2, default=float))
        except Exception:
            pass

    def write_plot(self, seed: int, filename: str, figure):
        """
        Save a matplotlib figure into the per-seed plots directory.
        """
        plots_dir = self.seed_plots_dir(seed)
        try:
            plots_dir.mkdir(parents=True, exist_ok=True)
            figure_path = plots_dir / filename
            figure.savefig(figure_path, bbox_inches="tight")
        except Exception:
            # Plotting is best-effort and must not break experiments.
            pass

    def save_summary(self, data):
        try:
            self.summary_path.write_text(json.dumps(data, indent=2, default=float))
        except Exception:
            pass
