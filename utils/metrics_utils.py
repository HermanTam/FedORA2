import json
from pathlib import Path
from statistics import mean, stdev, median
from typing import Dict, Iterable, List


def sample_stats(values: Iterable[float]) -> Dict[str, float]:
    values = [v for v in values if v is not None]
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "count": 0,
        }
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "std": 0.0, "min": v, "max": v, "median": v, "count": 1}
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(median(values)),
        "count": len(values),
    }


def aggregate_metric_lists(metric_lists: List[Dict]) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, List[float]] = {}
    for metrics in metric_lists:
        for key, value in metrics.items():
            aggregated.setdefault(key, []).append(value)
    return {key: sample_stats(vals) for key, vals in aggregated.items()}


def dump_metrics(path: Path, data: Dict):
    path.write_text(json.dumps(data, indent=2, default=float))

