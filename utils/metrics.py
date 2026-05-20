"""
Shared metrics and artifact helpers for decoded radar arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_dbz_metrics(data: np.ndarray) -> dict[str, Any]:
    """Summarise a decoded dBZ field."""
    finite_mask = np.isfinite(data)
    signal_mask = finite_mask & (data > 0)
    finite = data[finite_mask]

    if finite.size == 0:
        dbz_min = None
        dbz_max = None
        class_counts: dict[str, int] = {}
        class_percentages: dict[str, float] = {}
        unique_value_count = 0
    else:
        dbz_min = float(np.min(finite))
        dbz_max = float(np.max(finite))
        values, counts = np.unique(finite, return_counts=True)
        unique_value_count = int(values.size)
        if unique_value_count <= 32:
            class_counts = {
                _format_dbz_key(value): int(count)
                for value, count in zip(values.tolist(), counts.tolist())
            }
        else:
            class_counts = {}
        total_finite = int(finite.size)
        class_percentages = {
            key: round(count / total_finite * 100.0, 3)
            for key, count in class_counts.items()
        }

    return {
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "dbz_min": dbz_min,
        "dbz_max": dbz_max,
        "signal_pixels": int(np.sum(signal_mask)),
        "finite_pixels": int(np.sum(finite_mask)),
        "nan_pixels": int(np.sum(~finite_mask)),
        "unique_value_count": unique_value_count,
        "class_counts_dbz": class_counts,
        "class_percentages_dbz": class_percentages,
    }


def compute_difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Summarise candidate - reference over overlapping finite pixels."""
    mask = np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(mask):
        return {
            "overlap_pixels": 0,
            "mean_diff_dbz": None,
            "median_diff_dbz": None,
            "min_diff_dbz": None,
            "max_diff_dbz": None,
            "pct_positive_diff": None,
            "pct_negative_diff": None,
            "pct_zeroish_diff": None,
        }

    diff = candidate[mask] - reference[mask]
    total = diff.size
    return {
        "overlap_pixels": int(total),
        "mean_diff_dbz": float(np.mean(diff)),
        "median_diff_dbz": float(np.median(diff)),
        "min_diff_dbz": float(np.min(diff)),
        "max_diff_dbz": float(np.max(diff)),
        "pct_positive_diff": float(np.mean(diff > 1e-6) * 100.0),
        "pct_negative_diff": float(np.mean(diff < -1e-6) * 100.0),
        "pct_zeroish_diff": float(np.mean(np.abs(diff) <= 1e-6) * 100.0),
    }


def compare_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare two metric snapshots from the same decoded source."""
    ref_counts = reference.get("class_counts_dbz", {})
    cand_counts = candidate.get("class_counts_dbz", {})
    keys = sorted(set(ref_counts) | set(cand_counts), key=_dbz_sort_key)
    delta_counts = {
        key: int(cand_counts.get(key, 0) - ref_counts.get(key, 0))
        for key in keys
        if cand_counts.get(key, 0) != ref_counts.get(key, 0)
    }
    return {
        "same_shape": reference.get("shape") == candidate.get("shape"),
        "delta_signal_pixels": int(candidate.get("signal_pixels", 0) - reference.get("signal_pixels", 0)),
        "delta_finite_pixels": int(candidate.get("finite_pixels", 0) - reference.get("finite_pixels", 0)),
        "delta_class_counts_dbz": delta_counts,
        "exact_match": (
            reference.get("shape") == candidate.get("shape")
            and reference.get("signal_pixels") == candidate.get("signal_pixels")
            and reference.get("finite_pixels") == candidate.get("finite_pixels")
            and ref_counts == cand_counts
        ),
    }


def format_dbz_metrics_block(title: str, metrics: dict[str, Any], snr: float | None = None) -> str:
    """Render a human-readable metric block."""
    dbz_min = metrics.get("dbz_min")
    dbz_max = metrics.get("dbz_max")
    range_line = "n/a" if dbz_min is None or dbz_max is None else f"{dbz_min:.1f} ... {dbz_max:.1f} dBZ"

    lines = [
        title,
        f"Shape          : ({metrics['shape'][0]}, {metrics['shape'][1]})",
        f"Signal pixels  : {metrics['signal_pixels']:,}",
        f"Finite pixels  : {metrics['finite_pixels']:,}",
        f"NaN pixels     : {metrics['nan_pixels']:,}",
        f"Value range    : {range_line}",
    ]
    if snr is not None:
        lines.append(f"SNR            : {snr:.1f} dB")

    class_counts = metrics.get("class_counts_dbz", {})
    if class_counts:
        class_line = ", ".join(f"{key}:{value:,}" for key, value in class_counts.items())
        lines.append(f"Class counts   : {class_line}")
    elif metrics.get("unique_value_count", 0) > 0:
        lines.append(f"Unique values  : {metrics['unique_value_count']:,}")

    return "\n".join(lines)


def format_difference_block(title: str, metrics: dict[str, Any]) -> str:
    """Render a human-readable difference metric block."""
    if metrics.get("overlap_pixels", 0) == 0:
        return f"{title}\nOverlap pixels  : 0"

    return "\n".join(
        [
            title,
            f"Overlap pixels : {metrics['overlap_pixels']:,}",
            f"Diff mean      : {metrics['mean_diff_dbz']:.2f} dBZ",
            f"Diff median    : {metrics['median_diff_dbz']:.2f} dBZ",
            f"Diff min/max   : {metrics['min_diff_dbz']:.2f} ... {metrics['max_diff_dbz']:.2f} dBZ",
            f"Positive diff  : {metrics['pct_positive_diff']:.1f}%",
            f"Negative diff  : {metrics['pct_negative_diff']:.1f}%",
            f"Zero-ish diff  : {metrics['pct_zeroish_diff']:.1f}%",
        ]
    )


def write_metrics_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_text(path: Path | str, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_metrics_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_dbz_key(value: float) -> str:
    return f"{float(value):g}"


def _dbz_sort_key(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("inf")
