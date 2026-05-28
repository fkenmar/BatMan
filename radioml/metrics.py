from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AccuracyReport:
    accuracy: float
    low_snr_accuracy: float
    per_snr_accuracy: dict[int, float]


def accuracy_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    low_snr_min: int = -20,
    low_snr_max: int = 0,
) -> AccuracyReport:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    snr = np.asarray(snr).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(snr)):
        raise ValueError("y_true, y_pred, and snr must have the same length")

    correct = y_true == y_pred
    overall = float(correct.mean()) if len(correct) else 0.0

    low_mask = (snr >= low_snr_min) & (snr <= low_snr_max)
    low_acc = float(correct[low_mask].mean()) if low_mask.any() else 0.0

    per_snr = {}
    for value in sorted(np.unique(snr).astype(int).tolist()):
        mask = snr == value
        per_snr[value] = float(correct[mask].mean()) if mask.any() else 0.0

    return AccuracyReport(
        accuracy=overall,
        low_snr_accuracy=low_acc,
        per_snr_accuracy=per_snr,
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        matrix[int(target), int(pred)] += 1
    return matrix
