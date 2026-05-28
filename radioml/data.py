from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez(path, train=self.train, val=self.val, test=self.test)

    @classmethod
    def load(cls, path: str | Path) -> "SplitIndices":
        data = np.load(path)
        return cls(train=data["train"], val=data["val"], test=data["test"])


def _as_integer_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int64)
    raise ValueError(f"Unsupported label shape: {y.shape}")


def _as_snr_vector(z: np.ndarray) -> np.ndarray:
    if z.ndim == 1:
        return z.astype(np.int64)
    if z.ndim == 2 and z.shape[1] == 1:
        return z[:, 0].astype(np.int64)
    raise ValueError(f"Unsupported SNR shape: {z.shape}")


def read_labels_and_snr(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as h5:
        labels = _as_integer_labels(np.asarray(h5["Y"]))
        snr = _as_snr_vector(np.asarray(h5["Z"]))
    return labels, snr


def make_stratified_split(
    labels: np.ndarray,
    snr: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 1337,
) -> SplitIndices:
    if not (0.0 < train_frac < 1.0 and 0.0 <= val_frac < 1.0):
        raise ValueError("train_frac and val_frac must be fractions")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for test data")
    if len(labels) != len(snr):
        raise ValueError("labels and snr must have the same length")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    strata = np.stack([labels.astype(np.int64), snr.astype(np.int64)], axis=1)
    for label, snr_value in np.unique(strata, axis=0):
        idx = np.flatnonzero((labels == label) & (snr == snr_value))
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_frac))
        n_val = int(round(len(idx) * val_frac))
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])

    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    test = np.concatenate(test_parts)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitIndices(train=train, val=val, test=test)


class RadioMLDataset(Dataset[dict[str, Any]]):
    """Lazy HDF5 dataset for RadioML 2018.01A-style files."""

    def __init__(
        self,
        path: str | Path,
        indices: np.ndarray | None = None,
        normalize: bool = True,
    ) -> None:
        self.path = Path(path)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.normalize = normalize
        self._h5 = None
        self._x = None
        self._y = None
        self._z = None

        labels, snr = read_labels_and_snr(self.path)
        self.labels = labels if self.indices is None else labels[self.indices]
        self.snr = snr if self.indices is None else snr[self.indices]
        self.num_classes = int(labels.max()) + 1

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        labels = self.labels
        return len(labels)

    def _open(self) -> None:
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.path, "r")
            self._x = self._h5["X"]
            self._y = self._h5["Y"]
            self._z = self._h5["Z"]

    def __getitem__(self, item: int) -> dict[str, Any]:
        self._open()
        source_idx = int(self.indices[item]) if self.indices is not None else int(item)
        assert self._x is not None and self._y is not None and self._z is not None

        x = np.asarray(self._x[source_idx], dtype=np.float32)
        if x.shape[0] == 2 and x.ndim == 2:
            x = np.moveaxis(x, 0, -1)
        if x.ndim != 2 or x.shape[-1] != 2:
            raise ValueError(f"Expected I/Q sample with final dimension 2, got {x.shape}")

        if self.normalize:
            power = np.mean(np.square(x), dtype=np.float32)
            x = x / np.sqrt(power + 1e-8)

        y_raw = np.asarray(self._y[source_idx])
        label = int(np.argmax(y_raw)) if y_raw.ndim else int(y_raw)
        z_raw = np.asarray(self._z[source_idx])
        snr = int(z_raw.reshape(-1)[0])

        return {
            "iq": torch.from_numpy(x),
            "label": torch.tensor(label, dtype=torch.long),
            "snr": torch.tensor(snr, dtype=torch.long),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._x = None
        self._y = None
        self._z = None
