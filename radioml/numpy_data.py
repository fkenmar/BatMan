from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


class H5Batcher:
    def __init__(
        self,
        path: str | Path,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int = 1337,
        normalize: bool = True,
        sort_for_io: bool = False,
    ) -> None:
        self.path = Path(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.normalize = normalize
        self.sort_for_io = sort_for_io

    def __iter__(self):
        import h5py

        rng = np.random.default_rng(self.seed)
        order = self.indices.copy()
        if self.sort_for_io:
            order = np.sort(order)
        elif self.shuffle:
            rng.shuffle(order)

        with h5py.File(self.path, "r") as h5:
            x_data = h5["X"]
            y_data = h5["Y"]
            z_data = h5["Z"]
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                sorted_idx = np.sort(batch_idx)
                restore = np.argsort(np.argsort(batch_idx))

                x = np.asarray(x_data[sorted_idx], dtype=np.float32)[restore]
                if x.ndim != 3:
                    raise ValueError(f"Expected X batch to be 3D, got {x.shape}")
                if x.shape[1] == 2:
                    x = np.moveaxis(x, 1, -1)
                if x.shape[-1] != 2:
                    raise ValueError(f"Expected I/Q final dimension 2, got {x.shape}")

                if self.normalize:
                    power = np.mean(np.square(x), axis=(1, 2), keepdims=True, dtype=np.float32)
                    x = x / np.sqrt(power + 1e-8)

                y = _as_integer_labels(np.asarray(y_data[sorted_idx]))[restore]
                snr = _as_snr_vector(np.asarray(z_data[sorted_idx]))[restore]
                yield x, y, snr

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))


class ArrayBatcher:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        snr: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int = 1337,
    ) -> None:
        self.x = x
        self.y = y
        self.snr = snr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(self.y))
        if self.shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start : start + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx], self.snr[batch_idx]

    def __len__(self) -> int:
        return int(np.ceil(len(self.y) / self.batch_size))


def load_h5_arrays(
    path: str | Path,
    indices: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    indices = np.asarray(indices, dtype=np.int64)
    sorted_idx = np.sort(indices)
    restore = np.argsort(np.argsort(indices))

    with h5py.File(path, "r") as h5:
        x = np.asarray(h5["X"][sorted_idx], dtype=np.float32)[restore]
        if x.ndim != 3:
            raise ValueError(f"Expected X batch to be 3D, got {x.shape}")
        if x.shape[1] == 2:
            x = np.moveaxis(x, 1, -1)
        if x.shape[-1] != 2:
            raise ValueError(f"Expected I/Q final dimension 2, got {x.shape}")

        if normalize:
            power = np.mean(np.square(x), axis=(1, 2), keepdims=True, dtype=np.float32)
            x = x / np.sqrt(power + 1e-8)

        y = _as_integer_labels(np.asarray(h5["Y"][sorted_idx]))[restore]
        snr = _as_snr_vector(np.asarray(h5["Z"][sorted_idx]))[restore]
    return x, y, snr
