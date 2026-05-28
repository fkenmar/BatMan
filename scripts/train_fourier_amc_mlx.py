from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radioml.metrics import accuracy_report, confusion_matrix
from radioml.numpy_data import (
    ArrayBatcher,
    H5Batcher,
    SplitIndices,
    load_h5_arrays,
    make_stratified_split,
    read_labels_and_snr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fourier-native complex AMC model with MLX.")
    parser.add_argument("--data", type=Path, required=True, help="Path to RadioML 2018.01A HDF5 file.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fourier_complex_mlx"))
    parser.add_argument("--split", type=Path, default=None, help="Optional .npz split file to reuse.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--keep-bins", type=int, default=None)
    parser.add_argument("--low-snr-min", type=int, default=-20)
    parser.add_argument("--low-snr-max", type=int, default=0)
    parser.add_argument("--low-snr-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--profile-batches", action="store_true", help="Print per-batch timing diagnostics.")
    parser.add_argument("--skip-test", action="store_true", help="Stop after validation; useful for hyperparameter sweeps.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Cap training batches per epoch.")
    parser.add_argument("--val-steps", type=int, default=None, help="Cap validation batches per epoch.")
    parser.add_argument("--test-steps", type=int, default=None, help="Cap final test batches.")
    parser.add_argument("--train-metrics", choices=["loss", "full"], default="loss")
    parser.add_argument("--sequential-train-io", action="store_true", help="Read train batches in index order for faster HDF5 IO.")
    parser.add_argument(
        "--cache-data",
        choices=["none", "ram"],
        default="none",
        help="Cache selected train/val/test arrays in RAM after one HDF5 read.",
    )
    return parser.parse_args()


def import_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as exc:
        raise SystemExit(
            "MLX is not installed in this Python environment. Install it with:\n"
            f"{sys.executable} -m pip install mlx"
        ) from exc
    return mx, nn, optim


def configure_device(mx: Any, requested: str) -> str:
    if requested == "cpu":
        mx.set_default_device(mx.cpu)
        return "cpu"
    if requested == "gpu":
        if not mx.metal.is_available():
            raise RuntimeError("You requested --device gpu, but MLX Metal is not available.")
        mx.set_default_device(mx.gpu)
        return "gpu"
    if mx.metal.is_available():
        mx.set_default_device(mx.gpu)
        return "gpu"
    mx.set_default_device(mx.cpu)
    return "cpu"


def limit_indices(indices: np.ndarray, max_items: int | None) -> np.ndarray:
    if max_items is None or len(indices) <= max_items:
        return indices
    return indices[:max_items]


def build_split(args: argparse.Namespace) -> SplitIndices:
    if args.split is not None:
        return SplitIndices.load(args.split)
    labels, snr = read_labels_and_snr(args.data)
    return make_stratified_split(labels, snr, args.train_frac, args.val_frac, args.seed)


def count_parameters(tree: Any) -> int:
    from mlx.utils import tree_flatten

    total = 0
    for _, value in tree_flatten(tree):
        if hasattr(value, "size"):
            total += int(value.size)
    return total


def save_per_snr(path: Path, per_snr_accuracy: dict[int, float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr", "accuracy"])
        for snr, accuracy in sorted(per_snr_accuracy.items()):
            writer.writerow([snr, accuracy])


def json_ready(metrics: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metrics.items():
        if key in {"labels", "preds", "snr"}:
            continue
        if isinstance(value, dict):
            clean[key] = {str(k): float(v) for k, v in value.items()}
        elif isinstance(value, (np.floating, float)):
            clean[key] = float(value)
        else:
            clean[key] = value
    return clean


def args_ready(args: argparse.Namespace) -> dict[str, Any]:
    clean = {}
    for key, value in vars(args).items():
        clean[key] = str(value) if isinstance(value, Path) else value
    return clean


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def array_size_gib(x: np.ndarray, y: np.ndarray, snr: np.ndarray) -> float:
    return (x.nbytes + y.nbytes + snr.nbytes) / (1024**3)


def main() -> None:
    args = parse_args()
    if args.steps_per_epoch is not None and args.sequential_train_io:
        print(
            "Warning: disabling --sequential-train-io because --steps-per-epoch is set. "
            "Sequential capped epochs can repeatedly train on a biased slice of the HDF5 file.",
            flush=True,
        )
        args.sequential_train_io = False
    mx, nn, optim = import_mlx()
    from radioml.mlx_model import FourierComplexAMCMLX

    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = configure_device(mx, args.device)

    split = build_split(args)
    split = SplitIndices(
        train=limit_indices(split.train, args.max_train),
        val=limit_indices(split.val, args.max_val),
        test=limit_indices(split.test, args.max_test),
    )
    split.save(args.output_dir / "split.npz")

    labels, _ = read_labels_and_snr(args.data)
    num_classes = int(labels.max()) + 1
    model = FourierComplexAMCMLX(
        num_classes=num_classes,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        keep_bins=args.keep_bins,
    )
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(model, x, y, snr):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y, reduction="none")
        if args.low_snr_loss_weight != 1.0:
            low_mask = (snr >= args.low_snr_min) & (snr <= args.low_snr_max)
            weights = mx.where(low_mask, args.low_snr_loss_weight, 1.0)
            losses = losses * weights
        return mx.mean(losses)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    cached_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def make_batcher(name: str, indices: np.ndarray, training: bool, epoch_seed: int):
        if args.cache_data == "ram":
            if name not in cached_arrays:
                print(f"Loading {name} subset into RAM once from HDF5...", flush=True)
                cached_arrays[name] = load_h5_arrays(args.data, indices)
                x_cache, y_cache, snr_cache = cached_arrays[name]
                print(
                    f"Cached {name}: {len(y_cache):,} examples, "
                    f"{array_size_gib(x_cache, y_cache, snr_cache):.2f} GiB",
                    flush=True,
                )
            x_cache, y_cache, snr_cache = cached_arrays[name]
            return ArrayBatcher(
                x_cache,
                y_cache,
                snr_cache,
                batch_size=args.batch_size,
                shuffle=training,
                seed=epoch_seed,
            )

        return H5Batcher(
            args.data,
            indices,
            batch_size=args.batch_size,
            shuffle=training,
            seed=epoch_seed,
            sort_for_io=training and args.sequential_train_io,
        )

    def run_epoch(
        name: str,
        indices: np.ndarray,
        training: bool,
        epoch_seed: int,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        if training:
            model.train()
        else:
            model.eval()

        batcher = make_batcher(name, indices, training, epoch_seed)
        total_loss = 0.0
        total_items = 0
        all_labels: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_snr: list[np.ndarray] = []

        phase = "train" if training else "eval"
        total_batches = len(batcher) if max_steps is None else min(len(batcher), max_steps)
        batches = tqdm(batcher, total=total_batches, desc=phase, leave=False)
        for batch_number, (x_np, y_np, snr_np) in enumerate(batches, start=1):
            if max_steps is not None and batch_number > max_steps:
                break
            started = time.perf_counter()
            x = mx.array(x_np)
            y = mx.array(y_np)
            snr = mx.array(snr_np)

            if training:
                loss, grads = loss_and_grad(model, x, y, snr)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
            else:
                loss = loss_fn(model, x, y, snr)
                mx.eval(loss)

            collect_metrics = (not training) or args.train_metrics == "full"
            if collect_metrics:
                logits = model(x)
                preds = mx.argmax(logits, axis=1)
                mx.eval(preds)

            batch_size = len(y_np)
            total_loss += float(np.array(loss)) * batch_size
            total_items += batch_size
            if collect_metrics:
                all_labels.append(y_np)
                all_preds.append(np.array(preds))
                all_snr.append(snr_np)
            batches.set_postfix(loss=f"{total_loss / max(total_items, 1):.4f}")
            if args.profile_batches:
                elapsed = time.perf_counter() - started
                print(f"{phase} batch {batch_number}/{len(batcher)} took {elapsed:.2f}s", flush=True)

        if all_labels:
            labels_np = np.concatenate(all_labels)
            preds_np = np.concatenate(all_preds)
            snr_all = np.concatenate(all_snr)
            report = accuracy_report(labels_np, preds_np, snr_all, args.low_snr_min, args.low_snr_max)
            accuracy = report.accuracy
            low_snr_accuracy = report.low_snr_accuracy
            per_snr_accuracy = report.per_snr_accuracy
        else:
            labels_np = np.asarray([], dtype=np.int64)
            preds_np = np.asarray([], dtype=np.int64)
            snr_all = np.asarray([], dtype=np.int64)
            accuracy = None
            low_snr_accuracy = None
            per_snr_accuracy = {}
        return {
            "loss": total_loss / max(total_items, 1),
            "accuracy": accuracy,
            "low_snr_accuracy": low_snr_accuracy,
            "per_snr_accuracy": per_snr_accuracy,
            "labels": labels_np,
            "preds": preds_np,
            "snr": snr_all,
        }

    params = count_parameters(model.parameters())
    print(f"backend=mlx device={device} classes={num_classes} parameters={params:,}")
    best_val = -1.0
    best_epoch = 0
    history = []
    best_path = args.output_dir / "best.safetensors"
    latest_path = args.output_dir / "latest.safetensors"
    partial_metrics_path = args.output_dir / "metrics_partial.json"

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                "train",
                split.train,
                training=True,
                epoch_seed=args.seed + epoch,
                max_steps=args.steps_per_epoch,
            )
            val_metrics = run_epoch(
                "val",
                split.val,
                training=False,
                epoch_seed=args.seed,
                max_steps=args.val_steps,
            )
            row = {
                "epoch": epoch,
                "train": json_ready(train_metrics),
                "val": json_ready(val_metrics),
                "lr": args.lr,
            }
            history.append(row)
            model.save_weights(str(latest_path))
            train_acc_text = (
                f"{train_metrics['accuracy']:.4f}" if train_metrics["accuracy"] is not None else "not_collected"
            )
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_acc_text} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_low_snr={val_metrics['low_snr_accuracy']:.4f}"
            )
            if val_metrics["low_snr_accuracy"] > best_val:
                best_val = val_metrics["low_snr_accuracy"]
                best_epoch = epoch
                model.save_weights(str(best_path))
            save_json(
                partial_metrics_path,
                {
                    "backend": "mlx",
                    "device": device,
                    "parameters": params,
                    "best_epoch": best_epoch,
                    "best_val_low_snr_accuracy": best_val,
                    "args": args_ready(args),
                    "history": history,
                    "status": "incomplete",
                },
            )
    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. Completed epochs were saved to "
            f"{latest_path}; best checkpoint is {best_path if best_path.exists() else 'not available yet'}."
        )
        save_json(
            partial_metrics_path,
            {
                "backend": "mlx",
                "device": device,
                "parameters": params,
                "best_epoch": best_epoch,
                "best_val_low_snr_accuracy": best_val,
                "args": args_ready(args),
                "history": history,
                "status": "interrupted",
            },
        )
        return

    if args.skip_test:
        metrics = {
            "backend": "mlx",
            "device": device,
            "parameters": params,
            "best_epoch": best_epoch,
            "best_val_low_snr_accuracy": best_val,
            "args": args_ready(args),
            "history": history,
            "status": "validation_complete",
        }
        save_json(args.output_dir / "metrics.json", metrics)
        print(
            "validation complete "
            f"best_epoch={best_epoch} "
            f"best_val_low_snr_accuracy={best_val:.4f}"
        )
        return

    model.load_weights(str(best_path))
    test_metrics = run_epoch("test", split.test, training=False, epoch_seed=args.seed, max_steps=args.test_steps)
    matrix = confusion_matrix(test_metrics["labels"], test_metrics["preds"], num_classes)

    np.save(args.output_dir / "confusion_matrix.npy", matrix)
    save_per_snr(args.output_dir / "per_snr_accuracy.csv", test_metrics["per_snr_accuracy"])

    metrics = {
        "backend": "mlx",
        "device": device,
        "parameters": params,
        "best_epoch": best_epoch,
        "best_val_low_snr_accuracy": best_val,
        "args": args_ready(args),
        "history": history,
        "test": json_ready(test_metrics),
    }
    save_json(args.output_dir / "metrics.json", metrics)

    print(
        "test "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"low_snr_accuracy={test_metrics['low_snr_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
