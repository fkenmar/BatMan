from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radioml.data import RadioMLDataset, SplitIndices, make_stratified_split, read_labels_and_snr
from radioml.metrics import accuracy_report, confusion_matrix
from radioml.model import FourierComplexAMC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fourier-native complex AMC model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to RadioML 2018.01A HDF5 file.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fourier_complex"))
    parser.add_argument("--split", type=Path, default=None, help="Optional .npz split file to reuse.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
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
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "mps":
            try:
                torch.empty(1, device=device)
            except Exception as exc:
                raise RuntimeError(
                    "You requested --device mps, but PyTorch could not create an MPS tensor. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
            return torch.device("mps")
        except Exception as exc:
            print(f"MPS was reported available but failed tensor creation; falling back to CPU: {exc}")
    elif torch.backends.mps.is_built():
        print("MPS is built into PyTorch, but torch.backends.mps.is_available() is False; using CPU.")
    else:
        print("This PyTorch build does not include MPS support; using CPU.")
    return torch.device("cpu")


def limit_indices(indices: np.ndarray, max_items: int | None) -> np.ndarray:
    if max_items is None or len(indices) <= max_items:
        return indices
    return indices[:max_items]


def build_split(args: argparse.Namespace) -> SplitIndices:
    if args.split is not None:
        return SplitIndices.load(args.split)
    labels, snr = read_labels_and_snr(args.data)
    return make_stratified_split(
        labels=labels,
        snr=snr,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )


def make_loader(
    data_path: Path,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = RadioMLDataset(data_path, indices=indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def verify_device_for_model(model: nn.Module, device: torch.device, requested: str) -> torch.device:
    if device.type != "mps":
        return device
    try:
        model.train()
        dummy = torch.zeros(2, 1024, 2, device=device)
        loss = model(dummy).square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        return device
    except Exception as exc:
        if requested == "mps":
            raise RuntimeError(
                "The model could not run its complex FFT path on MPS. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        print(f"MPS failed the model smoke test; falling back to CPU: {exc}")
        model.to("cpu")
        model.zero_grad(set_to_none=True)
        return torch.device("cpu")


def weighted_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    snr: torch.Tensor,
    low_snr_min: int,
    low_snr_max: int,
    low_snr_weight: float,
) -> torch.Tensor:
    losses = criterion(logits, labels)
    if low_snr_weight != 1.0:
        low_mask = (snr >= low_snr_min) & (snr <= low_snr_max)
        weights = torch.ones_like(losses)
        weights = torch.where(low_mask, weights * low_snr_weight, weights)
        losses = losses * weights
    return losses.mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_snr: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, leave=False, disable=not training):
            iq = batch["iq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            snr = batch["snr"].to(device, non_blocking=True)

            logits = model(iq)
            loss = weighted_loss(
                criterion,
                logits,
                labels,
                snr,
                args.low_snr_min,
                args.low_snr_max,
                args.low_snr_loss_weight,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            batch_size = labels.numel()
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_snr.append(snr.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_preds)
    snr_np = np.concatenate(all_snr)
    report = accuracy_report(labels_np, preds_np, snr_np, args.low_snr_min, args.low_snr_max)

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": report.accuracy,
        "low_snr_accuracy": report.low_snr_accuracy,
        "per_snr_accuracy": report.per_snr_accuracy,
        "labels": labels_np,
        "preds": preds_np,
        "snr": snr_np,
    }


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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = build_split(args)
    split = SplitIndices(
        train=limit_indices(split.train, args.max_train),
        val=limit_indices(split.val, args.max_val),
        test=limit_indices(split.test, args.max_test),
    )
    split.save(args.output_dir / "split.npz")

    train_loader = make_loader(args.data, split.train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(args.data, split.val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(args.data, split.test, args.batch_size, False, args.num_workers)

    device = choose_device(args.device)
    num_classes = train_loader.dataset.num_classes
    model = FourierComplexAMC(
        num_classes=num_classes,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        keep_bins=args.keep_bins,
    ).to(device)
    device = verify_device_for_model(model, device, args.device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print(f"device={device} classes={num_classes} parameters={model.count_parameters():,}")
    best_val = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, args)
        val_metrics = run_epoch(model, val_loader, device, criterion, None, args)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train": json_ready(train_metrics),
            "val": json_ready(val_metrics),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_low_snr={val_metrics['low_snr_accuracy']:.4f}"
        )

        if val_metrics["low_snr_accuracy"] > best_val:
            best_val = val_metrics["low_snr_accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_args": {
                        "num_classes": num_classes,
                        "width": args.width,
                        "depth": args.depth,
                        "dropout": args.dropout,
                        "keep_bins": args.keep_bins,
                    },
                    "args": args_ready(args),
                    "epoch": epoch,
                    "val_low_snr_accuracy": best_val,
                    "parameters": model.count_parameters(),
                },
                args.output_dir / "best.pt",
            )

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, device, criterion, None, args)
    matrix = confusion_matrix(test_metrics["labels"], test_metrics["preds"], num_classes)

    np.save(args.output_dir / "confusion_matrix.npy", matrix)
    save_per_snr(args.output_dir / "per_snr_accuracy.csv", test_metrics["per_snr_accuracy"])

    metrics = {
        "parameters": model.count_parameters(),
        "best_epoch": checkpoint["epoch"],
        "history": history,
        "test": json_ready(test_metrics),
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        "test "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"low_snr_accuracy={test_metrics['low_snr_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
