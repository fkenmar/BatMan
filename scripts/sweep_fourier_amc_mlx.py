from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radioml.numpy_data import make_stratified_split, read_labels_and_snr


TRIALS: list[dict[str, Any]] = [
    {"name": "baseline", "width": 64, "depth": 6, "lr": 2e-3, "dropout": 0.1, "keep_bins": None, "low_snr_loss_weight": 1.0},
    {"name": "small", "width": 32, "depth": 4, "lr": 2e-3, "dropout": 0.1, "keep_bins": None, "low_snr_loss_weight": 1.0},
    {"name": "low_snr_weighted", "width": 64, "depth": 6, "lr": 2e-3, "dropout": 0.1, "keep_bins": None, "low_snr_loss_weight": 2.0},
    {"name": "center_bins_512", "width": 64, "depth": 6, "lr": 2e-3, "dropout": 0.1, "keep_bins": 512, "low_snr_loss_weight": 1.0},
    {"name": "wider", "width": 96, "depth": 6, "lr": 1e-3, "dropout": 0.1, "keep_bins": None, "low_snr_loss_weight": 1.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only hyperparameter sweep for the MLX AMC model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sweeps/mlx_low_snr"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--profile-batches", action="store_true")
    parser.add_argument("--cache-data", choices=["none", "ram"], default="none")
    parser.add_argument("--trial", choices=[trial["name"] for trial in TRIALS], action="append")
    parser.add_argument("--skip-completed", action="store_true", help="Do not rerun trials that already have metrics.json.")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild sweep_results.csv from completed trial folders.")
    return parser.parse_args()


def trial_args(trial: dict[str, Any]) -> list[str]:
    args = [
        "--width",
        str(trial["width"]),
        "--depth",
        str(trial["depth"]),
        "--lr",
        str(trial["lr"]),
        "--dropout",
        str(trial["dropout"]),
        "--low-snr-loss-weight",
        str(trial["low_snr_loss_weight"]),
    ]
    if trial["keep_bins"] is not None:
        args.extend(["--keep-bins", str(trial["keep_bins"])])
    return args


def read_score(metrics_path: Path) -> tuple[float, float, int]:
    metrics = json.loads(metrics_path.read_text())
    history = metrics.get("history", [])
    if not history:
        return -1.0, -1.0, 0
    best = max(history, key=lambda row: row["val"]["low_snr_accuracy"])
    return (
        float(best["val"]["low_snr_accuracy"]),
        float(best["val"]["accuracy"]),
        int(best["epoch"]),
    )


def row_for_trial(output_dir: Path, trial: dict[str, Any]) -> dict[str, Any] | None:
    metrics_path = output_dir / trial["name"] / "metrics.json"
    if not metrics_path.exists():
        return None
    low_snr, val_acc, best_epoch = read_score(metrics_path)
    return {
        "trial": trial["name"],
        "best_val_low_snr_accuracy": low_snr,
        "best_val_accuracy": val_acc,
        "best_epoch": best_epoch,
        **{key: value for key, value in trial.items() if key != "name"},
    }


def write_summary(output_dir: Path) -> list[dict[str, Any]]:
    rows = [row for trial in TRIALS if (row := row_for_trial(output_dir, trial)) is not None]
    rows.sort(key=lambda row: row["best_val_low_snr_accuracy"], reverse=True)
    summary_path = output_dir / "sweep_results.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "trial",
            "best_val_low_snr_accuracy",
            "best_val_accuracy",
            "best_epoch",
            "width",
            "depth",
            "lr",
            "dropout",
            "keep_bins",
            "low_snr_loss_weight",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote sweep summary: {summary_path}")
    if rows:
        best = rows[0]
        print(
            "best "
            f"trial={best['trial']} "
            f"val_low_snr={best['best_val_low_snr_accuracy']:.4f} "
            f"val_acc={best['best_val_accuracy']:.4f}"
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        write_summary(args.output_dir)
        return

    split_path = args.output_dir / "split.npz"
    if not split_path.exists():
        labels, snr = read_labels_and_snr(args.data)
        split = make_stratified_split(labels, snr, args.train_frac, args.val_frac, args.seed)
        split.save(split_path)
    else:
        print(f"Reusing existing split: {split_path}")

    selected = [trial for trial in TRIALS if args.trial is None or trial["name"] in args.trial]
    for trial in selected:
        trial_dir = args.output_dir / trial["name"]
        if args.skip_completed and (trial_dir / "metrics.json").exists():
            print(f"\n=== trial: {trial['name']} already complete; skipping ===", flush=True)
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_fourier_amc_mlx.py"),
            "--data",
            str(args.data),
            "--split",
            str(split_path),
            "--output-dir",
            str(trial_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--skip-test",
        ]
        if args.max_train is not None:
            cmd.extend(["--max-train", str(args.max_train)])
        if args.max_val is not None:
            cmd.extend(["--max-val", str(args.max_val)])
        if args.profile_batches:
            cmd.append("--profile-batches")
        if args.cache_data != "none":
            cmd.extend(["--cache-data", args.cache_data])
        cmd.extend(trial_args(trial))

        print(f"\n=== trial: {trial['name']} ===", flush=True)
        subprocess.run(cmd, check=True)
        write_summary(args.output_dir)

    write_summary(args.output_dir)


if __name__ == "__main__":
    main()
