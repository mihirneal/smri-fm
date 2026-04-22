#!/usr/bin/env python
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from path_utils import resolve_from_repo


def load_brainiac_modules(brainiac_src):
    brainiac_src = resolve_from_repo(brainiac_src)
    if not (brainiac_src / "dataset.py").exists() or not (brainiac_src / "model.py").exists():
        raise SystemExit(
            "Expected BrainIAC src directory with dataset.py and model.py: "
            f"{brainiac_src}"
        )
    sys.path.insert(0, str(brainiac_src))
    from dataset import BrainAgeDataset, get_validation_transform
    from model import Classifier, SingleScanModel, ViTBackboneNet
    return BrainAgeDataset, get_validation_transform, Classifier, SingleScanModel, ViTBackboneNet


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan")
    return mae, rmse, r2


def main():
    parser = argparse.ArgumentParser(
        description="Run BrainIAC brain-age inference on processed NIfTI files."
    )
    parser.add_argument(
        "--brainiac_src",
        default="BrainIAC/src",
        help="Path to the BrainIAC/src directory.",
    )
    parser.add_argument(
        "--input_csv",
        default="DLBS/brainage_100.csv",
        help="CSV with pat_id,label.",
    )
    parser.add_argument(
        "--root_dir",
        default="DLBS/processed_brainiac_100",
        help="Directory containing processed .nii.gz files.",
    )
    parser.add_argument(
        "--output_csv",
        default="DLBS/brainage_100_predictions.csv",
        help="Where to write predictions.",
    )
    parser.add_argument(
        "--metrics_json",
        default="DLBS/brainage_100_metrics.json",
        help="Where to write evaluation metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        default="BrainIAC/src/checkpoints/brainage.ckpt",
        help="Fine-tuned brain-age checkpoint.",
    )
    parser.add_argument(
        "--simclr_checkpoint",
        default="BrainIAC/src/checkpoints/BrainIAC.ckpt",
        help="BrainIAC backbone checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    brainiac_modules = load_brainiac_modules(args.brainiac_src)
    BrainAgeDataset, get_validation_transform, Classifier, SingleScanModel, ViTBackboneNet = (
        brainiac_modules
    )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = BrainAgeDataset(
        csv_path=resolve_from_repo(args.input_csv),
        root_dir=resolve_from_repo(args.root_dir),
        transform=get_validation_transform(image_size=(96, 96, 96)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    checkpoint_path = resolve_from_repo(args.checkpoint)
    simclr_checkpoint_path = resolve_from_repo(args.simclr_checkpoint)

    backbone = ViTBackboneNet(str(simclr_checkpoint_path))
    model = SingleScanModel(backbone, Classifier(d_model=768, num_classes=1))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key[6:] if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Brain-age inference"):
            images = batch["image"].to(device)
            outputs = model(images).flatten()
            predictions.extend(outputs.cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().flatten().tolist())

    input_csv = resolve_from_repo(args.input_csv)
    root_dir = resolve_from_repo(args.root_dir)
    df = pd.read_csv(input_csv)
    df["predicted_value"] = predictions
    df["error"] = df["predicted_value"] - df["label"]
    df["absolute_error"] = df["error"].abs()

    output_csv = resolve_from_repo(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    mae, rmse, r2 = regression_metrics(y_true, y_pred)
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "n": int(len(df)),
        "device": device,
        "input_csv": str(input_csv),
        "root_dir": str(root_dir),
        "checkpoint": str(checkpoint_path),
        "simclr_checkpoint": str(simclr_checkpoint_path),
        "mae_months": mae,
        "mae_years": mae / 12.0,
        "rmse_months": rmse,
        "rmse_years": rmse / 12.0,
        "r2": r2,
        "mean_prediction_months": float(y_pred.mean()),
        "mean_label_months": float(y_true.mean()),
    }

    metrics_json = resolve_from_repo(args.metrics_json)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote predictions to {output_csv}")
    print(f"Wrote metrics to {metrics_json}")
    print(f"n={metrics['n']}")
    print(f"MAE={metrics['mae_months']:.3f} months ({metrics['mae_years']:.3f} years)")
    print(f"RMSE={metrics['rmse_months']:.3f} months ({metrics['rmse_years']:.3f} years)")
    print(f"R2={metrics['r2']:.3f}")
    print(
        f"Prediction mean={metrics['mean_prediction_months']:.3f} months, "
        f"label mean={metrics['mean_label_months']:.3f} months"
    )


if __name__ == "__main__":
    main()
