#!/usr/bin/env python
"""Subject-grouped RidgeCV baseline for DLBS age from SynthSeg regional volumes.

Writes out-of-fold predictions, per-fold and overall metrics, full-fit standardized
coefficients, and a summary figure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from path_utils import resolve_from_repo


WAVE_TO_AGE = {"wave1": "AgeMRI_W1", "wave2": "AgeMRI_W2", "wave3": "AgeMRI_W3"}
VOLUME_SUFFIX = "_desc-synthseg_volumes.tsv"
FILENAME_RE = re.compile(
    r"(?P<participant>sub-[^_]+)_ses-(?P<wave>wave[123])_"
    r"(?P<scan>.+?)_space-[^_]+_desc-preproc_T1w_desc-synthseg_volumes\.tsv$"
)
WAVE_COLORS = {"wave1": "#16697a", "wave2": "#d1495b", "wave3": "#edae49"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthseg-dir", default="DLBS/synthseg")
    p.add_argument("--participants", default="DLBS/participants.tsv")
    p.add_argument("--output-dir", default="DLBS/qc/synthseg_ridge_simple")
    p.add_argument(
        "--feature-set",
        choices=["icv-normalized", "raw"],
        default="icv-normalized",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--alphas",
        default="0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,10000",
    )
    p.add_argument(
        "--min-qc",
        type=float,
        default=None,
        help="Drop scans whose SynthSeg min per-region QC score is below this.",
    )
    return p.parse_args()


def load_scans(synthseg_dir: Path, participants: pd.DataFrame, min_qc: float | None) -> pd.DataFrame:
    volume_frames: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for path in sorted(synthseg_dir.glob(f"sub-*/ses-*/anat/*{VOLUME_SUFFIX}")):
        match = FILENAME_RE.match(path.name)
        if match is None:
            continue
        participant_id = match.group("participant")
        wave = match.group("wave")
        scan = match.group("scan")

        if participant_id not in participants.index:
            continue
        age = participants.at[participant_id, WAVE_TO_AGE[wave]]
        if pd.isna(age):
            continue

        qc_path = path.with_name(path.name.replace("_volumes.tsv", "_qc.tsv"))
        qc_min = qc_mean = np.nan
        if qc_path.exists():
            qc = pd.read_csv(qc_path, sep="\t")
            if not qc.empty:
                qc_min = float(qc["qc_score"].min())
                qc_mean = float(qc["qc_score"].mean())
        if min_qc is not None and not np.isnan(qc_min) and qc_min < min_qc:
            continue

        vols = pd.read_csv(path, sep="\t").set_index("region")["volume_mm3"]
        if "total intracranial" not in vols.index:
            continue
        vols = vols.rename(lambda r: f"volume__{r}")
        pat_id = f"{participant_id}_ses-{wave}_{scan}_T1w"
        volume_frames.append(vols.to_frame(name=pat_id).T)
        meta_rows.append(
            {
                "pat_id": pat_id,
                "participant_id": participant_id,
                "wave": wave,
                "scan": scan,
                "age": float(age),
                "synthseg_qc_min": qc_min,
                "synthseg_qc_mean": qc_mean,
            }
        )

    if not meta_rows:
        raise SystemExit("No usable SynthSeg rows found.")

    meta = pd.DataFrame(meta_rows).set_index("pat_id")
    volumes = pd.concat(volume_frames).dropna(axis=1)
    joined = meta.join(volumes, how="inner")
    joined.index.name = "pat_id"
    return joined


def build_features(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.Series]:
    volume_cols = [c for c in df.columns if c.startswith("volume__")]
    tiv_col = "volume__total intracranial"

    if feature_set == "icv-normalized":
        region_cols = [c for c in volume_cols if c != tiv_col]
        x = df[region_cols].div(df[tiv_col], axis=0)
        x.columns = [c.replace("volume__", "").strip() + " / TIV" for c in x.columns]
    else:
        x = df[volume_cols].copy()
        x.columns = [c.replace("volume__", "") + " (mm^3)" for c in x.columns]

    return x, df["age"].astype(float)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "bias": float(np.mean(err)),
        "n": int(len(y_true)),
    }


def make_figure(
    meta: pd.DataFrame,
    y: np.ndarray,
    y_pred: np.ndarray,
    fold_mae: dict[int, float],
    fold_alpha: dict[int, float],
    coef_series: pd.Series,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    residuals = y_pred - y

    ax = axes[0, 0]
    for wave, color in WAVE_COLORS.items():
        mask = meta["wave"].to_numpy() == wave
        ax.scatter(y[mask], y_pred[mask], s=18, alpha=0.6, color=color, label=wave)
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title("Predicted vs true (out-of-fold)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    colors = meta["wave"].map(WAVE_COLORS).fillna("#333")
    ax.scatter(y, residuals, s=18, alpha=0.6, c=colors)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("True age (years)")
    ax.set_ylabel("Residual: predicted - true")
    ax.set_title("Residuals vs true age")

    ax = axes[0, 2]
    bins = np.linspace(residuals.min(), residuals.max(), 30)
    ax.hist(residuals, bins=bins, color="#2f3e46", alpha=0.8)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Residual (years)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution (bias={residuals.mean():+.2f})")

    ax = axes[1, 0]
    waves = ["wave1", "wave2", "wave3"]
    data = [np.abs(residuals[meta["wave"].to_numpy() == w]) for w in waves]
    bp = ax.boxplot(data, tick_labels=waves, patch_artist=True, widths=0.55)
    for patch, wave in zip(bp["boxes"], waves):
        patch.set_facecolor(WAVE_COLORS[wave])
        patch.set_alpha(0.55)
    ax.set_ylabel("Absolute error (years)")
    ax.set_title("Absolute error by wave")

    ax = axes[1, 1]
    fold_ids = sorted(fold_mae)
    maes = [fold_mae[f] for f in fold_ids]
    alphas = [fold_alpha[f] for f in fold_ids]
    bars = ax.bar(fold_ids, maes, color="#16697a", alpha=0.8)
    for bar, alpha_val, mae_val in zip(bars, alphas, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"α={alpha_val:g}\n{mae_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Per-fold MAE and selected alpha")
    ax.set_ylim(0, max(maes) * 1.25)

    ax = axes[1, 2]
    top = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index).head(15)[::-1]
    bar_colors = ["#0f766e" if v >= 0 else "#b42318" for v in top.values]
    ax.barh(range(len(top)), top.values, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([name[:40] for name in top.index], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Standardized ridge coefficient (full-data refit)")
    ax.set_title("Top 15 features by |coef|")

    fig.suptitle("DLBS SynthSeg ridge age baseline", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    synthseg_dir = resolve_from_repo(args.synthseg_dir)
    participants_path = resolve_from_repo(args.participants)
    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = np.asarray([float(v) for v in args.alphas.split(",") if v.strip()])

    participants = pd.read_csv(participants_path, sep="\t").set_index("participant_id")
    df = load_scans(synthseg_dir, participants, args.min_qc)
    x_df, y_series = build_features(df, args.feature_set)
    x = x_df.to_numpy()
    y = y_series.to_numpy()
    groups = df["participant_id"].to_numpy()

    n_groups = len(np.unique(groups))
    folds = min(args.folds, n_groups)
    cv = GroupKFold(n_splits=folds)

    def make_pipe() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=alphas, cv=folds)),
            ]
        )

    y_pred = cross_val_predict(make_pipe(), x, y, groups=groups, cv=cv, n_jobs=1)

    fold_numbers = np.full(len(y), -1, dtype=int)
    fold_mae: dict[int, float] = {}
    fold_alpha: dict[int, float] = {}
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(x, y, groups=groups), start=1):
        fold_numbers[test_idx] = fold_index
        sub_pipe = make_pipe()
        sub_pipe.fit(x[train_idx], y[train_idx])
        fold_alpha[fold_index] = float(sub_pipe.named_steps["ridge"].alpha_)
        fold_mae[fold_index] = float(np.mean(np.abs(sub_pipe.predict(x[test_idx]) - y[test_idx])))

    final_pipe = make_pipe()
    final_pipe.fit(x, y)
    final_alpha = float(final_pipe.named_steps["ridge"].alpha_)
    coef_series = pd.Series(
        final_pipe.named_steps["ridge"].coef_,
        index=x_df.columns,
        name="standardized_coefficient",
    )

    preds_df = df.reset_index()[
        ["pat_id", "participant_id", "wave", "scan", "age", "synthseg_qc_min", "synthseg_qc_mean"]
    ].copy()
    preds_df["predicted_age"] = y_pred
    preds_df["residual"] = y_pred - y
    preds_df["absolute_error"] = np.abs(preds_df["residual"])
    preds_df["outer_fold"] = fold_numbers
    preds_df["alpha"] = preds_df["outer_fold"].map(fold_alpha)
    preds_df.to_csv(output_dir / "predictions.csv", index=False)

    coef_df = (
        coef_series.sort_values(key=lambda s: s.abs(), ascending=False)
        .rename_axis("feature")
        .reset_index()
    )
    coef_df.to_csv(output_dir / "coefficients.csv", index=False)

    report = {
        "settings": {
            "synthseg_dir": str(synthseg_dir),
            "participants": str(participants_path),
            "feature_set": args.feature_set,
            "folds": folds,
            "alphas": alphas.tolist(),
            "min_qc": args.min_qc,
        },
        "data": {
            "n_scans": int(len(y)),
            "n_subjects": int(n_groups),
            "n_features": int(x.shape[1]),
            "age_min": float(y.min()),
            "age_max": float(y.max()),
            "age_mean": float(y.mean()),
        },
        "group_cv": metrics(y, y_pred),
        "mean_age_baseline": metrics(y, np.full_like(y, y.mean())),
        "by_wave": {
            wave: metrics(y[df["wave"].to_numpy() == wave], y_pred[df["wave"].to_numpy() == wave])
            for wave in sorted(df["wave"].unique())
        },
        "per_fold": [
            {"fold": f, "alpha": fold_alpha[f], "mae": fold_mae[f]} for f in sorted(fold_mae)
        ],
        "final_refit_alpha": final_alpha,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    make_figure(df, y, y_pred, fold_mae, fold_alpha, coef_series, output_dir / "figure.png")

    cv_metrics = report["group_cv"]
    print(
        f"scans={report['data']['n_scans']} subjects={report['data']['n_subjects']} "
        f"features={x.shape[1]}"
    )
    print(
        f"MAE={cv_metrics['mae']:.3f}y  RMSE={cv_metrics['rmse']:.3f}y  "
        f"R2={cv_metrics['r2']:.3f}  r={cv_metrics['pearson_r']:.3f}  bias={cv_metrics['bias']:+.3f}"
    )
    print(f"baseline MAE={report['mean_age_baseline']['mae']:.3f}y  final_alpha={final_alpha:g}")
    print(f"wrote {output_dir}/predictions.csv, coefficients.csv, metrics.json, figure.png")


if __name__ == "__main__":
    main()
