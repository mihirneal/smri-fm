# BrainIAC DLBS Brain-Age Evaluation

Running BrainIAC brain-age inference on DLBS.

Expected local layout from the `smri-fm` repo root:

```text
BrainIAC/        # local BrainIAC checkout with checkpoints
DLBS/            # local DLBS data
experiments/brainiac_dlbs_eval/
```

`BrainIAC/` and `DLBS/` are local inputs/outputs and are ignored by git.

## 1. Select 100 DLBS Images

This creates symlinks and a manifest:

```bash
uv run python experiments/brainiac_dlbs_eval/scripts/create_dlbs_subset.py --n 100 --overwrite
```

Defaults:

```text
input:    DLBS/images
output:   DLBS/images_100
manifest: DLBS/images_100_manifest.csv
```

## 2. Preprocess With BrainIAC

Run this with the BrainIAC environment:

```bash
python BrainIAC/src/preprocessing/mri_preprocess_3d_simple.py \
  --temp_img BrainIAC/src/preprocessing/atlases/temp_head.nii.gz \
  --input_dir DLBS/images_100 \
  --output_dir DLBS/processed_brainiac_100
```

## 3. Create Brain-Age CSV

This maps processed filenames to DLBS `AgeMRI_W1/W2/W3` labels in months:

```bash
uv run python experiments/brainiac_dlbs_eval/scripts/make_dlbs_brainage_csv.py
```

Defaults:

```text
processed images: DLBS/processed_brainiac_100
participants:     DLBS/participants.tsv
output CSV:       DLBS/brainage_100.csv
```

## 4. Run Brain-Age Inference

Run this with the BrainIAC environment:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python \
  experiments/brainiac_dlbs_eval/scripts/run_brainage_inference.py
```

Defaults:

```text
BrainIAC src:      BrainIAC/src
input CSV:         DLBS/brainage_100.csv
processed images:  DLBS/processed_brainiac_100
predictions CSV:   DLBS/brainage_100_predictions.csv
metrics JSON:      DLBS/brainage_100_metrics.json
brain-age ckpt:    BrainIAC/src/checkpoints/brainage.ckpt
BrainIAC ckpt:     BrainIAC/src/checkpoints/BrainIAC.ckpt
```

## Notes

- The inference script imports `dataset.py` and `model.py` from `BrainIAC/src`.
- It does not vendor BrainIAC code.
