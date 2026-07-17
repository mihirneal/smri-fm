# MRI Preprocessing

Processes anatomical images into MNI-space outputs. Supports T1w, T2w, and FLAIR inputs.

## Pipeline

1. Rigid registration to TemplateFlow `MNI152NLin2009cAsym` (ANTs)
2. Run SynthSeg on the processed image
3. Save a binary brain mask from the SynthSeg segmentation (`dseg > 0`)

## Outputs

For each input file the pipeline writes:

| File | Description |
|---|---|
| `<input>/processed/*_space-MNI152NLin2009cAsym_desc-processed.nii.gz` | MNI-space processed image |
| `<input>/derivatives/masks/*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz` | Binary brain mask for skull stripping |
| `<input>/derivatives/transforms/*_from-native_to-MNI152NLin2009cAsym_mode-image_xfm.mat` | Rigid transform to MNI |

Brains smaller than the template field of view will be surrounded by zeros. Anatomy that falls outside the template boundary after rigid alignment is clipped.

## SynthSeg Derivatives

The full pipeline also writes SynthSeg derivatives under `<input>/derivatives/synthseg/`:

| File | Description |
|---|---|
| `*_desc-synthseg_dseg.nii.gz` | Tissue segmentation |
| `*_volumes.csv` | Raw SynthSeg volumetric measurements |
| `*_qc.csv` | Raw SynthSeg QC scores |

## Local Runtime

Install the unified environment from the repo root:

```bash
uv sync
```

The pipeline scans `<input>` recursively for `.nii.gz` files and writes
outputs alongside them:

- processed outputs: `<input>/processed/`
- logs: `<input>/logs/`
- derivatives: `<input>/derivatives/`

Run the full pipeline:

```bash
uv run python src/preprocessing/pipeline.py --input <input>
```

Or submit as a Slurm job:

```bash
sbatch scripts/preprocess.sbatch <input>
```

The default SynthSeg backend is:

```bash
uvx --python 3.11 --from 'git+https://github.com/MedARC-AI/SynthSeg.git' SynthSeg
```

## Masked-anatomy pretraining targets

Masked-anatomy pretraining keeps MAE-style patch masking but replaces image
reconstruction with prediction of the SynthSeg label distribution in each
hidden patch. The provided vocabulary contains the 98 non-background hard
labels produced by SynthSeg 2.0 with `--parc`.

After the base sparse-image shards and SynthSeg derivatives exist, create
augmented shards:

```bash
uv run python scripts/add_masked_anatomy_targets.py \
  'datasets/FOMO_with_dwi/shard.{000000..001800}.tar' \
  --output-dir datasets/FOMO_with_dwi_anatomy \
  --source-root /path/to/FOMO300 \
  --patch-size 8 \
  --img-size 208 240 208
```

Each sample receives an `anatomy.npz` member containing compressed
`[num_patches, 98]` voxel counts. Train with:

```bash
uv run python src/smri_mae/main_pretrain.py \
  --cfg-path src/smri_mae/config/masked_anatomy_pretrain.yaml \
  --overrides \
  datasets.fomo_train.url='datasets/FOMO_with_dwi_anatomy/shard.{000000..001620}.tar' \
  datasets.fomo_val.url='datasets/FOMO_with_dwi_anatomy/shard.{001621..001800}.tar'
```
