# MRI Preprocessing

Preprocesses BIDS anatomical images and writes MNI-space derivatives. Supports T1w, T2w, and FLAIR inputs.

## Pipeline

1. Reorient to RAS
2. Resample to 1 mm isotropic when needed (ANTs B-spline interpolation)
3. Run SynthSeg on the 1 mm image
4. Rigid registration to TemplateFlow `MNI152NLin2009cAsym` (ANTs)

## Outputs

For each input file the pipeline writes three derivatives (BIDS-compliant naming):

| File | Description |
|---|---|
| `*_space-MNI152NLin2009cAsym_desc-preproc_{suffix}.nii.gz` | MNI-space preprocessed image |
| `*_space-MNI152NLin2009cAsym_desc-brain_mask_{suffix}.nii.gz` | MNI-space brain mask |
| `*_from-native_to-MNI152NLin2009cAsym_mode-image_desc-{suffix}_xfm.mat` | Rigid transform to MNI |

Brains smaller than the template field of view will be surrounded by zeros. Anatomy that falls outside the template boundary after rigid alignment is clipped.

## SynthSeg Derivatives

Normal preprocessing runs also write SynthSeg derivatives under
`preprocessing/data/processed/<dataset>/derivatives/synthseg/`:

| File | Description |
|---|---|
| `*_desc-synthseg_dseg.nii.gz` | Tissue segmentation |
| `*_volumes.csv` | Raw SynthSeg volumetric measurements |
| `*_qc.csv` | Raw SynthSeg QC scores |

Pass `--synthseg` to run only this SynthSeg stage on RAS-canonicalized raw inputs and skip MNI preprocessing outputs.

## Local Runtime

Use `uv` from the repo root:

```bash
uv sync --extra preprocessing
```

The pipeline expects dataset-scoped directories under `preprocessing/data/`:

- raw inputs: `preprocessing/data/raw/<dataset>/`
- preprocessed outputs: `preprocessing/data/processed/<dataset>/`
- logs: `preprocessing/data/logs/<dataset>/`
- SynthSeg outputs: `preprocessing/data/processed/<dataset>/derivatives/synthseg/`

Run preprocessing:

```bash
uv run preprocessing/pipeline.py --dataset <dataset>
```

SynthSeg-only mode:

```bash
uv run preprocessing/pipeline.py --dataset <dataset> --synthseg
```

The default SynthSeg backend is:

```bash
uvx --python 3.11 --from 'git+https://github.com/MedARC-AI/SynthSeg.git' SynthSeg
```
