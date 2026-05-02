# ADNI OOD Evaluation

Curates ADNI healthy CN and MCI-to-AD prognosis cohorts, then evaluates DLBS-trained
SynthSeg brain-age and cognition baselines out of distribution.

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/adni_ood_eval/scripts/build_adni_tables.py

env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/adni_ood_eval/scripts/run_brainage_transfer.py

env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/adni_ood_eval/scripts/run_cognition_transfer.py

env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/adni_ood_eval/scripts/run_brainage_transfer.py \
  --adni-test ../smri-dataset/ADNI/qc/adni_ood_eval/adni_mci_ad_prognosis_baseline.csv \
  --output-prefix mci_ad_dlbs

env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/adni_ood_eval/scripts/run_prognosis_models.py
```

Defaults use:

- ADNI BIDS root: `/teamspace/studios/this_studio/ADNI-bids`
- diagnosis: `/teamspace/studios/this_studio/DXSUM_01May2026.csv`
- demographics: `/teamspace/studios/this_studio/PTDEMOG_01May2026.csv`
- cognition composites: `/teamspace/studios/this_studio/Neuropsychological-2.zip`
- output: `../smri-dataset/ADNI/qc/adni_ood_eval`

Primary cohort definitions:

- Healthy ADNI is baseline CN with no later observed MCI/AD diagnosis and at least
  12 months observed follow-up.
- Prognosis ADNI is baseline MCI with `mci_to_ad_36mo=1` if AD occurs within
  36 months; non-converters require at least 30 months observed follow-up.
- Diagnosis and cognition are matched to MRI by nearest visit date within 90 days.
- Multiple T1 runs in a subject/session are deduplicated by highest mean SynthSeg QC,
  then lowest run index.

Primary cognition target is `ADNI_EF` from `UWNPSYCHSUM_01May2026.csv`, treated as
ADNI's fluid-like executive-function target. `ADNI_EF2`, `ADNI_MEM`, `ADNI_LAN`, and
`ADNI_VS` are retained in the curated tables for sensitivity analyses.

The default MCI-to-AD prognosis run excludes demographics and evaluates `bag_only`,
`synthseg_only`, and `synthseg_bag`. Demographic variants remain available explicitly
with `--variants demographics_only,demographics_synthseg,demographics_bag,demographics_synthseg_bag`.
