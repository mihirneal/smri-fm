# AABC/HCPA OOD Evaluation

Curates a representative AABC/HCPA subset and evaluates DLBS-trained tabular brain-age
baselines out of distribution.

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/aabc_ood_eval/scripts/build_aabc_tables.py --n 200

env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/aabc_ood_eval/scripts/run_brainage_transfer.py

env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  experiments/aabc_ood_eval/scripts/run_cognition_transfer.py
```

Defaults use:

- metadata: `/teamspace/studios/this_studio/AABC2_subjects_2026_04_30_10_04_46.csv`
- AABC root: `/teamspace/studios/this_studio/aabc`
- DLBS master: `../smri-dataset/DLBS/derivatives/master_tables/dlbs_master_long.tsv`
- output: `../smri-dataset/AABC/qc/aabc_ood_eval`

The curated subset is stratified by age bin, sex, site, scanner, and wave (`V1`, `V2`, `V3`).
Rows with `age_open="90 or older"` are excluded from metric-bearing outputs because the exact
age is top-coded.

The cognition transfer uses analog targets rather than identical instruments:
`cog_fluid_z -> FluidIQ_Tr35_60y`, `cog_vocabulary_z -> CrystIQ_Tr35_60y`,
`cog_working_memory_z -> tlbx_lswm_uncorrected_standard_score`,
`cog_reasoning_z -> tlbx_dccs_uncorrected_standard_score`, and
`cog_speed_z -> tlbx_pcps_uncorrected_standard_score`.
