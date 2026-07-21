#!/bin/bash
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=96:00:00
#SBATCH --output=/admin/home/mihir.neal/smri-fm/slurms/%x-%j.out
#SBATCH --error=/admin/home/mihir.neal/smri-fm/slurms/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {vitb|vitl_d8}" >&2
  exit 2
fi

readonly ARCHITECTURE=$1
readonly WORKTREE=/admin/home/mihir.neal/smri-fm
readonly PYTHON_BIN="${WORKTREE}/.venv/bin/python"
readonly TORCHRUN_BIN="${WORKTREE}/.venv/bin/torchrun"
readonly SUBJECT_INDEX="${WORKTREE}/checkpoints/joint_anatomy_qc07/fomo_anat_train.subject-index.json"
readonly TRAIN_SHARDS='datasets/FOMO_anat/train/shard.*.tar'
readonly VAL_SHARDS='datasets/FOMO_anat/val/shard.*.tar'
readonly SUBJECT_COUNT=52391
readonly SCAN_COUNT=244656
readonly PRETRAIN_SEED=7338

case "${ARCHITECTURE}" in
  vitb)
    readonly MODEL=joint_mae_anatomy_vit_base
    readonly MODEL_TAG=vitb_d4
    readonly DECODER_DEPTH=4
    readonly TARGET_NORM=patch
    readonly DEFAULT_MICROBATCH=64
    readonly DEFAULT_ACCUM_ITER=1
    ;;
  vitl_d8)
    readonly MODEL=joint_mae_anatomy_vit_large
    readonly MODEL_TAG=vitl_d8
    readonly DECODER_DEPTH=8
    readonly TARGET_NORM=none
    readonly DEFAULT_MICROBATCH=32
    readonly DEFAULT_ACCUM_ITER=2
    ;;
  *)
    echo "architecture must be vitb or vitl_d8, got: ${ARCHITECTURE}" >&2
    exit 2
    ;;
esac

readonly BATCH_SIZE=${MICROBATCH:-${DEFAULT_MICROBATCH}}
readonly ACCUM_ITER=${GRAD_ACCUM:-${DEFAULT_ACCUM_ITER}}
readonly GLOBAL_BATCH=$((BATCH_SIZE * ACCUM_ITER * 4))
if [[ "${GLOBAL_BATCH}" -ne 256 ]]; then
  echo "microbatch ${BATCH_SIZE} and accumulation ${ACCUM_ITER} produce global batch ${GLOBAL_BATCH}, not 256" >&2
  exit 2
fi

readonly RUN_NAME="joint_anat_qc07_${MODEL_TAG}_n${SUBJECT_COUNT}_mb${BATCH_SIZE}_gbs${GLOBAL_BATCH}_s${PRETRAIN_SEED}"
readonly OUTPUT_DIR="${WORKTREE}/checkpoints/joint_anatomy_qc07/${RUN_NAME}"
readonly GPU_LOG="${WORKTREE}/slurms/${RUN_NAME}-${SLURM_JOB_ID}-gpu.csv"

cd "${WORKTREE}"
export PYTHONPATH="${WORKTREE}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" || ! -s "${SUBJECT_INDEX}" ]]; then
  echo "missing training environment or subject index" >&2
  exit 2
fi

"${PYTHON_BIN}" -c '
import sys
from data.mri_data import SubjectBalancedSparseDataset

index_path, expected_subjects, expected_scans = sys.argv[1:]
dataset = SubjectBalancedSparseDataset(
    index_path,
    seed=7338,
    sampling="subject",
    include_anat=True,
)
if dataset.num_subjects != int(expected_subjects):
    raise SystemExit(f"unexpected subject count: {dataset.num_subjects} != {expected_subjects}")
if dataset.num_samples != int(expected_scans):
    raise SystemExit(f"unexpected scan count: {dataset.num_samples} != {expected_scans}")
print(f"preflight: subjects={dataset.num_subjects} scans={dataset.num_samples}")
' "${SUBJECT_INDEX}" "${SUBJECT_COUNT}" "${SCAN_COUNT}"

mkdir -p "${OUTPUT_DIR}" "${WORKTREE}/slurms"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export WANDB_RUN_ID="${RUN_NAME}"
export WANDB_RESUME=allow
mkdir -p "${MPLCONFIGDIR}"

echo "Run:                 ${RUN_NAME}"
echo "Model/decoder:       ${MODEL}/${DECODER_DEPTH}"
echo "Target norm:         ${TARGET_NORM}"
echo "Anatomy loss weight: 0.1"
echo "Subjects/scans:      ${SUBJECT_COUNT}/${SCAN_COUNT}"
echo "Sampling:            subject"
echo "Microbatch/accum:    ${BATCH_SIZE}/${ACCUM_ITER}"
echo "Global batch:        ${GLOBAL_BATCH}"
echo "Samples/epoch:       253440"
echo "Epochs/updates:      100/99000"
echo "Git commit:          $(git rev-parse HEAD)"
echo "Output:              ${OUTPUT_DIR}"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 30 > "${GPU_LOG}" &
readonly NVIDIA_SMI_PID=$!
trap 'kill "${NVIDIA_SMI_PID}" 2>/dev/null || true' EXIT

"${TORCHRUN_BIN}" \
  --nproc-per-node=4 \
  --master-port "${MASTER_PORT}" \
  src/smri_mae/main_pretrain.py \
  --cfg-path src/smri_mae/config/joint_mae_anatomy_pretrain.yaml \
  --overrides \
  name="${RUN_NAME}" \
  notes="joint_mae_anatomy_qc07_fixed_compute_${MODEL_TAG}_lambda0.1" \
  output_dir="${OUTPUT_DIR}" \
  model="${MODEL}" \
  model_kwargs.decoder_depth="${DECODER_DEPTH}" \
  model_kwargs.target_norm="${TARGET_NORM}" \
  model_kwargs.anatomy_loss_weight=0.1 \
  datasets.fomo_train.url="${TRAIN_SHARDS}" \
  datasets.fomo_train.subject_index="${SUBJECT_INDEX}" \
  datasets.fomo_train.sampling=subject \
  datasets.fomo_train.samples_per_epoch=253440 \
  datasets.fomo_train.read_ahead=2048 \
  datasets.fomo_val.url="${VAL_SHARDS}" \
  datasets.fomo_val.samples_per_epoch=25392 \
  batch_size="${BATCH_SIZE}" \
  accum_iter="${ACCUM_ITER}" \
  num_workers=8 \
  prefetch_factor=4 \
  epochs=100 \
  lr=null \
  base_lr=0.001 \
  warmup_epochs=10 \
  checkpoint_period=10 \
  max_checkpoints=0 \
  eval_period=10 \
  seed="${PRETRAIN_SEED}" \
  data_seed="${PRETRAIN_SEED}" \
  eval_seed="${PRETRAIN_SEED}" \
  wandb=true \
  wandb_project=smri-fm \
  auto_resume=true \
  resume=false \
  start_epoch=0 \
  ckpt=null
