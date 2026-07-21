#!/bin/bash
#SBATCH --job-name=joint-anat-bench
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --array=0-3%2
#SBATCH --output=/admin/home/mihir.neal/smri-fm/slurms/joint-anat-bench-%A_%a.out
#SBATCH --error=/admin/home/mihir.neal/smri-fm/slurms/joint-anat-bench-%A_%a.err

set -euo pipefail

readonly WORKTREE=/admin/home/mihir.neal/smri-fm
readonly TORCHRUN_BIN="${WORKTREE}/.venv/bin/torchrun"
readonly SUBJECT_INDEX="${WORKTREE}/checkpoints/joint_anatomy_qc07/fomo_anat_train.subject-index.json"
readonly TRAIN_SHARDS='datasets/FOMO_anat/train/shard.*.tar'
readonly VAL_SHARDS='datasets/FOMO_anat/val/shard.*.tar'

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) readonly MODEL=joint_mae_anatomy_vit_base MODEL_TAG=vitb_d4 BATCH_SIZE=64 ACCUM_ITER=1 DECODER_DEPTH=4 TARGET_NORM=patch ;;
  1) readonly MODEL=joint_mae_anatomy_vit_base MODEL_TAG=vitb_d4 BATCH_SIZE=32 ACCUM_ITER=2 DECODER_DEPTH=4 TARGET_NORM=patch ;;
  2) readonly MODEL=joint_mae_anatomy_vit_large MODEL_TAG=vitl_d8 BATCH_SIZE=64 ACCUM_ITER=1 DECODER_DEPTH=8 TARGET_NORM=none ;;
  3) readonly MODEL=joint_mae_anatomy_vit_large MODEL_TAG=vitl_d8 BATCH_SIZE=32 ACCUM_ITER=2 DECODER_DEPTH=8 TARGET_NORM=none ;;
esac

readonly RUN_NAME="joint_anat_bench_${MODEL_TAG}_mb${BATCH_SIZE}_${SLURM_JOB_ID}"
readonly OUTPUT_DIR="/tmp/${RUN_NAME}"
readonly SAMPLES_PER_EPOCH=10240
readonly GPU_LOG="${WORKTREE}/slurms/${RUN_NAME}-gpu.csv"

cd "${WORKTREE}"
export PYTHONPATH="${WORKTREE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if [[ ! -s "${SUBJECT_INDEX}" ]]; then
  echo "missing subject index: ${SUBJECT_INDEX}" >&2
  exit 2
fi

echo "model=${MODEL} decoder_depth=${DECODER_DEPTH} batch_size=${BATCH_SIZE} accum_iter=${ACCUM_ITER} global_batch=$((BATCH_SIZE * ACCUM_ITER * 4))"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 > "${GPU_LOG}" &
readonly NVIDIA_SMI_PID=$!
trap 'kill "${NVIDIA_SMI_PID}" 2>/dev/null || true' EXIT

"${TORCHRUN_BIN}" \
  --nproc-per-node=4 \
  --master-port "${MASTER_PORT}" \
  src/smri_mae/main_pretrain.py \
  --cfg-path src/smri_mae/config/joint_mae_anatomy_pretrain.yaml \
  --overrides \
  name="${RUN_NAME}" \
  notes=joint_anatomy_h100_microbatch_benchmark \
  output_dir="${OUTPUT_DIR}" \
  model="${MODEL}" \
  model_kwargs.decoder_depth="${DECODER_DEPTH}" \
  model_kwargs.target_norm="${TARGET_NORM}" \
  datasets.fomo_train.url="${TRAIN_SHARDS}" \
  datasets.fomo_train.subject_index="${SUBJECT_INDEX}" \
  datasets.fomo_train.samples_per_epoch="${SAMPLES_PER_EPOCH}" \
  datasets.fomo_val.url="${VAL_SHARDS}" \
  datasets.fomo_val.samples_per_epoch=4 \
  batch_size="${BATCH_SIZE}" \
  accum_iter="${ACCUM_ITER}" \
  num_workers=8 \
  prefetch_factor=4 \
  epochs=1 \
  warmup_epochs=0 \
  checkpoint_period=999 \
  max_checkpoints=1 \
  eval_period=0 \
  wandb=false \
  auto_resume=false \
  resume=false \
  print_freq=10
