#!/bin/bash
#SBATCH --job-name=hbond_head
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/hbond_head_%j.out
#SBATCH --error=logs/hbond_head_%j.err
#
# Train the H-bond / salt-bridge head on a FROZEN PottsMPNN (1 GPU), ONE 6 h job.
# Everything but the head is frozen. Single-GPU plain-python launch (NO srun): this cluster's
# srun gives tasks a broken PATH and 0 visible GPUs, so multi-GPU is parked. The head is tiny;
# the bottleneck is HBPLUS featurization (NUM_WORKERS parallel workers).
#
# RESUME: train_hbond.py builds CkptConfig(path=ckpt_dir) when ckpt_dir exists, so this job
# auto-resumes from the LATEST checkpoint (weights + optimizer + epoch). EPOCHS=100 is a high
# ceiling so a single job uses the full 6 h wall time. To continue later, just resubmit this
# script manually. NO self-resubmit chaining — nothing is queued automatically.
#
# Other run config (SUBSET, parquets, encoder ckpt, LR, NUM_WORKERS, ...) lives in the CONFIG
# block at the top of foundry/models/mpnn/src/mpnn/train_hbond.py.
#
#   sbatch scripts/submit_train_hbond_head.sh          # 6 h job (resumes from latest ckpt if any)
#   EPOCHS=50 sbatch scripts/submit_train_hbond_head.sh

PROJECT_DIR="/novo/users/cpjb/PHD/conditional_binding/ph"
VENV="$PROJECT_DIR/.venv"

set -euo pipefail

export EPOCHS="${EPOCHS:-100}"          # target total epochs (read by train_hbond.py); ceiling, not a chain

cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================================"
echo "Job ID:   ${SLURM_JOB_ID:-<interactive>}"
echo "Node:     ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<not set by SLURM>}"
echo "EPOCHS:   $EPOCHS  (single 6 h job; resumes from latest ckpt; no auto-resubmit)"
echo "Started:  $(date)"
echo "============================================================"

source "$VENV/bin/activate"
export HBPLUS_PATH="/novo/users/cpjb/tools/hbplus/hbplus"

# Sanity-check hbplus before launching (it is re-run on every structure each epoch).
echo "Checking hbplus..."
"$HBPLUS_PATH" < /dev/null > /dev/null 2>&1 || { echo "ERROR: hbplus at $HBPLUS_PATH is not functional on this node. Aborting."; exit 1; }
echo "hbplus OK."

"$VENV/bin/python" -m mpnn.train_hbond

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
