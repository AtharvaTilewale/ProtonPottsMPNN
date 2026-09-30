#!/bin/bash
#SBATCH --job-name=hbond_head_v6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # ONE TASK PER GPU — srun launches the DDP workers (see note below)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5         # 4 x 5 = 20 <= 24 CPUs on the g6-12xlarge (smallest 4-GPU node)
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hbond_head_v6_%j.out
#SBATCH --error=logs/hbond_head_v6_%j.err

# Train the H-bond / salt-bridge head (frozen v6 edge-coupling PottsMPNN encoder), reading the precomputed
# EV6 annotation cache. All run config (encoder ckpt, EXTENDED_VOCAB=v6, DATASET=afdb_pdb, head width,
# scope) lives in mpnn/train_hbond.py's config block; this script only sets the environment + launches DDP.
#
# To scale to 8 GPUs, swap the four resource lines for:
#   --ntasks-per-node=8  --gres=gpu:8  --cpus-per-task=11  --mem=600G   (g6-48xlarge: 96 CPU / 747 GB)

# ── Why srun + one task per GPU ───────────────────────────────────────────────
# `sbatch` sets SLURM_NTASKS, so Lightning's SLURMEnvironment fires even for a plain `python` launch and
# NEVER spawns the per-GPU workers. So SLURM must create them: `srun` with --ntasks-per-node = #GPUs.
# train_hbond.py reads SLURM_PROCID / SLURM_NTASKS first.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="/novo/users/cpjb/PHD/conditional_binding/ph"
VENV="$PROJECT_DIR/.venv"

echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Run:          hbond_head_v6_afdb_edge   (frozen encoder=potts_v6_afdb_edge, vocab=v6, dataset=afdb_pdb)"
echo "Node:         $SLURMD_NODENAME"
echo "Tasks/GPUs:   ${SLURM_NTASKS} tasks | GPUs: ${CUDA_VISIBLE_DEVICES:-<not set by SLURM>}"
echo "Started:      $(date)"
echo "============================================================"

cd "$PROJECT_DIR"
mkdir -p logs

source "$VENV/bin/activate"

# HBPLUS is not needed on cache hits (labels come from the snapshot's hbond_pairs/salt_pairs; a miss draws
# another cached example rather than annotating live), but set it so nothing surprises on an edge case.
export HBPLUS_PATH="/novo/users/cpjb/tools/hbplus/hbplus"

# ── run configuration (read by train_hbond.py from the environment) ─────────────
export MPNN_PRECOMPUTED_DIR="/novo/users/cpjb/rdd/cpjb/ev6_snapshots"   # skip per-epoch EV6/HBPLUS annotation
export MPNN_OUTPUT_DIR="trained_models"                                 # ckpts under the big-FS symlink
export EPOCHS="${EPOCHS:-100}"
export NUM_WORKERS="${NUM_WORKERS:-4}"                                  # <= --cpus-per-task

# This site sets SLURM_EXPORT_ENV=NONE, so srun would strip the environment. Restore it for the task steps.
export SLURM_EXPORT_ENV=ALL
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

# --gres on the STEP: this site does NOT propagate the job GRES to srun steps. --export=ALL restores the env;
# --kill-on-bad-exit tears the job down if one rank dies instead of hanging in an NCCL barrier.
srun --gres=gpu:"${SLURM_GPUS_ON_NODE}" --export=ALL --kill-on-bad-exit=1 \
     "$VENV/bin/python" -m mpnn.train_hbond

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
