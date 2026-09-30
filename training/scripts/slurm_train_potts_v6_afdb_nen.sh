#!/bin/bash
#SBATCH --job-name=potts_v6_afdb_nen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # ONE TASK PER GPU — srun launches the DDP workers (see note below)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5         # 4 x 5 = 20 <= 24 CPUs on the g6-12xlarge (smallest 4-GPU node)
#SBATCH --mem=150G                # measured MaxRSS ~21 GB/proc x 4 tasks + workers/CUDA (node has 182 GB)
#SBATCH --time=24:00:00
#SBATCH --output=logs/potts_v6_afdb_nen_%j.out
#SBATCH --error=logs/potts_v6_afdb_nen_%j.err

# v6 PottsMPNN on PDB+AFDB (mixed), reading the precomputed EV6 annotation cache. NODE_EDGE_NODE head:
# the Potts coupling head sees concat(h_V[i], h_E[i,k], h_V[j]) (3H) instead of h_E alone (H). The `edge`
# twin is scripts/slurm_train_potts_v6_afdb_edge.sh; everything else is held fixed so the head is the only
# variable. --etab-source node_edge_node sets etab_out's weight SHAPE, so it MUST stay on the command line
# -- a resume without it rebuilds an H-wide head and fails the strict=True load.
#
# To scale to 8 GPUs, swap the four resource lines above for:
#   --ntasks-per-node=8  --gres=gpu:8  --cpus-per-task=11  --mem=600G   (g6-48xlarge: 96 CPU / 747 GB)

# ── Why srun + one task per GPU ───────────────────────────────────────────────
# `sbatch` sets SLURM_NTASKS, so Lightning's SLURMEnvironment fires even for a plain `python` launch,
# reports world_size=SLURM_NTASKS and NEVER spawns the per-GPU workers. So SLURM must create them: `srun`
# with --ntasks-per-node = #GPUs. train.py reads SLURM_PROCID / SLURM_NTASKS first.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="/novo/users/cpjb/PHD/conditional_binding/ph"
VENV="$PROJECT_DIR/.venv"

echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Run:          potts_v6_afdb_nen   (head=node_edge_node, vocab=v6, dataset=afdb_pdb)"
echo "Node:         $SLURMD_NODENAME"
echo "Tasks/GPUs:   ${SLURM_NTASKS} tasks | GPUs: ${CUDA_VISIBLE_DEVICES:-<not set by SLURM>}"
echo "Started:      $(date)"
echo "============================================================"

cd "$PROJECT_DIR"
mkdir -p logs

source "$VENV/bin/activate"

export HBPLUS_PATH="/novo/users/cpjb/tools/hbplus/hbplus"

# Sanity-check hbplus ONCE, before launching the DDP workers (the miss-path annotation still needs it).
echo "Checking hbplus..."
"$HBPLUS_PATH" < /dev/null > /dev/null 2>&1 || { echo "ERROR: hbplus at $HBPLUS_PATH is not functional on this node. Aborting."; exit 1; }
echo "hbplus OK."

# ── run configuration (read by train.py from the environment) ──────────────────
export MPNN_DATASET="afdb_pdb"                                           # PDB experimental + AFDB predictions, mixed by AFDB_FRACTION
export MPNN_PRECOMPUTED_DIR="/novo/users/cpjb/rdd/cpjb/ev6_snapshots" # skip per-epoch EV6 annotation
export MPNN_OUTPUT_DIR="trained_models"                               # ckpts under the big-FS symlink

# This site sets SLURM_EXPORT_ENV=NONE, so srun would strip the environment (no venv PATH, no HBPLUS_PATH,
# none of the MPNN_* above). Restore it for the task steps.
export SLURM_EXPORT_ENV=ALL
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"   # SLURM >= 22.05 no longer propagates it to steps

# --gres on the STEP: this site does NOT propagate the job GRES to srun steps, so without it every task
#   sees 0 GPUs -> Lightning "No supported gpu backend found!". --export=ALL restores the env.
#   --kill-on-bad-exit tears the job down if one rank dies instead of hanging in an NCCL barrier.
# --vocab v6: the 30-token EV6 vocabulary (must match the precomputed labels).
srun --gres=gpu:"${SLURM_GPUS_ON_NODE}" --export=ALL --kill-on-bad-exit=1 \
     "$VENV/bin/python" -m mpnn.train potts_mpnn potts_v6_afdb_nen --vocab v6 --etab-source node_edge_node

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
