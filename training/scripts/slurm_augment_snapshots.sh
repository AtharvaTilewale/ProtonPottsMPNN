#!/bin/bash
#SBATCH --job-name=ev6_augment
#SBATCH --partition=highmem        # same atomworks/biotite footprint as the build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # EV6's fold ensembles thread over these
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --array=0-99               # 100 shards over the 256 top-level hash dirs
#SBATCH --output=logs/ev6_augment_%A_%a.out
#SBATCH --error=logs/ev6_augment_%A_%a.err

# Augment the EXISTING snapshot cache in place with continuous FLAML scores
# (scripts/augment_snapshots_flaml.py). Re-runs only the FLAML pass on the already-cropped, H-stripped
# atom_array + baked hbond_records — no CIF parse, no HBPLUS. Idempotent: a snapshot that already carries
# flaml_p is skipped.
#
#   sbatch scripts/slurm_augment_snapshots.sh /novo/users/cpjb/rdd/cpjb/ev6_snapshots
#
# NOTE: do NOT set EV6_*_PROB_THR here — persisted scores are threshold-independent and the token stays at
# thresholds.json defaults. The sweep happens at TRAIN time, not here.

set -euo pipefail
cd /novo/users/cpjb/PHD/conditional_binding/ph
mkdir -p logs

export HBPLUS_PATH=/novo/users/cpjb/tools/hbplus/hbplus
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PRECOMPUTED_DIR=${1:?arg 1: existing snapshot cache dir}
EXTENDED_VOCAB=${2:-v6}
# arg 3 = the sharding MODULUS. It MUST stay fixed across a resume: a snapshot's shard is `dir_idx % N_SHARDS`,
# so re-running a SUBSET array (e.g. only the timed-out shards) must keep the SAME modulus the full run used,
# NOT SLURM_ARRAY_TASK_COUNT (which would become the subset size and re-shard over the wrong directories).
# Pass 100 explicitly on a partial re-run: `sbatch --array=0,1,2 ... <cache> v6 100`.
N_SHARDS=${3:-${SLURM_ARRAY_TASK_COUNT:-100}}

echo "=========================================================="
echo " AUGMENT shard $SLURM_ARRAY_TASK_ID/$N_SHARDS  vocab=$EXTENDED_VOCAB"
echo " cache=$PRECOMPUTED_DIR"
echo " node $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  started $(date)"
echo "=========================================================="

srun .venv/bin/python scripts/augment_snapshots_flaml.py \
    --precomputed-dir "$PRECOMPUTED_DIR" \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --n-shards "$N_SHARDS" \
    --extended-vocab "$EXTENDED_VOCAB"

echo "finished $(date)"
