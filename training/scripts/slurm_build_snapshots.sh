#!/bin/bash
#SBATCH --job-name=ev6_snapshots
#SBATCH --partition=highmem        # the atomworks pipeline needs >compute's ~15.5G (see slurm_extract_potts)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # EV6's fold ensembles thread over these; the pipeline is single-structure
#SBATCH --mem=56G
#SBATCH --time=08:00:00            # ~3.5k examples/shard x ~2s + overhead; generous for slow structures
#SBATCH --array=0-99               # 100 shards per dataset; strided over the filtered example_ids
#SBATCH --output=logs/ev6_snapshots_%A_%a.out
#SBATCH --error=logs/ev6_snapshots_%A_%a.err

# Build the precomputed-annotation cache (scripts/build_snapshots.py) as a sharded array.
#
# Submit ONCE PER DATASET into the SAME cache dir (example_id is globally unique, hash-sharded, so no
# collision). The cache is ~2.4 TB total (349k PDB + 459k AFDB example_ids x ~3 MB) and ~449 CPU-h.
# Args are POSITIONAL ($1 dataset, $2 cache dir): this site sets SLURM_EXPORT_ENV=NONE so `VAR=x sbatch`
# env vars are DROPPED, but script args are always passed through.
#
#   sbatch scripts/slurm_build_snapshots.sh pdb  /path/to/cache
#   sbatch scripts/slurm_build_snapshots.sh afdb /path/to/cache
#
# Resumable: each task skips example_ids whose snapshot already exists, so requeue/rerun is cheap.

set -euo pipefail
cd /novo/users/cpjb/PHD/conditional_binding/ph
mkdir -p logs

export HBPLUS_PATH=/novo/users/cpjb/tools/hbplus/hbplus
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATASET=${1:?arg 1: dataset (pdb or afdb)}
PRECOMPUTED_DIR=${2:?arg 2: shared cache dir (~2.4 TB free)}
EXTENDED_VOCAB=${3:-v6}
# number of shards = array size; SLURM sets SLURM_ARRAY_TASK_COUNT, fall back to the --array upper bound+1
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-100}

echo "=========================================================="
echo " dataset=$DATASET  shard $SLURM_ARRAY_TASK_ID/$N_SHARDS  vocab=$EXTENDED_VOCAB"
echo " cache=$PRECOMPUTED_DIR"
echo " node $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  started $(date)"
echo "=========================================================="

srun .venv/bin/python scripts/build_snapshots.py \
    --dataset "$DATASET" \
    --precomputed-dir "$PRECOMPUTED_DIR" \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --n-shards "$N_SHARDS" \
    --extended-vocab "$EXTENDED_VOCAB"

echo "finished $(date)"
