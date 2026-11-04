#!/bin/bash -l
# Measured NPU ceilings on AI Hub's Galaxy S25, the denominators for "% of peak" in the hardware report:
#   compute:   chained MatMuls per precision -> P (TOPS) for w8a8 / w8a16 (SigLIP 2), w4a16 (Gemma 4 text), fp16 (vision)
#   bandwidth: single-token linear stacks at w4a16 too large for on-chip memory -> B (GB/s), refit on the
#              DRAM bytes QHAS measured (--hw) as a cross-check
# Results: $HEXPROF_DOWNLOAD_DIR/runs/ceil-compute/compute_ceiling.json and runs/ceil-bandwidth/bandwidth_ceiling.json
# Job ids are saved after every submission: re-submitting this script resumes instead of starting over.
#   sbatch slurm/ceilings.sh

#SBATCH --job-name=ceilings
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/%u/job_logs/hexprof/ceilings_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/job_logs/hexprof/ceilings_%j.err
#SBATCH --account=lp160

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hexprof

export PYTHONUNBUFFERED=1
export HEXPROF_DOWNLOAD_DIR=${HEXPROF_DOWNLOAD_DIR:-/capstor/scratch/cscs/${USER}/VLMs}

cd "${HEXPROF_REPO:-${HOME}/repos/hexprof}" || exit 1
RUNS=${HEXPROF_DOWNLOAD_DIR}/runs
# Same QAIRT as the deployment being compared (qai_hub_models pins 2.50 for its exports); override with HEXPROF_QAIRT
QAIRT=${HEXPROF_QAIRT:-2.50}
echo "HEXPROF_DOWNLOAD_DIR: ${HEXPROF_DOWNLOAD_DIR}"
echo "QAIRT:                ${QAIRT}"
echo "hexprof commit:       $(git log -1 --oneline)"

echo "================ compute ceiling ================"
srun -u hexprof ceilings compute --precisions w8a8 w8a16 w4a16 fp16 --size 2048 --chain 1 2 4 8 16 \
    --qairt-version "${QAIRT}" --hw --out "${RUNS}/ceil-compute"
compute_status=$?

echo "================ bandwidth ceiling ================"
srun -u hexprof ceilings bandwidth --precision w4a16 --widths 3072 4096 5120 6144 7168 --width-layers 16 \
    --qairt-version "${QAIRT}" --hw --out "${RUNS}/ceil-bandwidth"
bandwidth_status=$?

echo "Finished on: $(date) (compute exit ${compute_status}, bandwidth exit ${bandwidth_status})"
exit $(( compute_status || bandwidth_status ))
