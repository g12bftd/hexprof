#!/bin/bash -l
# SigLIP 2 (Qualcomm's W8A16 asset): profile the image and text encoders on AI Hub's Galaxy S25
# (random inputs, as AI Hub measures it). Results: $HEXPROF_DOWNLOAD_DIR/runs/siglip2-profile-<jobid>/
#   sbatch slurm/siglip2_profile.sh

#SBATCH --job-name=siglip2_profile
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/%u/job_logs/hexprof/siglip2_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/job_logs/hexprof/siglip2_%j.err
#SBATCH --account=lp160

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hexprof

export PYTHONUNBUFFERED=1
export HEXPROF_DOWNLOAD_DIR=${HEXPROF_DOWNLOAD_DIR:-/capstor/scratch/cscs/${USER}/VLMs}

cd "${HEXPROF_REPO:-${HOME}/repos/hexprof}" || exit 1
RUN_DIR=${HEXPROF_DOWNLOAD_DIR}/runs/siglip2-profile-${SLURM_JOB_ID}
echo "Python:               $(which python)"
echo "HEXPROF_DOWNLOAD_DIR: ${HEXPROF_DOWNLOAD_DIR}"
echo "hexprof commit:       $(git log -1 --oneline)"
echo "Results:              ${RUN_DIR}"

srun -u hexprof siglip2 profile --components image_encoder text_encoder --out "${RUN_DIR}"
status=$?

echo "Finished on: $(date) (exit code ${status})"
exit ${status}
