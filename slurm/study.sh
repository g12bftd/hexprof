#!/bin/bash -l
# One systematic study on AI Hub's Galaxy S25: every value of one setting is built with Qualcomm's own
# export / model code, compiled and profiled on the device, verified (what ran = what was requested),
# and compared. Arguments are passed to `hexprof study run` unchanged, e.g.
#   sbatch slurm/study.sh gemma4-text --axis context-length --values 512 1024 2048 4096
#   sbatch slurm/study.sh gemma4-vision --axis soft-tokens --values 70 140 280 560 1120
#   sbatch slurm/study.sh siglip2-text --axis batch --values 1 2 4 8 16
#   sbatch slurm/study.sh dummy-gemma4-text --axis layers --values 6 12 24 42 --dims hidden=1280 intermediate=5120
# Results: $HEXPROF_DOWNLOAD_DIR/studies/<target>__<axis>/comparison/index.html
# Re-submitting the same command resumes: finished values are not rebuilt.

#SBATCH --job-name=study
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/%u/job_logs/hexprof/study_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/job_logs/hexprof/study_%j.err
#SBATCH --account=lp160

if [ $# -eq 0 ]; then
  echo "usage: sbatch slurm/study.sh <target> --axis <axis> [--values ...] (see: hexprof study list)" >&2
  exit 2
fi

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "Study:           $*"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hexprof

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HEXPROF_DOWNLOAD_DIR=${HEXPROF_DOWNLOAD_DIR:-/capstor/scratch/cscs/${USER}/VLMs}

cd "${HEXPROF_REPO:-${HOME}/repos/hexprof}" || exit 1
echo "hexprof commit: $(git log -1 --oneline)"

srun -u hexprof study run "$@"
status=$?

echo "Finished on: $(date) (exit code ${status})"
exit ${status}
