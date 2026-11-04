#!/bin/bash -l
# Hardware metrics report (HTML + CSV) from the QHAS / optrace files of the two profiling jobs:
#   $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export/hw_report/index.html
#   $HEXPROF_DOWNLOAD_DIR/runs/siglip2-profile-<jobid>/hw_report/index.html
#   sbatch slurm/hw_report.sh

#SBATCH --job-name=hw_report
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/%u/job_logs/hexprof/hwreport_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/job_logs/hexprof/hwreport_%j.err
#SBATCH --account=lp160

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hexprof

export PYTHONUNBUFFERED=1
export HEXPROF_DOWNLOAD_DIR=${HEXPROF_DOWNLOAD_DIR:-/capstor/scratch/cscs/${USER}/VLMs}

cd "${HEXPROF_REPO:-${HOME}/repos/hexprof}" || exit 1
echo "HEXPROF_DOWNLOAD_DIR: ${HEXPROF_DOWNLOAD_DIR}"
echo "hexprof commit:       $(git log -1 --oneline)"

status=0
if [ -d "${HEXPROF_DOWNLOAD_DIR}/gemma4-s25-export" ]; then
  srun -u hexprof hw report "${HEXPROF_DOWNLOAD_DIR}/gemma4-s25-export" --title "Gemma 4 E4B on Galaxy S25" || status=$?
fi

for d in "${HEXPROF_DOWNLOAD_DIR}"/runs/siglip2-profile-*/; do
  [ -d "$d" ] || continue
  srun -u hexprof hw report "$d" --title "SigLIP 2 on Galaxy S25" || status=$?
done

echo "Finished on: $(date) (exit code ${status})"
exit ${status}
