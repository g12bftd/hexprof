#!/bin/bash -l
# One shrunk model with random weights on AI Hub's Galaxy S25: Hugging Face config with the requested
# knobs -> random checkpoint -> Qualcomm's own NPU-adapted export -> AI Hub quantize / compile / link /
# profile (QHAS + optrace) -> verification (the binary that ran has the requested shapes) -> report.
# Arguments are passed to `hexprof dummy run` unchanged, e.g.
#   sbatch slurm/dummy.sh gemma4-text --dims layers=12 hidden=1280 intermediate=5120
#   sbatch slurm/dummy.sh gemma4-text                       # full size: the fidelity check vs Qualcomm's export
#   sbatch slurm/dummy.sh siglip2-image --dims layers=6 hidden=512 heads=8 image-size=256 --batch 2
#   sbatch slurm/dummy.sh gemma4-vision --dims layers=8 soft-tokens=140
# Check a request first (seconds, no AI Hub): hexprof dummy plan <target> --dims ...
# Results: $HEXPROF_DOWNLOAD_DIR/dummy/<target>__<knobs>/ (hw_report/index.html, verification.json)
# Re-submitting the same command resumes from the recorded AI Hub jobs.

#SBATCH --job-name=dummy
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/%u/job_logs/hexprof/dummy_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/job_logs/hexprof/dummy_%j.err
#SBATCH --account=lp160

if [ $# -eq 0 ]; then
  echo "usage: sbatch slurm/dummy.sh <gemma4-text|gemma4-vision|siglip2-image|siglip2-text> [--dims knob=N ...]" >&2
  exit 2
fi

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "Dummy:           $*"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hexprof

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HEXPROF_DOWNLOAD_DIR=${HEXPROF_DOWNLOAD_DIR:-/capstor/scratch/cscs/${USER}/VLMs}

cd "${HEXPROF_REPO:-${HOME}/repos/hexprof}" || exit 1
echo "hexprof commit: $(git log -1 --oneline)"

srun -u hexprof dummy run "$@"
status=$?

echo "Finished on: $(date) (exit code ${status})"
exit ${status}
