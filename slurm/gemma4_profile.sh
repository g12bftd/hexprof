#!/bin/bash -l
# Gemma 4 E4B: compile for the Galaxy S25 NPU on AI Hub and profile every part (random inputs,
# as AI Hub measures it). Results: $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export/ (run_manifest.md: model,
# settings vs Qualcomm's, hardware, every AI Hub job). Extra arguments go to `hexprof gemma4 aihub`:
#   sbatch slurm/gemma4_profile.sh
#   sbatch slurm/gemma4_profile.sh --context-length 2048     # -> gemma4-s25-export-cl2048/

#SBATCH --job-name=gemma4_export
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/%u/job_logs/hexprof/job_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/job_logs/hexprof/error_%j.err
#SBATCH --account=lp160

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hexprof

export PYTHONUNBUFFERED=1                      # print logs as they happen
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HEXPROF_DOWNLOAD_DIR=${HEXPROF_DOWNLOAD_DIR:-/capstor/scratch/cscs/${USER}/VLMs}

cd "${HEXPROF_REPO:-${HOME}/repos/hexprof}" || exit 1
echo "Python:               $(which python)"
echo "Conda environment:    ${CONDA_DEFAULT_ENV}"
echo "HEXPROF_DOWNLOAD_DIR: ${HEXPROF_DOWNLOAD_DIR}"
echo "hexprof commit:       $(git log -1 --oneline)"
python -c "from importlib.metadata import version as v; print('qai-hub-models', v('qai-hub-models'), '| torch', v('torch'), '| transformers', v('transformers'))"

srun -u hexprof gemma4 aihub --export-only "$@"
status=$?

echo "Finished on: $(date) (exit code ${status})"
exit ${status}
