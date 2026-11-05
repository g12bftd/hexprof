#!/bin/bash -l
# Hardware report (HTML + CSV) from the QHAS / optrace files of the profiling jobs:
#   $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export/hw_report/index.html
#   $HEXPROF_DOWNLOAD_DIR/runs/siglip2-profile-<jobid>/hw_report/index.html
# 1. records each graph's compiled size, parameters and (for the Gemma 4 vision encoder) source ONNX
#    from AI Hub (cached in <results>/models/; set HEXPROF_FETCH_ONNX=all for per-op MACs of every graph,
#    several GB per Gemma 4 text part, or none)
# 2. uses the measured ceilings from slurm/ceilings.sh when they exist (% of P, % of B, roofline)
#   sbatch slurm/hw_report.sh

#SBATCH --job-name=hw_report
#SBATCH --time=02:00:00
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

CEIL=()
C=${HEXPROF_DOWNLOAD_DIR}/runs/ceil-compute/compute_ceiling.json
B=${HEXPROF_DOWNLOAD_DIR}/runs/ceil-bandwidth/bandwidth_ceiling.json
[ -f "$C" ] && CEIL+=(--compute-json "$C") || echo "No compute ceiling yet ($C): run slurm/ceilings.sh for % of P"
[ -f "$B" ] && CEIL+=(--bandwidth-json "$B") || echo "No bandwidth ceiling yet ($B): run slurm/ceilings.sh for % of B"

# your inference runs on real images and prompts (hexprof_vlm_infer.sbatch writes runs/<jobid>/{gemma4,siglip2})
shopt -s nullglob
GEMMA_REQ=("${HEXPROF_DOWNLOAD_DIR}"/runs/*/gemma4/results.json)
SIGLIP_REQ=("${HEXPROF_DOWNLOAD_DIR}"/runs/*/siglip2/results.json)
shopt -u nullglob
echo "Gemma 4 requests found: ${#GEMMA_REQ[@]}; SigLIP 2 requests found: ${#SIGLIP_REQ[@]}"

status=0
if [ $# -gt 0 ]; then  # explicit result folders, e.g. sbatch slurm/hw_report.sh $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export-cl2048
  for d in "$@"; do
    srun -u hexprof hw fetch-models "$d" --onnx "${HEXPROF_FETCH_ONNX:-vision}" \
      || echo "fetch-models failed: the report continues without AI Hub model facts"
    srun -u hexprof hw report "$d" --title "$(basename "$d") on Galaxy S25" "${CEIL[@]}" || status=$?
  done
  echo "Finished on: $(date) (exit code ${status})"
  exit ${status}
fi

# Qualcomm's settings (gemma4-s25-export) and any changed context / prompt length (gemma4-s25-export-cl*)
for GEMMA in "${HEXPROF_DOWNLOAD_DIR}"/gemma4-s25-export*/; do
  [ -d "$GEMMA" ] || continue
  GEMMA=${GEMMA%/}
  REQ=()
  [ "$(basename "$GEMMA")" = "gemma4-s25-export" ] && REQ=("${GEMMA_REQ[@]}")  # requests ran on Qualcomm's settings
  srun -u hexprof hw fetch-models "$GEMMA" --onnx "${HEXPROF_FETCH_ONNX:-vision}" \
    || echo "fetch-models failed: the report continues without AI Hub model facts"
  srun -u hexprof hw report "$GEMMA" --title "Gemma 4 E4B on Galaxy S25 ($(basename "$GEMMA"))" "${CEIL[@]}" \
    ${REQ:+--requests "${REQ[@]}"} || status=$?
done

for d in "${HEXPROF_DOWNLOAD_DIR}"/runs/siglip2-profile-*/; do
  [ -d "$d" ] || continue
  srun -u hexprof hw fetch-models "$d" || echo "fetch-models failed: the report continues without AI Hub model facts"
  srun -u hexprof hw report "$d" --title "SigLIP 2 on Galaxy S25" "${CEIL[@]}" \
    ${SIGLIP_REQ:+--requests "${SIGLIP_REQ[@]}"} || status=$?
done

echo "Finished on: $(date) (exit code ${status})"
exit ${status}
