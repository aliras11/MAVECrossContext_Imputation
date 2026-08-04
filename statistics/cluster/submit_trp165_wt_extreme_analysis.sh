#!/usr/bin/env bash
#SBATCH --job-name=trp165_wt
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --partition=dept_cpu

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 STAGE_ROOT RUN_ROOT OUTPUT_DIR REPO_COMMIT [RATES_CSV] [SPLITS]" >&2
  exit 2
fi

stage_root=$1
run_root=$2
output_dir=$3
repo_commit=$4
rates_csv=${5:-"10,20,40,60,80,90"}
splits=${6:-"1-50"}
IFS=',' read -r -a rate_args <<< "${rates_csv}"

eval "$(conda shell.bash hook)"
conda activate struct
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${stage_root}/code"
source_files=(
  "pyproject.toml"
  "code/mave_statistics/__init__.py"
  "code/mave_statistics/constants.py"
  "code/mave_statistics/trp165.py"
  "code/cli/__init__.py"
  "code/cli/analyze_trp165_wt_extremes.py"
  "cluster/submit_trp165_wt_extreme_analysis.sh"
)
source_checksum_args=()
for relative_path in "${source_files[@]}"; do
  absolute_path="${stage_root}/${relative_path}"
  [[ -f "${absolute_path}" ]] || {
    echo "missing staged source: ${absolute_path}" >&2
    exit 1
  }
  digest=$(sha256sum "${absolute_path}" | awk '{print $1}')
  source_checksum_args+=(--source-checksum "${relative_path}=${digest}")
done

python "${stage_root}/code/cli/analyze_trp165_wt_extremes.py" \
  --run-root "${run_root}" \
  --output-dir "${output_dir}" \
  --rates "${rate_args[@]}" \
  --splits "${splits}" \
  --repo-commit "${repo_commit}" \
  "${source_checksum_args[@]}"
