#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

activate_cluster_environment() {
    eval "$(conda shell.bash hook)"
    conda activate struct
}

run_finalize() {
    finalize_args=(
        -m decomposition.build_decomposition_stats finalize
        --manifest "$DECOMP_MANIFEST"
        --run-root "$DECOMP_RUN_ROOT"
        --output-root "$DECOMP_OUTPUT_ROOT"
        --task-manifest "$DECOMP_TASK_MANIFEST"
        --run-id "$DECOMP_RUN_ID"
        --repository-commit "$DECOMP_REPOSITORY_COMMIT"
    )
    if [[ -n "$DECOMP_GENERATION_COMMIT" ]]; then
        finalize_args+=(--generation-commit "$DECOMP_GENERATION_COMMIT")
    fi
    "$DECOMP_PYTHON" "${finalize_args[@]}"
}

run_summaries() {
    "$DECOMP_PYTHON" -m decomposition.summarize_decomposition \
        --stats "$DECOMP_OUTPUT_ROOT/decomposition_sufficient_stats.csv" \
        --validation "$DECOMP_OUTPUT_ROOT/decomposition_file_validation.csv" \
        --output-root "$DECOMP_OUTPUT_ROOT"
    "$DECOMP_PYTHON" -m decomposition.summarize_split_variability \
        --stats "$DECOMP_OUTPUT_ROOT/decomposition_sufficient_stats.csv" \
        --output-root "$DECOMP_OUTPUT_ROOT" \
        --expected-splits "$DECOMP_EXPECTED_SPLITS"
    "$DECOMP_PYTHON" -m decomposition.build_decomposition_stats complete \
        --output-root "$DECOMP_OUTPUT_ROOT"
}

case "${1:-}" in
    __worker)
        activate_cluster_environment
        cd "$DECOMP_REPO_DIR"
        "$DECOMP_PYTHON" -m decomposition.build_decomposition_stats shard \
            --manifest "$DECOMP_MANIFEST" \
            --run-root "$DECOMP_RUN_ROOT" \
            --output-root "$DECOMP_OUTPUT_ROOT" \
            --task-manifest "$DECOMP_TASK_MANIFEST" \
            --task-index "$SLURM_ARRAY_TASK_ID"
        exit 0
        ;;
    __finalize)
        activate_cluster_environment
        cd "$DECOMP_REPO_DIR"
        run_finalize
        exit 0
        ;;
    __summarize)
        activate_cluster_environment
        cd "$DECOMP_REPO_DIR"
        run_summaries
        exit 0
        ;;
esac

usage() {
    echo "Usage: $0 [--local] --run-root PATH --run-id ID --models FAMILY:VARIANT,... --rates N,... --splits N,... [options]"
    echo
    echo "Options:"
    echo "  --output-root PATH       Default: RUN_ROOT/decomposition_outputs/RUN_ID"
    echo "  --manifest PATH          Default: decomposition/model_layouts.json"
    echo "  --generation-commit SHA  Required for corrected SingleAE, DualAE, or PCA outputs"
    echo "  --development-only       Permit local pre-fix outputs; marks artifacts non-scientific"
    echo "  --python PATH            Python executable (default: python)"
    echo "  --max-concurrent N       Maximum simultaneous Slurm shard jobs (default: 64)"
    echo "  --local                  Run every stage synchronously without Slurm"
}

local_mode=false
run_root=""
output_root=""
run_id=""
models_csv=""
rates_csv=""
splits_csv=""
manifest="$script_dir/model_layouts.json"
generation_commit=""
development_only=false
python_bin="python"
max_concurrent=64

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            local_mode=true
            shift
            ;;
        --run-root)
            run_root="$2"
            shift 2
            ;;
        --output-root)
            output_root="$2"
            shift 2
            ;;
        --run-id)
            run_id="$2"
            shift 2
            ;;
        --models)
            models_csv="$2"
            shift 2
            ;;
        --rates)
            rates_csv="$2"
            shift 2
            ;;
        --splits)
            splits_csv="$2"
            shift 2
            ;;
        --manifest)
            manifest="$2"
            shift 2
            ;;
        --generation-commit)
            generation_commit="$2"
            shift 2
            ;;
        --development-only)
            development_only=true
            shift
            ;;
        --python)
            python_bin="$2"
            shift 2
            ;;
        --max-concurrent)
            max_concurrent="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$run_root" || -z "$run_id" || -z "$models_csv" || -z "$rates_csv" || -z "$splits_csv" ]]; then
    usage >&2
    exit 2
fi
if ! [[ "$max_concurrent" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrent must be a positive integer" >&2
    exit 2
fi
if [[ -z "$output_root" ]]; then
    output_root="$run_root/decomposition_outputs/$run_id"
fi

IFS=',' read -r -a model_args <<< "$models_csv"
IFS=',' read -r -a rate_args <<< "$rates_csv"
IFS=',' read -r -a split_args <<< "$splits_csv"
if [[ ${#model_args[@]} -eq 0 || ${#rate_args[@]} -eq 0 || ${#split_args[@]} -eq 0 ]]; then
    echo "--models, --rates, and --splits must each contain at least one value" >&2
    exit 2
fi

mkdir -p "$output_root/logs"
plan_args=(
    -m decomposition.build_decomposition_stats plan
    --manifest "$manifest"
    --output-root "$output_root"
    --models "${model_args[@]}"
    --rates "${rate_args[@]}"
    --splits "${split_args[@]}"
)
if [[ -n "$generation_commit" ]]; then
    plan_args+=(--generation-commit "$generation_commit")
fi
if [[ "$development_only" == true ]]; then
    plan_args+=(--development-only)
fi

cd "$repo_dir"
"$python_bin" "${plan_args[@]}"

task_manifest="$output_root/decomposition_task_manifest.tsv"
task_count=$(( $(wc -l < "$task_manifest") - 1 ))
if [[ "$task_count" -le 0 ]]; then
    echo "The decomposition task plan is empty" >&2
    exit 1
fi
expected_splits=${#split_args[@]}
repository_commit="$(git rev-parse HEAD)"

export DECOMP_REPO_DIR="$repo_dir"
export DECOMP_PYTHON="$python_bin"
export DECOMP_MANIFEST="$manifest"
export DECOMP_RUN_ROOT="$run_root"
export DECOMP_OUTPUT_ROOT="$output_root"
export DECOMP_TASK_MANIFEST="$task_manifest"
export DECOMP_RUN_ID="$run_id"
export DECOMP_GENERATION_COMMIT="$generation_commit"
export DECOMP_REPOSITORY_COMMIT="$repository_commit"
export DECOMP_EXPECTED_SPLITS="$expected_splits"

if [[ "$local_mode" == true ]]; then
    for ((task_index = 0; task_index < task_count; task_index++)); do
        "$python_bin" -m decomposition.build_decomposition_stats shard \
            --manifest "$manifest" \
            --run-root "$run_root" \
            --output-root "$output_root" \
            --task-manifest "$task_manifest" \
            --task-index "$task_index"
    done
    run_finalize
    run_summaries
    echo "Decomposition artifacts: $output_root"
    exit 0
fi

last_task_index=$((task_count - 1))
export_list="ALL,DECOMP_REPO_DIR,DECOMP_PYTHON,DECOMP_MANIFEST,DECOMP_RUN_ROOT,DECOMP_OUTPUT_ROOT,DECOMP_TASK_MANIFEST,DECOMP_RUN_ID,DECOMP_GENERATION_COMMIT,DECOMP_REPOSITORY_COMMIT,DECOMP_EXPECTED_SPLITS"
array_job_raw=$(
    /opt/slurm/bin/sbatch --parsable \
        --partition=dept_cpu \
        --time=04:00:00 \
        --mem=16G \
        --cpus-per-task=1 \
        --array="0-${last_task_index}%${max_concurrent}" \
        --job-name="decomp_${run_id}_shards" \
        --output="$output_root/logs/shard_%A_%a.out" \
        --error="$output_root/logs/shard_%A_%a.err" \
        --export="$export_list" \
        "$script_dir/submit_decomposition.sh" __worker
)
array_job_id="${array_job_raw%%;*}"

finalize_job_raw=$(
    /opt/slurm/bin/sbatch --parsable \
        --partition=dept_cpu \
        --time=01:00:00 \
        --mem=16G \
        --cpus-per-task=1 \
        --dependency="afterok:${array_job_id}" \
        --job-name="decomp_${run_id}_finalize" \
        --output="$output_root/logs/finalize_%j.out" \
        --error="$output_root/logs/finalize_%j.err" \
        --export="$export_list" \
        "$script_dir/submit_decomposition.sh" __finalize
)
finalize_job_id="${finalize_job_raw%%;*}"

summary_job_raw=$(
    /opt/slurm/bin/sbatch --parsable \
        --partition=dept_cpu \
        --time=01:00:00 \
        --mem=16G \
        --cpus-per-task=1 \
        --dependency="afterok:${finalize_job_id}" \
        --job-name="decomp_${run_id}_summary" \
        --output="$output_root/logs/summary_%j.out" \
        --error="$output_root/logs/summary_%j.err" \
        --export="$export_list" \
        "$script_dir/submit_decomposition.sh" __summarize
)
summary_job_id="${summary_job_raw%%;*}"

echo "Shard array job: $array_job_id"
echo "Finalizer job: $finalize_job_id"
echo "Summary job: $summary_job_id"
echo "Decomposition artifacts: $output_root"
