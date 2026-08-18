#!/bin/bash

# Run inside a Perlmutter GPU allocation. This builds the current WarpX tree,
# profiles identical sync-on and no-sync cases, and writes compact summaries.

set -eo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this script inside a Slurm allocation." >&2
    exit 2
fi

work_root="${WORK_ROOT:-/pscratch/sd/s/swu264/NERSC_HACKATHON_2026}"
warpx_dir="${WARPX_DIR:-${work_root}/warpx}"
build_dir="${BUILD_DIR:-${warpx_dir}/build_pm_mlmg_nvtx}"
profile_script="${warpx_dir}/Tools/Performance/profile_mlmg_perlmutter.sbatch"
comparison_dir="${COMPARISON_DIR:-${work_root}/profiles/mlmg-${SLURM_JOB_ID}-nosync-ab}"
build_jobs="${BUILD_JOBS:-16}"

if [[ -z "${MY_PROFILE:-}" ]]; then
    profile_file="${PERLMUTTER_GPU_PROFILE:-${HOME}/perlmutter_gpu_warpx.profile}"
    # shellcheck disable=SC1090
    source "${profile_file}"
fi
set -u

mkdir -p "${comparison_dir}"

echo "WarpX commit: $(git -C "${warpx_dir}" rev-parse HEAD)"
echo "Building: ${build_dir}"
cmake --build "${build_dir}" -j "${build_jobs}"

run_case()
{
    local label="$1"
    local target_args="$2"
    local case_dir="${comparison_dir}/${label}"

    echo "Running ${label}: TARGET_ARGS=${target_args:-<empty>}"
    PROFILE_DIR="${case_dir}" \
    TARGET_ARGS="${target_args}" \
        bash "${profile_script}" 2>&1 | tee "${comparison_dir}/${label}.log"

    grep -E \
        '^(MLMG: Initial rhs|MLMG: Final Iter|MLMG: Timers|Total Time)' \
        "${comparison_dir}/${label}.log" > "${comparison_dir}/${label}-timers.txt"

    nsys stats \
        --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
        "${case_dir}/mlmg-rank0.nsys-rep" \
        > "${comparison_dir}/${label}-rank0-stats.txt"
}

# The default remains false, reproducing the original synchronization behavior.
run_case "sync-on" "warpx.projection_div_cleaner.no_gpu_sync=0"
run_case "no-sync" "warpx.projection_div_cleaner.no_gpu_sync=1"

echo
echo "===== sync-on ====="
cat "${comparison_dir}/sync-on-timers.txt"
echo "===== no-sync ====="
cat "${comparison_dir}/no-sync-timers.txt"
echo "Comparison artifacts: ${comparison_dir}"
