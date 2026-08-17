#!/bin/bash

# Synchronize the two hackathon forks and build WarpX against the AMReX worktree.
# Run this script on a Perlmutter login node after sourcing the WarpX GPU profile.

set -euo pipefail

work_root="${WORK_ROOT:-/pscratch/sd/s/swu264/NERSC_HACKATHON_2026}"
warpx_dir="${WARPX_DIR:-${work_root}/warpx}"
amrex_dir="${AMREX_DIR:-${work_root}/amrex}"
branch="${MLMG_BRANCH:-hackathon-mlmg-nvtx}"
build_dir="${BUILD_DIR:-${warpx_dir}/build_pm_mlmg_nvtx}"
build_jobs="${BUILD_JOBS:-16}"

sync_repo()
{
    local repo_dir="$1"
    local repo_url="$2"

    if [[ -d "${repo_dir}/.git" ]]; then
        if ! git -C "${repo_dir}" diff --quiet || \
           ! git -C "${repo_dir}" diff --cached --quiet; then
            echo "Refusing to switch a worktree with tracked changes: ${repo_dir}" >&2
            return 2
        fi
        git -C "${repo_dir}" remote set-url origin "${repo_url}"
        git -C "${repo_dir}" fetch origin "${branch}"
        if git -C "${repo_dir}" show-ref --verify --quiet "refs/heads/${branch}"; then
            git -C "${repo_dir}" switch "${branch}"
            git -C "${repo_dir}" merge --ff-only "origin/${branch}"
        else
            git -C "${repo_dir}" switch --create "${branch}" --track "origin/${branch}"
        fi
    elif [[ -e "${repo_dir}" ]]; then
        echo "Path exists but is not a Git worktree: ${repo_dir}" >&2
        return 2
    else
        git clone --branch "${branch}" "${repo_url}" "${repo_dir}"
    fi
}

if [[ -z "${MY_PROFILE:-}" ]]; then
    profile_file="${PERLMUTTER_GPU_PROFILE:-${HOME}/perlmutter_gpu_warpx.profile}"
    if [[ ! -f "${profile_file}" ]]; then
        echo "Missing Perlmutter GPU profile: ${profile_file}" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${profile_file}"
fi

sync_repo "${amrex_dir}" "https://github.com/shixun404/amrex.git"
sync_repo "${warpx_dir}" "https://github.com/shixun404/warpx.git"

echo "WarpX commit: $(git -C "${warpx_dir}" rev-parse HEAD)"
echo "AMReX commit: $(git -C "${amrex_dir}" rev-parse HEAD)"

cmake --fresh -S "${warpx_dir}" -B "${build_dir}" \
    -DWarpX_COMPUTE=CUDA \
    -DWarpX_DIMS=3 \
    -DWarpX_APP=OFF \
    -DWarpX_PYTHON=ON \
    -DWarpX_PYTHON_IPO=OFF \
    -DWarpX_FFT=OFF \
    -DWarpX_OPENPMD=OFF \
    -DWarpX_QED=OFF \
    -DWarpX_amrex_src="${amrex_dir}" \
    -DAMReX_TINY_PROFILE=ON

cmake --build "${build_dir}" -j "${build_jobs}" --target pip_install

cache_file="${build_dir}/CMakeCache.txt"
grep -E '^(WarpX_amrex_src|WarpX_COMPUTE|WarpX_DIMS|WarpX_MPI|AMReX_CUDA|AMReX_TINY_PROFILE):' \
    "${cache_file}"

echo "Build ready: ${build_dir}"
