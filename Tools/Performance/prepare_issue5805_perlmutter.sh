#!/bin/bash

# Prepare the 2D CUDA executable used by issue #5805 on Perlmutter.

set -euo pipefail

work_root="${WORK_ROOT:-/pscratch/sd/s/swu264/NERSC_HACKATHON_2026}"
warpx_dir="${WARPX_DIR:-${work_root}/warpx}"
amrex_dir="${AMREX_DIR:-${work_root}/amrex}"
build_dir="${BUILD_DIR:-${warpx_dir}/build_pm_issue5805_2d}"
build_jobs="${BUILD_JOBS:-16}"
marker_patch="${warpx_dir}/Tools/Performance/amrex_issue5805_compute_nvtx.patch"

if [[ -z "${MY_PROFILE:-}" ]]; then
    profile_file="${PERLMUTTER_GPU_PROFILE:-${HOME}/perlmutter_gpu_warpx.profile}"
    # shellcheck disable=SC1090
    source "${profile_file}"
fi

if ! grep -q 'MLMG::mgVcycle_down::amr=' \
    "${amrex_dir}/Src/LinearSolvers/MLMG/AMReX_MLMG.H"; then
    echo "AMReX is missing the hierarchical mgVcycle markers." >&2
    echo "Check out the AMReX hackathon-mlmg-nvtx branch first." >&2
    exit 2
fi

if ! grep -q 'MLCurlCurl::kernel::smooth4::amr=' \
    "${amrex_dir}/Src/LinearSolvers/MLMG/AMReX_MLCurlCurl.cpp"; then
    git -C "${amrex_dir}" apply --check "${marker_patch}"
    git -C "${amrex_dir}" apply "${marker_patch}"
    echo "Applied MLCurlCurl compute-range markers to ${amrex_dir}."
fi

cmake --fresh -S "${warpx_dir}" -B "${build_dir}" \
    -DWarpX_COMPUTE=CUDA \
    -DWarpX_DIMS=2 \
    -DWarpX_APP=ON \
    -DWarpX_MPI=ON \
    -DWarpX_PYTHON=OFF \
    -DWarpX_PRECISION=DOUBLE \
    -DWarpX_PARTICLE_PRECISION=DOUBLE \
    -DWarpX_FFT=OFF \
    -DWarpX_OPENPMD=OFF \
    -DWarpX_QED=OFF \
    -DWarpX_amrex_src="${amrex_dir}" \
    -DAMReX_TINY_PROFILE=ON

cmake --build "${build_dir}" -j "${build_jobs}"

grep -E \
    '^(WarpX_amrex_src|WarpX_COMPUTE|WarpX_DIMS|WarpX_MPI|AMReX_CUDA|AMReX_TINY_PROFILE):' \
    "${build_dir}/CMakeCache.txt"

echo "WarpX commit: $(git -C "${warpx_dir}" rev-parse HEAD)"
echo "AMReX commit: $(git -C "${amrex_dir}" rev-parse HEAD)"
echo "Executable: ${build_dir}/bin/warpx.2d"
