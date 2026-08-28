#!/usr/bin/env bash

set -euo pipefail

ARG1=${1:-fast}

# Set the WarpX directory
WARPX_DIR=${WARPX_DIR:-$(git rev-parse --show-toplevel)}

: "${AMREX_DIR:?Set AMREX_DIR to the AMReX source checkout}"
AMREX_DIR=$(cd -- "${AMREX_DIR}" && pwd)

BUILD_DIR=${WARPX_BUILD_DIR:-${WARPX_DIR}/build-roelof-sp}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-$(command -v python)}
PYTHON_ROOT_DIR=${PYTHON_ROOT_DIR:-${VIRTUAL_ENV:-${CONDA_PREFIX:-}}}

if [[ -z "${PYTHON_ROOT_DIR}" ]]; then
    echo "Activate a virtual/Conda environment or set PYTHON_ROOT_DIR." >&2
    exit 2
fi

case "${BUILD_DIR}/" in
    "${WARPX_DIR}/"*) ;;
    *)
        echo "WARPX_BUILD_DIR must be inside the WarpX repository: ${WARPX_DIR}" >&2
        exit 2
        ;;
esac

if [[ "${ARG1}" == "clean" ]]; then
    echo "cleaning ${BUILD_DIR}"
    cmake -E remove_directory "${BUILD_DIR}"
elif [[ "${ARG1}" != "fast" ]]; then
    echo "Usage: $0 [fast|clean]" >&2
    exit 2
fi

# Set the dims variable to list all the dimensions you'd like to build for.
# Only building for the dimension you are interested in will save time.
dims="3" #"2;3;RZ"

# Set WarpX compile time variables as desired and configure the build
cmake --fresh -S "${WARPX_DIR}" -B "${BUILD_DIR}" \
    -DWarpX_DIMS=$dims -DWarpX_OPENPMD=OFF \
    -DWarpX_QED=OFF -DWarpX_PYTHON=ON -DWarpX_FFT=OFF \
    -DWarpX_COMPUTE=CUDA \
    -DWarpX_PRECISION=SINGLE \
    -DWarpX_PARTICLE_PRECISION=SINGLE \
-DWarpX_amrex_src="${AMREX_DIR}" \
-DAMReX_DIFFERENT_COMPILER=ON \
-DPython_ROOT_DIR="${PYTHON_ROOT_DIR}" \
-DPython_EXECUTABLE="${PYTHON_EXECUTABLE}" \
-DPython_FIND_VIRTUALENV=ONLY
    # -DCMAKE_BUILD_TYPE=Debug \
    # -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    # -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
    # -DWarpX_amrex_src=~/repositories/amrex/ \
    # -DWarpX_pyamrex_src=/home/rgroenewald/repositories/pyamrex/ \
    # -DPython_ROOT_DIR=/home/rgroenewald/venvs/warpx/bin/python3.11/ \
    # -DPython_EXECUTABLE=$(which python3.11) \
    # -DPython_FIND_VIRTUALENV=ONLY \
    # -DWarpX_pyamrex_repo=https://github.com/roelof-groenewald/pyamrex.git \
    # -DWarpX_pyamrex_branch=lincomb_intvect \

# Run the build with 16 procs
cmake --build "${BUILD_DIR}" -j "${BUILD_JOBS:-16}" --target pip_install
