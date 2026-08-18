#!/bin/bash -l

# Submit with, for example:
# sbatch -A "$proj" Tools/Performance/profile_mlmg_perlmutter.sbatch

#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -J mlmg-nvtx
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH -o mlmg-nvtx.o%j
#SBATCH -e mlmg-nvtx.e%j

set -eo pipefail

work_root="${WORK_ROOT:-/pscratch/sd/s/swu264/NERSC_HACKATHON_2026}"
warpx_dir="${WARPX_DIR:-${work_root}/warpx}"
build_dir="${BUILD_DIR:-${warpx_dir}/build_pm_mlmg_nvtx}"
profile_dir="${PROFILE_DIR:-${work_root}/profiles/mlmg-${SLURM_JOB_ID}}"
target_input="${TARGET_INPUT:-${warpx_dir}/Tools/Performance/inputs_mlmg_magnetic_mirror}"

if [[ -z "${MY_PROFILE:-}" ]]; then
    profile_file="${PERLMUTTER_GPU_PROFILE:-${HOME}/perlmutter_gpu_warpx.profile}"
    # shellcheck disable=SC1090
    source "${profile_file}"
fi
set -u

if [[ ! -f "${target_input}" ]]; then
    echo "Target input does not exist: ${target_input}" >&2
    exit 2
fi

if [[ "${target_input}" == *.py ]]; then
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        standard_venv="${SW_DIR:-}/venvs/warpx-gpu"
        if [[ -n "${WARPX_VENV:-}" ]]; then
            venv_dir="${WARPX_VENV}"
        elif [[ -n "${SW_DIR:-}" && -d "${standard_venv}" ]]; then
            venv_dir="${standard_venv}"
        else
            venv_dir="${work_root}/.venv-mlmg"
        fi
        if [[ -f "${venv_dir}/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source "${venv_dir}/bin/activate"
        fi
    fi
    python_deps="${PYTHON_DEPS:-${work_root}/python_deps}"
    if [[ -d "${python_deps}" ]]; then
        export PYTHONPATH="${python_deps}:${warpx_dir}/Python:${PYTHONPATH:-}"
    fi
    profile_exe="${PYTHON_EXE:-$(command -v python3)}"
else
    profile_exe="${WARPX_EXE:-${build_dir}/bin/warpx.3d}"
fi
if [[ ! -x "${profile_exe}" ]]; then
    echo "Profile executable is missing or not executable: ${profile_exe}" >&2
    exit 2
fi

mkdir -p "${profile_dir}"
cd "${profile_dir}"

export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.use_gpu_aware_mpi=1}"
export MPLBACKEND=Agg
export PROFILE_DIR="${profile_dir}"
export PROFILE_EXE="${profile_exe}"
export TARGET_INPUT="${target_input}"
export TARGET_ARGS="${TARGET_ARGS:-}"

# TARGET_INPUT can point at the team's exact native input deck or Python application.
srun --cpu-bind=cores bash -c '
    export CUDA_VISIBLE_DEVICES=$((3-SLURM_LOCALID))
    read -r -a target_args <<< "${TARGET_ARGS}"
    nsys profile \
        --trace=cuda,nvtx,mpi,osrt \
        --mpi-impl=mpich \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output="${PROFILE_DIR}/mlmg-rank${SLURM_PROCID}" \
        "${PROFILE_EXE}" "${TARGET_INPUT}" "${target_args[@]}"
'

echo "Nsight Systems reports: ${profile_dir}/mlmg-rank*.nsys-rep"
