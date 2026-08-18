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
profile_dir="${PROFILE_DIR:-${work_root}/profiles/mlmg-${SLURM_JOB_ID}}"
target_input="${TARGET_INPUT:-${warpx_dir}/Tools/Performance/inputs_mlmg_magnetic_mirror_picmi.py}"

if [[ -z "${MY_PROFILE:-}" ]]; then
    profile_file="${PERLMUTTER_GPU_PROFILE:-${HOME}/perlmutter_gpu_warpx.profile}"
    # shellcheck disable=SC1090
    source "${profile_file}"
fi
set -u

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    standard_venv="${SW_DIR:-}/venvs/warpx-gpu"
    if [[ -n "${WARPX_VENV:-}" ]]; then
        venv_dir="${WARPX_VENV}"
    elif [[ -n "${SW_DIR:-}" && -d "${standard_venv}" ]]; then
        venv_dir="${standard_venv}"
    else
        venv_dir="${work_root}/.venv-mlmg"
    fi

    if [[ ! -f "${venv_dir}/bin/activate" ]]; then
        echo "Missing Python environment: ${venv_dir}" >&2
        echo "Run Tools/Performance/prepare_mlmg_perlmutter.sh first." >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
fi

if [[ ! -f "${target_input}" ]]; then
    echo "Target input does not exist: ${target_input}" >&2
    exit 2
fi

mkdir -p "${profile_dir}"
cd "${profile_dir}"

export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.use_gpu_aware_mpi=1}"
export MPLBACKEND=Agg
export PROFILE_DIR="${profile_dir}"
export TARGET_INPUT="${target_input}"

# MLMG_NCELL, MLMG_MAX_GRID_SIZE, MLMG_BLOCKING_FACTOR and MLMG_RTOL can be
# exported before sbatch. TARGET_INPUT can point at the team's exact application.
srun --cpu-bind=cores bash -c '
    export CUDA_VISIBLE_DEVICES=$((3-SLURM_LOCALID))
    nsys profile \
        --trace=cuda,nvtx,mpi,osrt \
        --mpi-impl=mpich \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output="${PROFILE_DIR}/mlmg-rank${SLURM_PROCID}" \
        python3 "${TARGET_INPUT}"
'

echo "Nsight Systems reports: ${profile_dir}/mlmg-rank*.nsys-rep"
