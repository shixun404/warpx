#!/bin/bash -l

# Fixed four-GPU issue #5805 sweep: np=8, nx=16,32,64,128,256.
# Clean runs produce wall_times.csv. A separate Nsight pass produces one
# .nsys-rep per MPI rank and text reports with GPU-projected NVTX ranges.

#SBATCH -A m4642
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -J issue5805-nx
#SBATCH -o issue5805-nx.o%j
#SBATCH -e issue5805-nx.e%j

set -euo pipefail

work_root="${WORK_ROOT:-/pscratch/sd/s/swu264/NERSC_HACKATHON_2026}"
warpx_dir="${WARPX_DIR:-${work_root}/warpx}"
build_dir="${BUILD_DIR:-${warpx_dir}/build_pm_issue5805_2d}"
warpx_exe="${WARPX_EXE:-${build_dir}/bin/warpx.2d}"
input_file="${TARGET_INPUT:-${warpx_dir}/Tools/Performance/inputs_issue5805_uniform_plasma_2d}"
profile_script="${warpx_dir}/Tools/Performance/profile_mlmg_perlmutter.sbatch"
result_dir="${RESULT_DIR:-${work_root}/profiles/issue5805-${SLURM_JOB_ID}-4gpu-np8-nx-sweep}"

nx_values=(${NX_VALUES:-16 32 64 128 256})
nppc="${NPPC:-8}"
max_step="${MAX_STEP:-10}"
nrepeat="${NREPEAT:-3}"
run_nsys="${RUN_NSYS:-1}"
nsys_trace="${NSYS_TRACE:-cuda,nvtx,mpi,osrt}"

if [[ -z "${MY_PROFILE:-}" ]]; then
    profile_file="${PERLMUTTER_GPU_PROFILE:-${HOME}/perlmutter_gpu_warpx.profile}"
    # shellcheck disable=SC1090
    source "${profile_file}"
fi

if [[ ! -x "${warpx_exe}" ]]; then
    echo "Missing executable: ${warpx_exe}" >&2
    exit 2
fi

mkdir -p "${result_dir}"

export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export MPICH_OFI_NIC_POLICY="${MPICH_OFI_NIC_POLICY:-GPU}"
export CRAY_ACCEL_TARGET="${CRAY_ACCEL_TARGET:-nvidia80}"
export AMREX_DEFAULT_INIT="${AMREX_DEFAULT_INIT:-amrex.use_gpu_aware_mpi=1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job=${SLURM_JOB_ID}"
    echo "nodes=${SLURM_JOB_NUM_NODES:-unknown}"
    echo "tasks=${SLURM_NTASKS:-unknown}"
    echo "host=$(hostname)"
    echo "warpx_commit=$(git -C "${warpx_dir}" rev-parse HEAD)"
    echo "executable=${warpx_exe}"
    echo "input=${input_file}"
    echo "nx_values=${nx_values[*]}"
    echo "nppc=${nppc}"
    echo "max_step=${max_step}"
    echo "nrepeat=${nrepeat}"
    echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
} > "${result_dir}/metadata.txt"
module list >> "${result_dir}/metadata.txt" 2>&1
nvidia-smi -L >> "${result_dir}/metadata.txt"

wall_csv="${result_dir}/wall_times.csv"
echo 'nx,grid_cells,yee_e_dofs,particles_two_species,repeat,wall_seconds' > "${wall_csv}"

for nx in "${nx_values[@]}"; do
    case_dir="${result_dir}/nx${nx}"
    mkdir -p "${case_dir}/wall"

    cells=$((nx*nx))
    # Ex(nx+1,nx), Ey(nx,nx), Ez(nx,nx+1) for a square 2D Yee mesh.
    yee_e_dofs=$((3*cells+2*nx))
    particles=$((2*cells*nppc*nppc))

    for ((rep=1; rep<=nrepeat; ++rep)); do
        log_file="${case_dir}/wall/rep${rep}.log"
        echo "Running clean case nx=${nx}, np=${nppc}, repeat=${rep}"
        srun --cpu-bind=cores bash -c '
            export CUDA_VISIBLE_DEVICES=$((3-SLURM_LOCALID))
            exec "$@"
        ' _ \
            "${warpx_exe}" "${input_file}" \
            "amr.n_cell=${nx}" "${nx}" \
            "warpx.do_device_synchronize=${DO_DEVICE_SYNCHRONIZE:-0}" \
            "my_constants.nppcz=${nppc}" \
            "max_step=${max_step}" \
            > "${log_file}" 2>&1

        wall_time="$(awk '/^Total Time[[:space:]]*:/{value=$NF} END{print value}' "${log_file}")"
        if [[ -z "${wall_time}" ]]; then
            echo "Could not find Total Time in ${log_file}" >&2
            exit 3
        fi
        echo "${nx},${cells},${yee_e_dofs},${particles},${rep},${wall_time}" >> "${wall_csv}"
    done
done

awk -F, '
    NR == 1 { next }
    {
        key = $1 FS $2 FS $3 FS $4
        count[key]++
        sum[key] += $6
        if (!(key in min) || $6 < min[key]) min[key] = $6
        if (!(key in max) || $6 > max[key]) max[key] = $6
    }
    END {
        print "nx,grid_cells,yee_e_dofs,particles_two_species,repeats,wall_min_s,wall_avg_s,wall_max_s"
        for (key in count) {
            print key FS count[key] FS min[key] FS sum[key]/count[key] FS max[key]
        }
    }
' "${wall_csv}" > "${result_dir}/wall_summary.unsorted.csv"
{
    head -n 1 "${result_dir}/wall_summary.unsorted.csv"
    tail -n +2 "${result_dir}/wall_summary.unsorted.csv" | sort -t, -k1,1n
} > "${result_dir}/wall_summary.csv"

if [[ "${run_nsys}" == "1" ]]; then
    for nx in "${nx_values[@]}"; do
        case_dir="${result_dir}/nx${nx}"
        nsys_dir="${case_dir}/nsys"
        echo "Profiling nx=${nx}, np=${nppc}"

        PROFILE_DIR="${nsys_dir}" \
        WARPX_EXE="${warpx_exe}" \
        TARGET_INPUT="${input_file}" \
        TARGET_ARGS="amr.n_cell=${nx} ${nx} my_constants.nppcz=${nppc} max_step=${max_step} warpx.do_device_synchronize=0" \
        NSYS_TRACE="${nsys_trace}" \
            bash "${profile_script}" > "${case_dir}/nsys.log" 2>&1

        for rank in 0 1 2 3; do
            report="${nsys_dir}/mlmg-rank${rank}.nsys-rep"
            stats="${nsys_dir}/rank${rank}-nvtx-gpu.txt"
            nsys stats \
                --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_kern_exec_sum \
                "${report}" > "${stats}"
            grep -E \
                'MLMG::mgVcycle_(down|up|bottom)::amr=|MLCurlCurl::kernel::(smooth4|smooth1D|compresid)::amr=|MLCurlCurl::(apply|smooth|restriction|interpolation)::amr=' \
                "${stats}" > "${nsys_dir}/rank${rank}-mlmg-markers.txt" || true
        done
    done
fi

rm -f "${result_dir}/wall_summary.unsorted.csv"

echo "Wall-time summary: ${result_dir}/wall_summary.csv"
echo "Nsight reports: ${result_dir}/nx*/nsys/mlmg-rank*.nsys-rep"
echo "NVTX/GPU reports: ${result_dir}/nx*/nsys/rank*-nvtx-gpu.txt"
