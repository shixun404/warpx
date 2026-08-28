# SIPIC GPU performance reproduction

This directory contains the 3D SIPIC/PICMI problem used to compare GPU performance on
Curiosity and NERSC systems. The simulation entry point is `test_run1/run_simulation.py`;
`test_run1/warpx_used_inputs` and the `log_*.out` files are generated/reference output and
are not simulation entry points.

## Prerequisites

- A CUDA-capable GPU node
- CMake, CUDA, MPI, and a supported C++ compiler
- An activated Python virtual or Conda environment
- A separate AMReX source checkout

Set the AMReX checkout through the environment. The build script intentionally contains no
machine- or user-specific AMReX path.

```bash
export AMREX_DIR=/path/to/amrex
source /path/to/warpx-python-env/bin/activate
```

Record the source revisions before comparing timings:

```bash
git rev-parse HEAD
git -C "${AMREX_DIR}" rev-parse HEAD
```

## Build

Run the script from anywhere inside the WarpX checkout:

```bash
./WARPX_B300/build_warpx.sh clean
```

The default configuration builds 3D CUDA WarpX with single-precision fields and particles,
matching the copied build script. The build directory defaults to `build-roelof-sp` inside
the WarpX checkout. Optional environment variables are:

```text
WARPX_DIR             WarpX source checkout (defaults to the current Git repository root)
WARPX_BUILD_DIR       Build directory; it must remain inside WARPX_DIR
PYTHON_EXECUTABLE     Python executable (defaults to python on PATH)
PYTHON_ROOT_DIR       Python environment root (defaults to VIRTUAL_ENV or CONDA_PREFIX)
BUILD_JOBS            Parallel build jobs (defaults to 16)
```

The build enables `AMReX_DIFFERENT_COMPILER=ON` because the NVHPC host compiler and CUDA
compiler can otherwise trigger AMReX's compiler-consistency assertion.

Verify the installed Python module before running:

```bash
python -c "import pywarpx; print(pywarpx.__file__)"
python -c "from pywarpx import picmi; print('PICMI import OK')"
```

## Run

Run from the problem directory so that relative paths to `bfield_rz.npz` and
`sim_parameters.dpkl` resolve correctly:

```bash
cd WARPX_B300/test_run1
export OMP_NUM_THREADS=1
```

For an MPI launcher that maps node-local ranks to GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 mpirun -np 1 python -u run_simulation.py \
    2>&1 | tee repro_1gpu.out

CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 python -u run_simulation.py \
    2>&1 | tee repro_2gpu.out

CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 python -u run_simulation.py \
    2>&1 | tee repro_4gpu.out
```

On a Slurm system such as Perlmutter, use the site-recommended allocation and task launcher.
Within an allocated GPU node, the corresponding four-GPU launch is typically:

```bash
srun -N 1 -n 4 --gpus-per-task=1 --gpu-bind=closest \
    python -u run_simulation.py 2>&1 | tee repro_4gpu.out
```

Do not overwrite the copied `log_1gpu.out`, `log_2gpu.out`, and `log_4gpu.out` reference logs.
For reproducible comparisons, record the WarpX and AMReX commits, compiler, CUDA and MPI
versions, Python environment, precision, GPU model, and GPU/rank binding.
