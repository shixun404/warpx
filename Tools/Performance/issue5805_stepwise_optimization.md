# Issue #5805: Stepwise MLMG Coarse-Level Optimization

This document tracks a stepwise optimization study for WarpX issue #5805 on
Perlmutter.  It is intended to be both an execution checklist and a stable context
index for later discussion.

## 1. Scope

Target configuration:

- WarpX implicit electromagnetic solver in 2D.
- Four MPI ranks and four GPUs on one Perlmutter node.
- `np = 8` particles per cell per dimension.
- Primary diagnostic cases: `nx = 16` and `nx = 32`.
- Full comparison sweep: `nx = 16, 32, 64, 128, 256`.
- Curl-curl preconditioner: `pc_curl_curl_mlmg`.
- AMReX and WarpX branches: `hackathon-mlmg-nvtx`.

The main performance problem is that higher MG levels contain very little useful
GPU work, while MPI polling, synchronization, and CUDA kernel-launch overhead remain.

## 2. Important terminology

- **Single MPI rank:** the current MPI communicator contains one rank,
  `ParallelContext::NProcsSub() == 1`.
- **Single-owner MG level:** the MG level's `DistributionMapping` assigns all boxes
  to one rank, although the current communicator may still contain four ranks.
- **Active rank:** a rank that owns at least one box on the current MG level.
- **Empty MPI test:** `MPI_Testall` is called with an empty request vector or zero
  active receive requests.
- **No-sync execution:** kernels are enqueued without a host-side synchronization
  after each launch. Dependencies are preserved by using one CUDA stream.

These terms must not be treated as interchangeable. In particular, `mg=1` can be a
single-owner level while `NProcsSub()` is still four.

## 3. Current grid-layout behavior

AMReX defaults to:

```text
amr.refine_grid_layout = 1
```

It calls `ChopGrids(..., ParallelDescriptor::NProcs())`, so the base AMR level is
split into at least four boxes for a four-rank run even when `amr.max_grid_size` is
not specified.

Observed `mg=0` workloads:

| `nx` | Total cells | Number of boxes | Typical box size | Typical cells/rank |
|---:|---:|---:|---:|---:|
| 16 | 256 | 4 | `8 x 8` | 64 |
| 32 | 1,024 | 4 | `16 x 16` | 256 |
| 64 | 4,096 | 4 | `32 x 32` | 1,024 |
| 128 | 16,384 | 4 | `64 x 64` | 4,096 |
| 256 | 65,536 | 4 | `128 x 128` | 16,384 |

The current log does not print the BoxArray and DistributionMapping for every MG
level. A separate diagnostic should eventually print, for every `(amrlev, mglev)`:

- domain dimensions;
- number of boxes and total cells;
- owner of every box;
- boxes and cells per rank;
- number of active ranks;
- whether agglomeration or consolidation has occurred;
- whether adjacent levels require a redistribution in restriction/interpolation.

## 4. Measurement rules

### 4.1 Keep wall-time and Nsys configurations consistent

The Nsys pass currently includes:

```text
warpx.do_device_synchronize = 0
```

The clean wall-time runs must use the same value. Add the following run-time
argument to the clean-run command in
`Tools/Performance/run_issue5805_nx_sweep_perlmutter.sbatch`:

```bash
"warpx.do_device_synchronize=${DO_DEVICE_SYNCHRONIZE:-0}"
```

Alternatively, add the following temporarily to the input file:

```text
warpx.do_device_synchronize = 0
```

Do not compare variants that use different profiler synchronization settings unless
disabling profiler synchronization is itself the step being measured.

### 4.2 Use clean runs for wall time

Nsys instrumentation changes runtime. Use clean runs for the primary wall-time
comparison and a separate Nsys run to explain the result.

Recommended minimum:

```bash
NREPEAT=5 \
RUN_NSYS=0 \
bash Tools/Performance/run_issue5805_nx_sweep_perlmutter.sbatch
```

Then profile the most representative case separately. For MPI analysis, explicitly
include MPI because the sweep script's default trace list may omit it:

```bash
NX_VALUES="16 32" \
NREPEAT=1 \
RUN_NSYS=1 \
NSYS_TRACE=cuda,nvtx,mpi,osrt \
bash Tools/Performance/run_issue5805_nx_sweep_perlmutter.sbatch
```

### 4.3 Correctness gate for every step

Do not proceed to the next step until the current variant passes:

- no CUDA or MPI errors;
- same MLMG iteration count;
- comparable initial and final residuals;
- no new convergence failures;
- final fields agree within the expected floating-point tolerance;
- results are repeatable across at least three runs.

### 4.4 Metrics to record

For every variant record:

- total WarpX wall time;
- inclusive and exclusive `MLMG::solve()` time;
- inclusive `CurlCurlMLMGPC::Apply()` time;
- `MPI_Testall`, `MPI_Isend`, `MPI_Irecv`, and `MPI_Waitall` counts and time;
- `cudaStreamSynchronize` and `cudaDeviceSynchronize` counts and time;
- `cudaLaunchKernel` count and API time;
- GPU kernel execution time;
- kernel-to-kernel idle gaps;
- CUDA Graph launch count and time when applicable.

## 5. Experiment matrix

| ID | Change | Purpose |
|---|---|---|
| B0 | Unmodified execution with fixed measurement settings | Reproducible baseline |
| S1 | Skip `MPI_Testall` when the request vector is empty | Remove empty MPI polling |
| S2a | Disable profiler device synchronization | Remove instrumentation-induced waits |
| S2b | Enable MLMG no-GPU-sync mode | Enqueue dependent kernels without per-kernel host waits |
| S3 | No separate code change; characterize asynchronous enqueue | Confirm that no-sync already provides async submission |
| S4 | Add CUDA Graph for a fixed single-owner smoother sequence | Reduce CUDA launch overhead |
| S5 | Optional safe kernel fusion | Reduce launch count further if Graph is insufficient |

Each result directory should include the experiment ID, for example:

```bash
export RESULT_DIR="${HACK_ROOT}/profiles/issue5805-${SLURM_JOB_ID}-S1"
```

## 6. B0: Baseline

### Changes

No optimization changes. Fix all measurement inputs and record the exact WarpX and
AMReX commit hashes.

### Run

```bash
git -C "${WARPX_DIR}" rev-parse HEAD
git -C "${AMREX_DIR}" rev-parse HEAD

NREPEAT=5 \
RUN_NSYS=0 \
bash Tools/Performance/run_issue5805_nx_sweep_perlmutter.sbatch
```

Run one MPI-aware Nsys profile for `nx=16` and one for `nx=32`.

### Expected observation

- `mg=0` has four boxes and real inter-rank halo exchanges.
- Higher MG levels may have a single owner while still using the four-rank
  communicator.
- Coarse-level CUDA kernels are very short.
- MPI polling, synchronization, and launch overhead can dominate useful work.

## 7. S1: Skip empty `MPI_Testall`

### Objective

Remove MPI calls that test an empty receive-request vector. Do not remove tests for
real outstanding communication.

### Implementation

MLCurlCurl uses the batched `FillBoundary(Vector<MultiFab*>)` path. In
`AMReX_FabArrayCommI.H`, guard both batched calls to `ParallelDescriptor::Test`:

```cpp
#if !defined(AMREX_DEBUG)
    int recv_flag = 1;
    if (!recv_reqs.empty()) {
        ParallelDescriptor::Test(recv_reqs, recv_flag, recv_stat);
    }
#endif
```

Apply the same guard after local-copy work:

```cpp
#if !defined(AMREX_DEBUG)
    if (!recv_reqs.empty()) {
        ParallelDescriptor::Test(recv_reqs, recv_flag, recv_stat);
    }
#endif
```

Also guard the scalar `FabArray::FillBoundary_test()` path:

```cpp
if (!fbd->recv_reqs.empty()) {
    int flag;
    ParallelDescriptor::Test(fbd->recv_reqs, flag, fbd->recv_stat);
}
```

### Safety rule

Do not skip `MPI_Testall` merely because only one rank owns the MG level. If
`recv_reqs` is non-empty, the test is performing real progress and must remain.

### Expected result

- empty `MPI_Testall` calls disappear;
- `MPI_Isend/Irecv` counts for real communication do not change;
- numerical results do not change;
- total speedup is likely small because this step only removes short CPU API calls.

### Decision gate

Proceed if MPI traces confirm that only empty tests were removed. Revert if any
communication request remains incomplete or iteration behavior changes.

## 8. S2a: Disable profiler device synchronization

### Objective

Remove synchronization inserted only to make TinyProfiler/NVTX region timing
synchronous with the GPU.

### Configuration

```text
warpx.do_device_synchronize = 0
```

Apply this consistently to clean and Nsys runs.

### Expected result

- fewer `cudaDeviceSynchronize` calls;
- NVTX CPU ranges may end before their asynchronous GPU kernels complete;
- use `nvtx_gpu_proj_sum` or kernel-launch correlation rather than visual CPU-range
  overlap to attribute GPU work.

## 9. S2b: Enable MLMG no-GPU-sync mode

### Objective

Remove repeated `Gpu::streamSynchronize()` calls while preserving kernel ordering.

### Recommended quick implementation

Add a WarpX parameter:

```text
pc_curl_curl_mlmg.no_gpu_sync = true
```

Add a member to `CurlCurlMLMGPC`:

```cpp
bool m_no_gpu_sync = false;
```

Read it in `readParameters()`:

```cpp
pp.query("no_gpu_sync", m_no_gpu_sync);
```

After constructing `m_solver`, call:

```cpp
m_solver->setNoGpuSync(m_no_gpu_sync);
```

AMReX then enables both:

```cpp
Gpu::setSingleStreamRegion(true);
Gpu::setNoSyncRegion(true);
```

Single-stream execution preserves the dependency order between color 0, 1, 2, and
3. No host synchronization is required between kernels on the same stream.

### First A/B test

For the fastest proof of concept, enable no-sync for the whole MLMG solve. If it is
correct and beneficial, implement a more selective single-owner-level version.

### Selective single-owner detection

Do not use local box count. Determine ownership from the global mapping:

```text
unique(m_dmap[amrlev][mglev].ProcessorMap()).size() == 1
```

This produces the same decision on every rank without an MPI collective.

### Level-boundary safety

If no-sync is scoped only to a single-owner level, synchronize once before a later
operation that redistributes its result to multiple ranks:

```text
many local kernels without host waits
    -> one synchronization at the level boundary
    -> MPI ParallelCopy/interpolation/restriction
```

This consolidates many waits into one safe boundary wait.

### Expected result

- large reduction in `cudaStreamSynchronize` count;
- fewer GPU idle gaps between short kernels;
- unchanged CUDA kernel-launch count;
- unchanged numerical result.

## 10. S3: Characterize asynchronous execution

No new asynchronous kernel API is required. CUDA kernel launches through
`amrex::ParallelFor` are already asynchronous.

Before no-sync:

```text
launch K0 -> wait -> launch K1 -> wait -> launch K2 -> wait
```

After no-sync:

```text
host: launch K0 -> launch K1 -> launch K2
GPU:       K0 -> K1 -> K2
```

The CPU must still execute every `cudaLaunchKernel`, so asynchronous enqueue does
not eliminate launch overhead.

### Multiple-stream warning

Do not place the four smoother colors on independent streams without explicit
events. Each color depends on the updates from the previous color. Multiple streams
would add synchronization complexity without reducing the number of launches.

### Possible later fusion targets

Independent component kernels in restriction/interpolation may be candidates for
fusion. The complete four-color sequence is not a simple fusion candidate because
it requires inter-color global ordering and ghost updates.

## 11. S4: CUDA Graph for a single-owner smoother

### Objective

Replace many individual host kernel launches with one `cudaGraphLaunch` while
preserving the existing GPU dependency order.

### Initial capture scope

Capture one fixed four-color smoother iteration, not the whole V-cycle:

```text
local FillBoundary
    -> smooth color 0
    -> local FillBoundary
    -> smooth color 1
    -> local FillBoundary
    -> smooth color 2
    -> local FillBoundary
    -> smooth color 3
```

Default pre- and post-smoothing use two iterations, so the graph can be replayed
twice.

### Do not initially capture

- MPI communication;
- restriction/interpolation across different DistributionMappings;
- host-side residual or convergence decisions;
- reductions that return scalars to the host;
- the complete bottom Krylov solver;
- variable-iteration control flow.

### Required enable conditions

Enable the graph only when all are true:

- the MG level has one unique owner;
- all FillBoundary operations in the captured segment have zero remote sends and
  receives;
- BoxArray and DistributionMapping are unchanged;
- device pointers are stable;
- `niter` and `skip_fillboundary` behavior are fixed;
- the sequence contains no stream/device synchronization;
- all operations are CUDA stream-capture safe.

### Graph cache key

At minimum distinguish:

- AMR level and MG level;
- pre-, post-, and bottom-smoothing role;
- solution and RHS device pointers;
- BoxArray and DistributionMapping identity;
- number of smoothing iterations;
- `skip_fillboundary` state.

A cache keyed only by `mglev` is unsafe because the same level can operate on
different MultiFab allocations.

### AMReX infrastructure

Relevant existing components:

- `AMReX_CudaGraph.H`: `CudaGraph<T>`;
- `AMReX_GpuDevice.H`: graph recording and execution APIs;
- `AMReX_GpuControl.H`: `GraphSafeGuard`;
- local FillBoundary CUDA Graph helpers in `AMReX_FabArrayCommI.H`.

These are building blocks, not an existing MLCurlCurl run-time switch.

### Expected result

- many `cudaLaunchKernel` API calls are replaced by `cudaGraphLaunch`;
- host launch/API time decreases;
- coarse-level GPU idle gaps decrease;
- graph instantiation cost is paid once and amortized across repeated V-cycles and
  solves.

### Decision gate

Keep the graph implementation only if replay occurs enough times to amortize graph
capture and instantiation. A graph containing only one small kernel is unlikely to
help.

## 12. S5: Optional kernel fusion

Consider fusion only after S4 measurements. CUDA Graph preserves separate kernels
but reduces host launch overhead. Fusion can also reduce device-side scheduling and
memory traffic, but is more invasive.

Potential targets:

- component-wise restriction kernels with compatible iteration spaces;
- component-wise interpolation kernels;
- small setup or residual kernels that use the same layout and have no intervening
  dependency.

Do not naively fuse the four smoother colors into a normal kernel. Their global
dependency requires grid-wide synchronization that a normal CUDA kernel does not
provide.

## 13. Nsys checks

### MPI

Run with:

```bash
NSYS_TRACE=cuda,nvtx,mpi,osrt
```

Check whether `MPI_Testall` events have zero or nonzero request counts, and record
their enclosing NVTX ranges.

### CUDA API

Compare:

```text
cudaStreamSynchronize
cudaDeviceSynchronize
cudaLaunchKernel
cudaGraphLaunch
```

### NVTX ranges

Focus on:

```text
MLMG::mgVcycle_down::amr=0::mg=*
MLMG::mgVcycle_up::amr=0::mg=*
MLMG::mgVcycle_bottom::amr=0::mg=*
MLCurlCurl::smooth::amr=0::mg=*
MLCurlCurl::restriction::*
MLCurlCurl::interpolation::*
MLCurlCurl::kernel::smooth4::*
```

When `warpx.do_device_synchronize=0`, CPU NVTX rectangle boundaries do not
necessarily overlap the asynchronous GPU execution in time. Use GPU-projected NVTX
reports and CUDA launch correlation.

## 14. Result record

Fill one row for each case and variant.

| ID | Commit(s) | nx | Wall avg (s) | MLMG incl. (s) | Testall count/time | Stream sync count/time | Launch count/time | Graph launches | Correct? | Notes |
|---|---|---:|---:|---:|---|---|---|---:|---|---|
| B0 | | 16 | | | | | | 0 | | |
| S1 | | 16 | | | | | | 0 | | |
| S2a | | 16 | | | | | | 0 | | |
| S2b | | 16 | | | | | | 0 | | |
| S4 | | 16 | | | | | | | | |
| B0 | | 32 | | | | | | 0 | | |
| S1 | | 32 | | | | | | 0 | | |
| S2a | | 32 | | | | | | 0 | | |
| S2b | | 32 | | | | | | 0 | | |
| S4 | | 32 | | | | | | | | |

## 15. Expected benefit order

The initial hypothesis is:

```text
no-sync benefit > CUDA Graph benefit > empty MPI_Testall benefit
```

This is only a hypothesis. The measured upper bound on CUDA Graph benefit is the
fraction of time currently spent in launch/API gaps. CUDA Graph will not fix low
occupancy inside an individual kernel, and removing empty MPI tests will not reduce
CUDA launch overhead.

## 16. Context index for discussion

Use the following labels when discussing results:

- **GRID:** AMR/MG BoxArray and DistributionMapping diagnostics.
- **B0:** fixed baseline.
- **S1:** empty `MPI_Testall` removal.
- **S2a:** TinyProfiler device-sync removal.
- **S2b:** MLMG no-sync execution.
- **S3:** asynchronous enqueue characterization; no separate implementation.
- **S4:** CUDA Graph smoother prototype.
- **S5:** optional kernel fusion.
- **CORRECTNESS:** residual, iteration, and field comparison.
- **MPI:** request counts, communication peers, and MPI time.
- **CUDA-API:** launch and synchronization API time.
- **GPU-GAPS:** idle gaps on the GPU timeline.

Example follow-up references:

```text
S1 removes Testall calls but MPI wait time is unchanged.
S2b reduces stream syncs but CORRECTNESS fails at mg=0 -> mg=1.
S4 graph replay works for mg=2 but its instantiation cost is not amortized.
```
