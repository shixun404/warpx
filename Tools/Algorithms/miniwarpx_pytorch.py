#!/usr/bin/env python3
"""A minimal electrostatic Particle-In-Cell code with a geometric multigrid solver.

This is a teaching model for the main WarpX dataflow and the MLMG-focused NERSC
hackathon work.  It is not a numerical replacement for WarpX or AMReX MLMG.

The state consists of

* continuous particle positions and velocities; and
* charge, potential and electric fields on a periodic Cartesian mesh.

Each time step follows the core electrostatic PIC cycle::

    particles --CIC deposit--> rho --multigrid Poisson--> phi --> E
        ^                                                        |
        +------------------ CIC gather + push -------------------+

The multigrid V-cycle contains the same conceptual phases that matter in MLMG:

    smooth -> residual -> restrict -> coarse solve -> prolong -> correct -> smooth

Examples
--------

Run a small CPU case::

    python Tools/Algorithms/miniwarpx_pytorch.py --device cpu

Run on a GPU and compare precision::

    python Tools/Algorithms/miniwarpx_pytorch.py --device cuda --dtype float32
    python Tools/Algorithms/miniwarpx_pytorch.py --device cuda --dtype float64

Increase the grid and particles per cell::

    python Tools/Algorithms/miniwarpx_pytorch.py \
        --device cuda --grid 256 --particles-per-cell 16 --steps 100

Outputs are intentionally analogous to common WarpX outputs:

* stdout: per-step status and component timings;
* diagnostics.csv: reduced diagnostics; and
* final_state.pt: a compact field/particle snapshot and restart-like state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypeVar

import torch


Tensor = torch.Tensor
T = TypeVar("T")


@dataclass
class Config:
    """Dimensionless numerical and physical parameters for the toy problem."""

    grid: int = 64
    particles_per_cell: int = 4
    steps: int = 40
    dt: float = 0.05
    length: float = 2.0 * math.pi
    perturbation: float = 0.05
    mg_tolerance: float = 1.0e-6
    mg_max_cycles: int = 12
    mg_pre_smooth: int = 3
    mg_post_smooth: int = 3
    mg_coarse_smooth: int = 80
    jacobi_omega: float = 2.0 / 3.0
    report_every: int = 5
    warm_start: bool = True
    seed: int = 2026


@dataclass
class Timing:
    deposit_ms: float = 0.0
    solve_ms: float = 0.0
    gather_push_ms: float = 0.0


@dataclass
class CICStencil:
    """Four grid indices and interpolation weights for every particle."""

    indices: tuple[Tensor, Tensor, Tensor, Tensor]
    weights: tuple[Tensor, Tensor, Tensor, Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal PyTorch electrostatic PIC + geometric multigrid tutorial"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--grid", type=int, default=Config.grid)
    parser.add_argument("--particles-per-cell", type=int, default=Config.particles_per_cell)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--perturbation", type=float, default=Config.perturbation)
    parser.add_argument("--mg-tolerance", type=float, default=Config.mg_tolerance)
    parser.add_argument("--mg-max-cycles", type=int, default=Config.mg_max_cycles)
    parser.add_argument("--mg-pre-smooth", type=int, default=Config.mg_pre_smooth)
    parser.add_argument("--mg-post-smooth", type=int, default=Config.mg_post_smooth)
    parser.add_argument("--mg-coarse-smooth", type=int, default=Config.mg_coarse_smooth)
    parser.add_argument("--report-every", type=int, default=Config.report_every)
    parser.add_argument("--cold-start", action="store_true", help="Start phi=0 every step")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-dir", type=Path, default=Path("miniwarpx_output"))
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def validate_config(config: Config) -> None:
    if config.grid < 16 or config.grid & (config.grid - 1):
        raise ValueError("--grid must be a power of two and at least 16")
    if config.particles_per_cell < 1:
        raise ValueError("--particles-per-cell must be positive")
    if config.steps < 1 or config.dt <= 0.0:
        raise ValueError("--steps and --dt must be positive")
    if not 0.0 <= config.perturbation < 0.5:
        raise ValueError("--perturbation must be in [0, 0.5)")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, function: Callable[[], T]) -> tuple[T, float]:
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, 1.0e3 * (time.perf_counter() - start)


def remove_mean(value: Tensor) -> Tensor:
    """Project out the constant null-space of periodic Poisson."""

    return value - value.mean()


def apply_negative_laplacian(phi: Tensor, spacing: float) -> Tensor:
    """Apply -Laplacian(phi) with a five-point periodic stencil."""

    neighbors = (
        torch.roll(phi, 1, 0)
        + torch.roll(phi, -1, 0)
        + torch.roll(phi, 1, 1)
        + torch.roll(phi, -1, 1)
    )
    return (4.0 * phi - neighbors) / (spacing * spacing)


def weighted_jacobi(
    phi: Tensor,
    rhs: Tensor,
    spacing: float,
    iterations: int,
    omega: float,
) -> Tensor:
    """Damp short-wavelength error; conceptually an MLMG smoother kernel."""

    h2 = spacing * spacing
    for _ in range(iterations):
        neighbors = (
            torch.roll(phi, 1, 0)
            + torch.roll(phi, -1, 0)
            + torch.roll(phi, 1, 1)
            + torch.roll(phi, -1, 1)
        )
        jacobi_update = 0.25 * (neighbors + h2 * rhs)
        phi = remove_mean(phi + omega * (jacobi_update - phi))
    return phi


def restrict_full_weighting(fine: Tensor) -> Tensor:
    """Restrict a nodal fine residual with the periodic 2D full-weighting stencil."""

    center = fine[0::2, 0::2]
    north = torch.roll(fine, 1, 0)[0::2, 0::2]
    south = torch.roll(fine, -1, 0)[0::2, 0::2]
    west = torch.roll(fine, 1, 1)[0::2, 0::2]
    east = torch.roll(fine, -1, 1)[0::2, 0::2]
    northwest = torch.roll(torch.roll(fine, 1, 0), 1, 1)[0::2, 0::2]
    northeast = torch.roll(torch.roll(fine, 1, 0), -1, 1)[0::2, 0::2]
    southwest = torch.roll(torch.roll(fine, -1, 0), 1, 1)[0::2, 0::2]
    southeast = torch.roll(torch.roll(fine, -1, 0), -1, 1)[0::2, 0::2]
    return (
        4.0 * center
        + 2.0 * (north + south + west + east)
        + northwest
        + northeast
        + southwest
        + southeast
    ) / 16.0


def prolong_bilinear(coarse: Tensor) -> Tensor:
    """Periodically bilinear-interpolate a coarse correction to the fine grid."""

    coarse_size = coarse.shape[0]
    fine = torch.empty(
        (2 * coarse_size, 2 * coarse_size), dtype=coarse.dtype, device=coarse.device
    )
    right = torch.roll(coarse, -1, 1)
    down = torch.roll(coarse, -1, 0)
    down_right = torch.roll(down, -1, 1)

    fine[0::2, 0::2] = coarse
    fine[0::2, 1::2] = 0.5 * (coarse + right)
    fine[1::2, 0::2] = 0.5 * (coarse + down)
    fine[1::2, 1::2] = 0.25 * (coarse + right + down + down_right)
    return fine


def multigrid_v_cycle(phi: Tensor, rhs: Tensor, spacing: float, config: Config) -> Tensor:
    """Apply one recursive correction-scheme geometric multigrid V-cycle."""

    if phi.shape[0] <= 4:
        return weighted_jacobi(
            phi,
            rhs,
            spacing,
            config.mg_coarse_smooth,
            config.jacobi_omega,
        )

    phi = weighted_jacobi(
        phi, rhs, spacing, config.mg_pre_smooth, config.jacobi_omega
    )
    residual = remove_mean(rhs - apply_negative_laplacian(phi, spacing))

    coarse_rhs = remove_mean(restrict_full_weighting(residual))
    coarse_error = torch.zeros_like(coarse_rhs)
    coarse_error = multigrid_v_cycle(coarse_error, coarse_rhs, 2.0 * spacing, config)

    phi = remove_mean(phi + prolong_bilinear(coarse_error))
    return weighted_jacobi(
        phi, rhs, spacing, config.mg_post_smooth, config.jacobi_omega
    )


def solve_poisson_multigrid(
    rho: Tensor,
    spacing: float,
    config: Config,
    initial_phi: Tensor | None,
) -> tuple[Tensor, int, float]:
    """Solve -Laplacian(phi)=rho and report V-cycle count and relative residual."""

    rhs = remove_mean(rho)
    rhs_norm = torch.linalg.vector_norm(rhs)
    if float(rhs_norm) == 0.0:
        return torch.zeros_like(rhs), 0, 0.0

    if initial_phi is None or not config.warm_start:
        phi = torch.zeros_like(rhs)
    else:
        phi = remove_mean(initial_phi)

    relative_residual = math.inf
    cycle = 0
    for cycle in range(1, config.mg_max_cycles + 1):
        phi = multigrid_v_cycle(phi, rhs, spacing, config)
        residual = rhs - apply_negative_laplacian(phi, spacing)
        relative_residual = float(torch.linalg.vector_norm(residual) / rhs_norm)
        if relative_residual <= config.mg_tolerance:
            break

    return phi, cycle, relative_residual


def electric_field(phi: Tensor, spacing: float) -> tuple[Tensor, Tensor]:
    """Compute E=-grad(phi) with centered periodic differences."""

    ex = -0.5 * (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / spacing
    ey = -0.5 * (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / spacing
    return ex, ey


def cic_stencil(positions: Tensor, grid: int, length: float) -> CICStencil:
    """Build cloud-in-cell indices/weights for particle-grid coupling."""

    spacing = length / grid
    grid_position = positions / spacing
    lower = torch.floor(grid_position).to(torch.int64)
    fraction = grid_position - lower

    ix0 = lower[:, 0].remainder(grid)
    iy0 = lower[:, 1].remainder(grid)
    ix1 = (ix0 + 1).remainder(grid)
    iy1 = (iy0 + 1).remainder(grid)
    fx = fraction[:, 0]
    fy = fraction[:, 1]

    return CICStencil(
        indices=(iy0 * grid + ix0, iy0 * grid + ix1, iy1 * grid + ix0, iy1 * grid + ix1),
        weights=((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy),
    )


def deposit_charge(
    stencil: CICStencil,
    grid: int,
    spacing: float,
    electron_macro_charge: float,
    ion_background_density: float,
    template: Tensor,
) -> Tensor:
    """Deposit electron charge with CIC and add a uniform immobile ion background."""

    rho_flat = torch.full(
        (grid * grid,), ion_background_density, dtype=template.dtype, device=template.device
    )
    charge_density_scale = electron_macro_charge / (spacing * spacing)
    for indices, weights in zip(stencil.indices, stencil.weights, strict=True):
        rho_flat.scatter_add_(0, indices, charge_density_scale * weights)
    return rho_flat.reshape(grid, grid)


def gather_field(field: Tensor, stencil: CICStencil) -> Tensor:
    """Gather one grid field component to particles using CIC."""

    flat = field.reshape(-1)
    result = torch.zeros_like(stencil.weights[0])
    for indices, weights in zip(stencil.indices, stencil.weights, strict=True):
        result = result + weights * flat[indices]
    return result


def initialize_particles(
    config: Config, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Create a quiet electron distribution with one sinusoidal density perturbation."""

    torch.manual_seed(config.seed)
    n = config.grid
    ppc = config.particles_per_cell
    spacing = config.length / n

    coordinate = (torch.arange(n, device=device, dtype=dtype) + 0.5) * spacing
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    cell_centers = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    positions = cell_centers.repeat_interleave(ppc, dim=0)

    jitter = (torch.rand(positions.shape, device=device, dtype=dtype) - 0.5) * 0.8 * spacing
    positions = torch.remainder(positions + jitter, config.length)

    wave_number = 2.0 * math.pi / config.length
    displacement = (config.perturbation / wave_number) * torch.sin(
        wave_number * positions[:, 0]
    )
    positions[:, 0] = torch.remainder(positions[:, 0] + displacement, config.length)
    velocities = torch.zeros_like(positions)
    return positions, velocities


def diagnostic_row(
    step: int,
    positions: Tensor,
    velocities: Tensor,
    rho: Tensor,
    ex: Tensor,
    ey: Tensor,
    spacing: float,
    particle_mass: float,
    cycles: int,
    residual: float,
    timing: Timing,
) -> dict[str, float | int]:
    kinetic = 0.5 * particle_mass * torch.sum(velocities * velocities)
    field = 0.5 * spacing * spacing * torch.sum(ex * ex + ey * ey)
    total = kinetic + field
    return {
        "step": step,
        "particles": positions.shape[0],
        "net_charge": float(spacing * spacing * rho.sum()),
        "kinetic_energy": float(kinetic),
        "field_energy": float(field),
        "total_energy": float(total),
        "mg_cycles": cycles,
        "mg_relative_residual": residual,
        "deposit_ms": timing.deposit_ms,
        "solve_ms": timing.solve_ms,
        "gather_push_ms": timing.gather_push_ms,
    }


def write_diagnostics(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(config: Config, device: torch.device, dtype: torch.dtype, output_dir: Path) -> None:
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    spacing = config.length / config.grid
    positions, velocities = initialize_particles(config, device, dtype)
    particle_count = positions.shape[0]

    # Dimensionless normalization: n_i=1, |q_e|=m_e=epsilon_0=1.
    # Each macroparticle represents an equal portion of the physical electron density.
    electron_macro_charge = -(config.length * config.length) / particle_count
    particle_mass = (config.length * config.length) / particle_count
    charge_to_mass = -1.0

    print("Mini-WarpX: 2D electrostatic PIC + geometric multigrid")
    print(
        f"device={device} dtype={dtype} grid={config.grid}x{config.grid} "
        f"particles={particle_count:,} dt={config.dt} warm_start={config.warm_start}"
    )
    print(
        "step    field_E      kinetic_E    total_E      "
        "MG cycles/residual    deposit  solve  push"
    )

    phi: Tensor | None = None
    rows: list[dict[str, float | int]] = []
    accumulated = Timing()
    last_rho = last_ex = last_ey = None

    for step in range(config.steps + 1):
        def build_charge_density() -> tuple[CICStencil, Tensor]:
            particle_stencil = cic_stencil(positions, config.grid, config.length)
            charge_density = deposit_charge(
                particle_stencil,
                config.grid,
                spacing,
                electron_macro_charge,
                1.0,
                positions,
            )
            return particle_stencil, charge_density

        (stencil, rho), deposit_ms = timed(
            device,
            build_charge_density,
        )

        (phi, cycles, residual), solve_ms = timed(
            device,
            lambda: solve_poisson_multigrid(rho, spacing, config, phi),
        )
        ex, ey = electric_field(phi, spacing)

        timing = Timing(deposit_ms=deposit_ms, solve_ms=solve_ms)
        row = diagnostic_row(
            step,
            positions,
            velocities,
            rho,
            ex,
            ey,
            spacing,
            particle_mass,
            cycles,
            residual,
            timing,
        )
        rows.append(row)
        last_rho, last_ex, last_ey = rho, ex, ey

        accumulated.deposit_ms += deposit_ms
        accumulated.solve_ms += solve_ms
        if step < config.steps:
            def gather_and_push() -> None:
                particle_ex = gather_field(ex, stencil)
                particle_ey = gather_field(ey, stencil)
                particle_e = torch.stack((particle_ex, particle_ey), dim=1)
                velocities.add_(charge_to_mass * config.dt * particle_e)
                positions.copy_(
                    torch.remainder(positions + config.dt * velocities, config.length)
                )

            _, gather_push_ms = timed(device, gather_and_push)
            timing.gather_push_ms = gather_push_ms
            rows[-1]["gather_push_ms"] = gather_push_ms
            accumulated.gather_push_ms += gather_push_ms

        if step % config.report_every == 0 or step == config.steps:
            print(
                f"{step:4d}  {row['field_energy']:11.4e}  {row['kinetic_energy']:11.4e}  "
                f"{row['total_energy']:11.4e}  {cycles:3d}/{residual:8.2e}  "
                f"{deposit_ms:7.2f} {solve_ms:7.2f} {timing.gather_push_ms:7.2f} ms"
            )

    write_diagnostics(output_dir / "diagnostics.csv", rows)
    torch.save(
        {
            "config": asdict(config),
            "positions": positions.cpu(),
            "velocities": velocities.cpu(),
            "rho": last_rho.cpu(),
            "phi": phi.cpu(),
            "Ex": last_ex.cpu(),
            "Ey": last_ey.cpu(),
        },
        output_dir / "final_state.pt",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    measured_steps = config.steps + 1
    print("\nAverage component time")
    print(f"  deposit:     {accumulated.deposit_ms / measured_steps:9.3f} ms")
    print(f"  MG solve:    {accumulated.solve_ms / measured_steps:9.3f} ms")
    print(f"  gather/push: {accumulated.gather_push_ms / config.steps:9.3f} ms")
    print(f"Outputs: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    config = Config(
        grid=args.grid,
        particles_per_cell=args.particles_per_cell,
        steps=args.steps,
        dt=args.dt,
        perturbation=args.perturbation,
        mg_tolerance=args.mg_tolerance,
        mg_max_cycles=args.mg_max_cycles,
        mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth,
        mg_coarse_smooth=args.mg_coarse_smooth,
        report_every=args.report_every,
        warm_start=not args.cold_start,
        seed=args.seed,
    )
    run(config, device, dtype, args.output_dir)


if __name__ == "__main__":
    main()
