#!/usr/bin/env python3

"""Small, configurable MLMG benchmark based on a magnetic-mirror field."""

import os

from pywarpx import picmi


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


ncell = env_int("MLMG_NCELL", 256)
max_grid_size = env_int("MLMG_MAX_GRID_SIZE", 64)
blocking_factor = env_int("MLMG_BLOCKING_FACTOR", 8)
rtol = env_float("MLMG_RTOL", 5.0e-12)

# The first three terms form a divergence-free paraxial magnetic mirror.
# A small sinusoidal perturbation gives the projection cleaner non-zero work.
b0 = 1.0
alpha = 0.25
epsilon = 0.05
two_pi = 6.283185307179586
bx = f"-{alpha}*x*z + {epsilon}*sin({two_pi}*x)"
by = f"-{alpha}*y*z"
bz = f"{b0}*(1.0 + {alpha}*z*z)"

grid = picmi.Cartesian3DGrid(
    number_of_cells=[ncell, ncell, ncell],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=blocking_factor,
    lower_bound=[-1.0, -1.0, -1.0],
    upper_bound=[1.0, 1.0, 1.0],
    lower_boundary_conditions=["dirichlet", "dirichlet", "neumann"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "neumann"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
)

field = picmi.AnalyticInitialField(
    Bx_expression=bx,
    By_expression=by,
    Bz_expression=bz,
    warpx_do_initial_div_cleaning=True,
    warpx_projection_div_cleaner_rtol=rtol,
)

simulation = picmi.Simulation(
    solver=picmi.ElectrostaticSolver(grid=grid),
    max_steps=0,
    time_step_size=1.0e-9,
    verbose=1,
    warpx_do_dynamic_scheduling=False,
)
simulation.add_applied_field(field)

# Initializing WarpX loads the field and executes the projection-cleaner MLMG solve.
simulation.initialize_inputs()
simulation.initialize_warpx()
