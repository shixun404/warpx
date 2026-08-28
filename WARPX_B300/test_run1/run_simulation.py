#!/usr/bin/env python3
#
# --- Script to set up a mirror plasma inside a cylindrical vessel.
#

import os
slurm_id = os.environ.get('SLURM_LOCALID', None)
# if slurm_id is not None:
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(0) # os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] 
    # str(
    #    int(os.environ.get('SLURM_GPUS_PER_NODE', 4)) - 1 - int(slurm_id)
    # )

# os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

import argparse
import sys

import dill
import numpy as np
from scipy import constants

from pywarpx import picmi

import sim_utils
from bfield_loader import BFieldLoader


class SimulationSetup(object):

    # Vessel parameters
    R_MAX = 0.4        # m
    Z_MAX = 1.5        # m
    R_CV  = 0.3        # m
    Z_CV  = 0.8        # m
    R_THROAT = 0.06    # m
    Z_THROAT = 0.88    # m

    # Mesh parameters
    NZ = 352
    NX = NY = 128

    # Temporal domain (if not run as a test)
    LT = 0.1e-3  # s
    DT = 1e-9  # s
    DIAG_TIME = 0.5e-6 # s

    # Particle parameters
    T_ION = 50  # eV
    T_ELEC = 50 # eV
    ION_SPECIES = "D"
    ELEC_MASS_FAC = 1.0
    NPPC = 250
    N0 = 1e19  # m^-3
    LZ_SCALE = Z_CV
    LR_SCALE = 0.2  # m

    # Solver parameters
    N_FLOOR = N0 / 100.0  # m^-3
    C_SI = 10

    def __init__(self, test, bfield_path="bfield_rz.npz"):
        self.test = test

        # modify spatial resolution and endtime if run as a test
        if self.test:
            # self.NZ = 128
            # self.NX = self.NY = 64
            self.LT = 100 * self.DT
            self.DIAG_TIME = 100*self.DT
            self.NPPC = 250

        self.dx = 2.0 * self.R_MAX / (self.NX - 7)
        self.dy = 2.0 * self.R_MAX / (self.NY - 7)
        self.dz = 2.0 * self.Z_MAX / (self.NZ - 3)

        self.LX = self.dx * self.NX
        self.LY = self.dy * self.NY
        self.LZ = self.dz * self.NZ

        # set wall configuration
        self.wall_rz_points = [
            (self.R_CV,                           0.0),
            (self.R_CV,                           self.Z_CV),
            (0.75*self.R_THROAT + 0.25*self.R_CV, self.Z_CV),
            (self.R_THROAT,                       0.25*self.Z_THROAT + 0.75*self.Z_CV),
            (self.R_THROAT,                       1.5*self.Z_THROAT - 0.5*self.Z_CV),
            ((self.R_THROAT + self.R_CV)/2,       2*self.Z_THROAT - self.Z_CV),
            (self.R_MAX,                          2*self.Z_THROAT - self.Z_CV),
            (self.R_MAX,                          self.Z_MAX),
        ]

        # load the precomputed external magnetic field
        self.bfield_loader = BFieldLoader(bfield_path)

        # number of steps between diagnostics / total steps
        self.max_steps = int(round(self.LT / self.DT))
        self.diag_steps = max(1, int(round(self.DIAG_TIME / self.DT)))

        self.calculate_plasma_parameters()

        # dump all the current attributes to a dill pickle file
        if sim_utils.get_rank() == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        self.setup_run()

    def calculate_plasma_parameters(self):
        """Function to collect and print plasma parameters for the current
        run. Also prints standard PIC parameters."""
        # species mass
        self.ion_mass = (
            2.0 if self.ION_SPECIES == 'D' else 1.0
        ) * constants.m_p
        self.electron_mass = self.ELEC_MASS_FAC * constants.m_e

        # get plasma frequency and SIPIC adjusted frequency
        self.plasma_freq = sim_utils.plasma_frequency(self.N0, self.electron_mass)
        self.plasma_freq_SIPIC = self.plasma_freq / np.sqrt(
            1 + self.C_SI * self.plasma_freq**2 * np.pi**2 * self.DT**2
        )

        # get thermal speeds
        self.v_th_i = sim_utils.thermal_velocity(self.T_ION, self.ion_mass)
        self.v_th_e = sim_utils.thermal_velocity(self.T_ELEC, self.electron_mass)
        self.cfl = self.v_th_e * self.DT / min(self.dx, self.dy, self.dz)

        # magnetic field properties (pulled from the precomputed npz field)
        self.B_midplane = self.bfield_loader.B_midplane
        self.B_plug = self.bfield_loader.B_plug

        # ion Larmor orbit sizes
        self.rho_i_midplane = sim_utils.larmor_radius(
            T=self.T_ION, B=self.B_midplane, m=self.ion_mass, Z=1
        )
        self.rho_i_plug = sim_utils.larmor_radius(
            T=self.T_ION, B=self.B_plug, m=self.ion_mass, Z=1
        )

        # ion cyclotron frequency at the mirror plug
        self.omega_ci_plug = sim_utils.cyclotron_frequency(
            B=self.B_plug, m=self.ion_mass, Z=1
        )

        # Coulomb collision frequencies (intra + inter species)
        self.nu_e, _, self.nu_i = sim_utils.collision_frequency(
            n_e=self.N0, T_e=self.T_ELEC, n_i=self.N0, T_i=self.T_ION,
            m_i=self.ion_mass, Z=1
        )

        # gas-dynamic trap confinement time: tau = 0.5 * R * L / v_th_i
        self.mirror_ratio = self.B_plug / self.B_midplane
        self.tau_GDT = (
            0.5 * self.mirror_ratio * (2.0 * self.Z_CV) / self.v_th_i
        )

        if sim_utils.get_rank() == 0:
            W = 52
            print()
            print("=" * W)
            print("  PLASMA PARAMETERS")
            print("=" * W)
            print(f"  Ion mass:              {self.ion_mass / constants.m_p:.1f} m_p")
            print(f"  Electron mass:         {self.electron_mass / constants.m_e:.1f} m_e")
            print(f"  Ion temperature:       {self.T_ION:.1f} eV")
            print(f"  Electron temperature:  {self.T_ELEC:.1f} eV")
            print(f"  Plasma freq:           {self.plasma_freq * 1e-9:.4f} GHz")
            print(f"  Plasma freq (SIPIC):   {self.plasma_freq_SIPIC * 1e-9:.4f} GHz")
            print(f"  B midplane:            {self.B_midplane * 1e3:.1f} mT")
            print(f"  B plug:                {self.B_plug:.3f} T")
            print(f"  Mirror ratio:          {self.mirror_ratio:.2f}")
            print(f"  rho_i midplane:        {self.rho_i_midplane * 1e2:.2f} cm")
            print(f"  rho_i plug:            {self.rho_i_plug * 1e2:.2f} cm")
            print(f"  a0 / rho_i:            {self.LR_SCALE / self.rho_i_midplane:.2f}")
            print(f"  Electron scatter time: {1.0 / self.nu_e * 1e9:.2f} ns")
            print(f"  Ion scatter time:      {1.0 / self.nu_i * 1e6:.2f} us")
            print(f"  GDT confinement time:  {self.tau_GDT * 1e6:.2f} us")
            print("=" * W)
            print("  NUMERICAL PARAMETERS")
            print("=" * W)
            print(f"  dx, dy, dz:            {self.dx * 1e3:.2f} mm,  {self.dy * 1e3:.2f} mm, {self.dz * 1e3:.2f} mm")
            print(f"  dt:                    {self.DT * 1e9:.2f} ns")
            print(f"  C_SI                   {self.C_SI}")
            print(f"  N_FLOOR                {self.N_FLOOR:.1e} m^-3")
            print(f"  w_p dt:                {self.plasma_freq * 2*np.pi * self.DT:.4f}")
            print(f"  w_p dt (SIPIC):        {self.plasma_freq_SIPIC * 2*np.pi * self.DT:.4f}")
            print(f"  w_ci dt (plug):        {self.omega_ci_plug * self.DT:.4f}")
            print(f"  Electron CFL:          {self.cfl:.4f}")
            print(f"  rho_i / dx:            {self.rho_i_midplane / self.dx:.2f}")
            print("=" * W)
            print("  SIMULATION PARAMETERS")
            print("=" * W)
            print(f"  Grid cells (NX x NY x NZ): {self.NX} x {self.NY} x {self.NZ}")
            print(f"  Total steps:           {self.max_steps:,}")
            print("=" * W)
            print("", flush=True)

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Simulation object                                                    #
        #######################################################################

        self.sim = picmi.Simulation(
            time_step_size=self.DT,
            max_steps=self.max_steps,
            verbose=self.test,
        )

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.NX, self.NY, self.NZ],
            lower_bound=[-self.LX / 2.0, -self.LY / 2.0, -self.LZ / 2.0],
            upper_bound=[self.LX / 2.0, self.LY / 2.0, self.LZ / 2.0],
            lower_boundary_conditions=['dirichlet']*3,
            upper_boundary_conditions=['dirichlet']*3,
            lower_boundary_conditions_particles=['absorbing']*3,
            upper_boundary_conditions_particles=['absorbing']*3,
            warpx_blocking_factor=4,
            warpx_max_grid_size_x=self.NX//2,
            warpx_max_grid_size_y=self.NY//2,
            warpx_max_grid_size=self.NZ,
        )
        self.sim.particle_shape = 3

        self.sim.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function=sim_utils.implicit_function_from_rz_curve(
                self.wall_rz_points
            )
        )

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        ions = picmi.Species(
            name="ions",
            charge=constants.e,
            mass=self.ion_mass
        )
        ions.initial_distribution=picmi.AnalyticDistribution(
            density_expression=
                f"{self.N0}*{self.get_W_exp('z', self.LZ_SCALE, 0.667)}*"
                f"{self.get_W_exp('sqrt(x*x+y*y)', self.LR_SCALE, 0.333)}",
            rms_velocity=[sim_utils.thermal_velocity(self.T_ION, ions.mass)]*3
        )
        self.sim.add_species(
            ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        electrons = picmi.Species(
            name="electrons",
            charge=-constants.e,
            mass=self.electron_mass
        )
        electrons.initial_distribution=picmi.AnalyticDistribution(
            density_expression=
                f"{self.N0}*{self.get_W_exp('z', self.LZ_SCALE, 0.667)}*"
                f"{self.get_W_exp('sqrt(x*x+y*y)', self.LR_SCALE, 0.333)}",
            rms_velocity=[sim_utils.thermal_velocity(self.T_ELEC, electrons.mass)]*3
        )
        self.sim.add_species(
            electrons,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        self.solver = picmi.ElectrostaticSolver(
            grid=self.grid,
            method='Multigrid',
            required_precision=1e-5,
            warpx_effective_potential=True,
            warpx_effective_potential_factor=self.C_SI,
            warpx_effective_potential_density_floor = self.N_FLOOR,
            warpx_effective_potential_time_filter_param = 0.1, # 1 - filter
            warpx_self_fields_verbosity=1,
        )
        self.sim.solver = self.solver

        # set biasing boundary condition
        self.V_bias = -750.0 # V
        self.r_bias = 0.4   # m
        if self.V_bias != 0.0:
            self.sim.embedded_boundary.potential = (
                f"if(abs(z)>{self.Z_MAX-self.dz},"
                    f"if(sqrt(x*x+y*y)<{self.r_bias},"
                    f"{self.V_bias}-{self.V_bias/self.r_bias}*sqrt(x*x+y*y),0)"
                ",0)"
            )

        # register the precomputed external field with WarpX. The loader
        # builds B as the Yee curl of the vector potential, so what lands on
        # the mesh is divergence free to round-off.
        self.bfield_loader.install_field_loader(self.sim)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_steps,
            data_list=[
                'B', 'E', 'phi',
                'rho', f'rho_{ions.name}', f'rho_{electrons.name}',
            ] + [f'T_{species.name}' for species in self.sim.species],
            write_dir='diags/',
            warpx_file_prefix='field_diag',
            warpx_format='openpmd',
            warpx_openpmd_backend='bp'
        )
        # self.sim.add_diagnostic(field_diag)

        particle_diag = picmi.ParticleDiagnostic(
            name="particle_diag",
            period=self.diag_steps*10,
            species=[species for species in self.sim.species],
            write_dir='diags/',
            warpx_file_prefix='particle_diag',
            warpx_format='openpmd',
            warpx_openpmd_backend='bp'
        )
        # self.sim.add_diagnostic(particle_diag)

        # --- reduced diagnostics (particle count/energy, field energy) -----
        self.sim.add_diagnostic(picmi.ReducedDiagnostic(
            diag_type='ParticleNumber', name='particle_number', period=100,
        ))
        self.sim.add_diagnostic(picmi.ReducedDiagnostic(
            diag_type='FieldEnergy', name='field_energy', period=100,
        ))
        self.sim.add_diagnostic(picmi.ReducedDiagnostic(
            diag_type='ParticleEnergy', name='particle_energy', period=100,
        ))

    def get_W_exp(self, var, x, alpha):
        return (
            f"if(abs({var}/{x})<{alpha},1,"
            f"if(abs({var}/{x})<1,0.5+0.5*cos({np.pi}*(abs({var}/{x})-{alpha})/({1-alpha})),0))"
        )


##########################
# parse input parameters
##########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--test",
    help="toggle whether this script is run as a short test",
    action="store_true",
)
parser.add_argument(
    "--bfield",
    default="bfield_rz.npz",
    help="path to the precomputed external B field (see generate_bfield.py)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = SimulationSetup(test=args.test, bfield_path=args.bfield)
run.sim.step()
