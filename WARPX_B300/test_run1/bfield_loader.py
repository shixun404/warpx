"""
Loads the precomputed axisymmetric mirror-coil field from bfield_rz.npz
(produced by generate_bfield.py) and installs it into a running
WarpX/PICMI simulation as the external B field. No realwx dependency.

WHY THIS GOES THROUGH THE VECTOR POTENTIAL
------------------------------------------
An earlier version of this module interpolated Br/Bz straight onto the Yee
mesh. That is wrong. The interpolant is only an *approximation* of B, and
nothing constrains the approximation error to be solenoidal, so WarpX's
cell-centred divergence stencil sees

    (div B)_cell ~ (interpolation error) / dx

i.e. a spurious magnetic-monopole density. It does not converge away with
the timestep, it breaks the mirror force / magnetic-moment conservation
along the axis, and on a mirror geometry it shows up as an artificial
radial drift of the confined population.

The fix is to discretise the *potential* instead of the field.
generate_bfield.py stores the poloidal flux function Psi = r * A_phi on the
coil's native (r, z) grid. Here we

  1. turn Psi into the Cartesian vector potential -- the coil field is
     purely poloidal, so A = A_phi phi_hat with A_phi = Psi / r, giving

         Ax = -y * (Psi / r^2),   Ay = +x * (Psi / r^2),   Az = 0

  2. sample A at the Yee *edge* centres, i.e. exactly where Ex, Ey, Ez
     live (E = -dA/dt - grad(phi) puts A on the E-field staggering), and

  3. set B = curl A with the standard Yee curl, which lands each component
     exactly on its own face centre: Bx at (i, j+1/2, k+1/2), By at
     (i+1/2, j, k+1/2), Bz at (i+1/2, j+1/2, k).

The Yee divergence annihilates the Yee curl of *any* edge-centred field --
the telescoping terms cancel term by term -- so the resulting B is
divergence free to round-off regardless of how good the interpolation of
Psi was, and even out in the corners of the box where Psi has to be
extrapolated. Interpolation quality now only sets how *accurate* B is,
never whether it is solenoidal.

Note that Psi/r^2 (rather than A_phi = Psi/r) is what gets interpolated:
Psi/r^2 is smooth and even in r, tending to Bz(0, z)/2 on axis, whereas
A_phi has a kink there. Multiplying by x / -y afterwards restores the
correct linear-in-r behaviour of A near the axis.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class BFieldLoader:
    def __init__(self, npz_path="bfield_rz.npz"):
        data = np.load(npz_path)
        self.r = np.asarray(data["r"], dtype=float)
        self.z = np.asarray(data["z"], dtype=float)

        if "psi" not in data:
            raise KeyError(
                f"{npz_path} contains no 'psi' array. Regenerate it with the "
                "current generate_bfield.py: the loader needs the poloidal "
                "flux function, not Br/Bz, to build a divergence-free B."
            )
        # Psi = r * A_phi  [T m^2], shape (nz, nr)
        self.psi = np.asarray(data["psi"], dtype=float)

        # g == A_phi / r == Psi / r^2.  Smooth and even in r (g -> Bz(0,z)/2
        # on axis), so this -- not A_phi itself -- is the safe thing to
        # interpolate near r = 0.  fill_value=None extrapolates linearly,
        # which is needed in the box corners where sqrt(x^2+y^2) runs past
        # the coil grid's r_max.
        self.g = self._psi_to_g(self.r, self.psi)
        self._g_interp = RegularGridInterpolator(
            (self.z, self.r), self.g, bounds_error=False, fill_value=None
        )

        # Diagnostic-only field interpolators + the scalar summaries that
        # calculate_plasma_parameters() prints. These are NOT used to fill
        # the WarpX arrays -- see the module docstring.
        if "Bz" in data and "Br" in data:
            Bz_grid = np.asarray(data["Bz"], dtype=float)
            Br_grid = np.asarray(data["Br"], dtype=float)
        else:
            Bz_grid, Br_grid = self._B_from_psi(self.r, self.z, self.psi)
        self.Bz_interp = RegularGridInterpolator(
            (self.z, self.r), Bz_grid, bounds_error=False, fill_value=None
        )
        self.Br_interp = RegularGridInterpolator(
            (self.z, self.r), Br_grid, bounds_error=False, fill_value=None
        )
        nz = Bz_grid.shape[0]
        self.B_midplane = float(Bz_grid[nz // 2, 0])
        self.B_plug = float(np.max(Bz_grid[:, 0]))

    # ------------------------------------------------------------------
    # Psi -> vector potential
    # ------------------------------------------------------------------

    @staticmethod
    def _psi_to_g(r, psi):
        """g(r, z) = Psi / r^2 = A_phi / r, with the r = 0 column filled in
        by its limit rather than by 0/0."""
        g = np.empty_like(psi)
        off_axis = r > 0.0
        g[:, off_axis] = psi[:, off_axis] / r[off_axis] ** 2

        if not off_axis.all():
            # g is even in r near the axis: g(r) = g0 + a r^2 + O(r^4).
            # Cancel the a r^2 term using the two innermost off-axis columns
            # so that g(0, z) comes out as Bz(0, z)/2 and not as zero.
            i1, i2 = np.flatnonzero(off_axis)[:2]
            r1, r2 = r[i1], r[i2]
            g0 = (r2**2 * g[:, i1] - r1**2 * g[:, i2]) / (r2**2 - r1**2)
            g[:, ~off_axis] = g0[:, None]

        return g

    @staticmethod
    def _B_from_psi(r, z, psi):
        """Bz = (1/r) dPsi/dr, Br = -(1/r) dPsi/dz. Only used to rebuild the
        diagnostic interpolators if the npz predates them."""
        dpsi_dz, dpsi_dr = np.gradient(psi, z, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            Bz = dpsi_dr / r[None, :]
            Br = -dpsi_dz / r[None, :]
        axis = r == 0.0
        if axis.any():
            # Bz(0, z) = 2 * lim_{r->0} Psi/r^2, and Br vanishes on axis.
            g = BFieldLoader._psi_to_g(r, psi)
            Bz[:, axis] = 2.0 * g[:, axis]
            Br[:, axis] = 0.0
        return Bz, Br

    def _A_on_mesh(self, mf, component, max_points=2_000_000):
        """Sample a Cartesian component of A at the staggered locations of
        the multifab `mf` (an Ex or Ey multifab), ghost cells included.

        Ax = -y * g(r, z),  Ay = +x * g(r, z),  with g = Psi/r^2.

        Evaluated in slabs of at most `max_points` points: the full mesh runs
        to several million points at production resolution, and the
        interpolator's temporaries scale with the number of points per call.
        """
        xs = mf.mesh("x", include_ghosts=True)
        ys = mf.mesh("y", include_ghosts=True)
        zs = mf.mesh("z", include_ghosts=True)

        # r is independent of z, so it only needs to be built once
        R_xy = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2)
        xy_factor = (-ys[None, :] if component == "x" else xs[:, None])[:, :, None]

        A = np.empty((xs.size, ys.size, zs.size))
        nz_slab = max(1, int(max_points // R_xy.size))
        for k in range(0, zs.size, nz_slab):
            z_slab = zs[k:k + nz_slab]
            R = np.broadcast_to(R_xy[:, :, None], R_xy.shape + (z_slab.size,))
            Z = np.broadcast_to(z_slab[None, None, :], R.shape)
            A[:, :, k:k + nz_slab] = xy_factor * self._g_interp((Z, R))
        return A

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def eval_xyz(self, x, y, z):
        """Evaluate (Bx, By, Bz) at Cartesian points by interpolating the
        stored Br/Bz directly.

        FOR PLOTTING / PROBING ONLY. This is exactly the interpolation that
        is *not* divergence free, which is why install_field_loader() does
        not use it -- do not fill grid arrays with the result.
        """
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        pts = np.stack([np.ravel(z), np.ravel(r)], axis=-1)
        Bz = self.Bz_interp(pts).reshape(np.shape(z))
        Br = self.Br_interp(pts).reshape(np.shape(z))
        Bx = Br * np.cos(theta)
        By = Br * np.sin(theta)
        return Bx, By, Bz

    # ------------------------------------------------------------------
    # WarpX hook-up
    # ------------------------------------------------------------------

    def install_field_loader(self, sim):
        """Register the external B field with the PICMI simulation `sim`.

        Uses picmi.LoadInitialFieldFromPython, which is WarpX's supported
        entry point for "load a static external field from an array": it
        sets B_ext_grid_init_style = load_from_python and calls back during
        WarpX::LoadExternalFields, where the callback is expected to fill
        the Bfield_fp_external multifabs. WarpX then adds that field in at
        gather time, so it survives every step -- unlike writing to
        Bfield_aux from an 'afterinit' callback, which gets overwritten.

        The projection-based divergence cleaner is left off: B is built as a
        discrete curl here, so it is already divergence free to round-off
        and there is nothing for the cleaner to do.
        """
        from pywarpx import picmi

        sim.add_applied_field(
            picmi.LoadInitialFieldFromPython(
                load_from_python=lambda: self._load_B(sim),
                load_E=False,
                load_B=True,
                warpx_do_initial_div_cleaning=False,
            )
        )

    def _load_B(self, sim):
        """Fill Bfield_fp_external with the Yee curl of the interpolated A.

        The multifab wrappers present a global view of the arrays, so every
        rank builds the whole field and __setitem__ keeps the blocks it owns.
        That is a one-off init cost (~0.2 s and a few hundred MB at
        112 x 112 x 352) and it is what WarpX's own load-from-python examples
        do; it is the first thing to revisit if init memory becomes a problem
        at much larger grids.
        """
        fields = sim.fields
        Ex = fields.get("Efield_fp", dir="x", level=0)
        Ey = fields.get("Efield_fp", dir="y", level=0)
        Bx = fields.get("Bfield_fp_external", dir="x", level=0)
        By = fields.get("Bfield_fp_external", dir="y", level=0)
        Bz = fields.get("Bfield_fp_external", dir="z", level=0)

        geom = sim.extension.warpx.Geom(0).data()
        dx, dy, dz = (geom.CellSize(i) for i in range(3))

        # A on the E-field staggering: Ax where Ex lives (i+1/2, j, k),
        # Ay where Ey lives (i, j+1/2, k). Az is identically zero for a
        # purely poloidal coil field, so every dAz/d. term below drops out.
        Ax = self._A_on_mesh(Ex, "x")
        Ay = self._A_on_mesh(Ey, "y")

        # Yee curl. Each difference lands on exactly the right face centre --
        # the shape check below is what verifies that.
        Bx_new = -(Ay[:, :, 1:] - Ay[:, :, :-1]) / dz          # -dAy/dz
        By_new = (Ax[:, :, 1:] - Ax[:, :, :-1]) / dz           #  dAx/dz
        Bz_new = ((Ay[1:, :, :] - Ay[:-1, :, :]) / dx          #  dAy/dx
                  - (Ax[:, 1:, :] - Ax[:, :-1, :]) / dy)       # -dAx/dy

        for name, new, mf in (("Bx", Bx_new, Bx), ("By", By_new, By), ("Bz", Bz_new, Bz)):
            # shape_with_ghosts carries a trailing component axis; drop it
            got, want = new.shape, tuple(mf.shape_with_ghosts)[:3]
            if got != want:
                raise RuntimeError(
                    f"{name}: curl of A has shape {got} but Bfield_fp_external "
                    f"expects {want}. The Efield_fp and Bfield_fp_external "
                    "multifabs must carry the same number of ghost cells for "
                    "the Yee curl to line up."
                )

        Bx[()] = Bx_new
        By[()] = By_new
        Bz[()] = Bz_new

        self._report_divergence(Bx_new, By_new, Bz_new, dx, dy, dz)

    @staticmethod
    def _report_divergence(Bx, By, Bz, dx, dy, dz):
        """Print the discrete div(B) actually loaded onto the mesh.

        This should come out at round-off (~1e-16 relative). If it does not,
        the staggering assumed above does not match the build's Yee layout.
        """
        try:
            from sim_utils import get_rank
            if get_rank() != 0:
                return
        except Exception:
            pass

        div = ((Bx[1:, :, :] - Bx[:-1, :, :]) / dx
               + (By[:, 1:, :] - By[:, :-1, :]) / dy
               + (Bz[:, :, 1:] - Bz[:, :, :-1]) / dz)
        B_max = max(np.abs(Bx).max(), np.abs(By).max(), np.abs(Bz).max())
        scale = B_max / min(dx, dy, dz)     # units of div(B), for normalising
        print(
            f"  External B loaded: max|B| = {B_max:.4f} T, "
            f"max|div B| / (max|B|/dx_min) = {np.abs(div).max() / scale:.2e}"
        )
