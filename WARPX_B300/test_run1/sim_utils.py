"""
Standalone physics, geometry, species, and MPI helper functions.
"""

import numpy as np
from scipy import constants


# ---------------------------------------------------------------------------
# Parallel / MPI helpers  (replaces realwx.parallel_util.get_rank)
# ---------------------------------------------------------------------------

def get_rank():
    """Return this process's MPI rank (0 if MPI isn't available)."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        try:
            from pywarpx import amr
            return amr.ParallelDescriptor.MyProc()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Plasma physics helpers (replaces the corresponding realwx.util functions)
# ---------------------------------------------------------------------------

def thermal_velocity(T_eV, mass):
    """Most-probable thermal speed sqrt(2 k T / m); T given in eV."""
    T_joules = T_eV * constants.e
    return np.sqrt(2.0 * T_joules / mass)


def plasma_frequency(n, mass, Z=1):
    """Angular plasma frequency (rad/s) for density n [m^-3]."""
    q = Z * constants.e
    return np.sqrt(n * q**2 / (mass * constants.epsilon_0))


def cyclotron_frequency(B, m, Z=1):
    """Angular cyclotron frequency (rad/s)."""
    return Z * constants.e * B / m


def larmor_radius(T, B, m, Z=1):
    """Thermal Larmor (gyro) radius v_th/omega_c; T given in eV."""
    v_th = thermal_velocity(T, m)
    omega_c = cyclotron_frequency(B, m, Z)
    return v_th / omega_c


def Coulomb_log(n_e, T_e):
    """NRL-formulary-style Coulomb logarithm (T_e in eV, n_e in m^-3)."""
    n_e_cm3 = n_e * 1e-6
    return 23.0 - np.log(np.sqrt(n_e_cm3) * T_e ** -1.5)


def collision_frequency(n_e, T_e, n_i, T_i, m_i, Z=1):
    """Order-of-magnitude NRL-formulary collision frequencies [1/s].

    Returns (nu_e, nu_ei, nu_i). These are meant for the numerical-
    parameter console printout, not physics fidelity -- if the hackathon
    work depends on precise collision rates, re-derive/verify this against
    a realwx run of the same case before relying on it.
    """
    n_e_cm3 = n_e * 1e-6
    n_i_cm3 = n_i * 1e-6
    lnL = Coulomb_log(n_e, T_e)

    nu_e = 2.91e-6 * n_e_cm3 * lnL * T_e ** -1.5
    nu_i = 4.80e-8 * (Z ** 4) * n_i_cm3 * lnL * T_i ** -1.5 * np.sqrt(constants.m_p / m_i)
    nu_ei = nu_e  # same order-of-magnitude; refine if the hackathon needs it

    return nu_e, nu_ei, nu_i


# ---------------------------------------------------------------------------
# Geometry helper (replaces realwx.warpx_utils.implicit_function_from_rz_curve)
# ---------------------------------------------------------------------------

def implicit_function_from_rz_curve(rz_points, symmetric=True, close_ends=True):
    """Build a WarpX embedded-boundary implicit-function string for a body
    of revolution defined by a piecewise-linear (r, z) wall profile.

    `rz_points` is a list of (r, z) pairs, the same format used by
    `wall_rz_points` in the original script. The returned expression is > 0
    outside the vessel and < 0 inside it, matching what
    `picmi.EmbeddedBoundary(implicit_function=...)` expects.

    Parameters
    ----------
    rz_points:
        Wall profile points, in any order. With `symmetric`, give z >= 0
        only and the profile is mirrored about z = 0.
    symmetric:
        Mirror the profile about z = 0 by testing abs(z) rather than z.
    close_ends:
        Treat everything beyond the last z as solid, i.e. cap the vessel
        with flat end plates at |z| = z_last. Those plates are the only
        embedded-boundary surface an end electrode can attach to, so an
        `embedded_boundary.potential` biasing the ends silently does
        nothing without them.
    """
    pts = sorted(rz_points, key=lambda p: p[1])
    zv = "abs(z)" if symmetric else "z"

    def num(v):
        return f"{v:.6g}"

    def segment(i):
        """Linear r(z) interpolating pts[i] -> pts[i+1]."""
        (r0, z0), (r1, z1) = pts[i], pts[i + 1]
        # Coincident z is a vertical step in the profile. Nudge the spacing
        # so the slope stays finite; the branch is unreachable anyway, since
        # the enclosing test for the same z already caught everything below.
        dz = (z1 - z0) or 1e-9
        return f"({num(r0)}+{num((r1 - r0) / dz)}*({zv}-{num(z0)}))"

    # Nest inside out, so the *lowest* z test ends up outermost:
    #   if(z<z1, seg0, if(z<z2, seg1, ... r_last))
    # Wrapping the other way round makes the highest-z test outermost, where
    # it swallows every segment beneath it and flattens the whole profile to
    # a single cylinder.
    r_wall = num(pts[-1][0])
    for i in range(len(pts) - 2, -1, -1):
        r_wall = f"if({zv}<{num(pts[i + 1][1])},{segment(i)},{r_wall})"

    # clamp below the first point, for a profile that does not start at z = 0
    if pts[0][1] > 0:
        r_wall = f"if({zv}<{num(pts[0][1])},{num(pts[0][0])},{r_wall})"

    radial = f"sqrt(x*x+y*y)-({r_wall})"
    if not close_ends:
        return radial

    z_last = num(pts[-1][1])
    return f"if({zv}<{z_last},{radial},abs(z)-{z_last})"
