"""
pfpiml -- a mesh-free, automatic-differentiation Deep Energy Method for diffuse phase-field
fracture.

The solver represents the displacement and phase fields with one network on a multiresolution
B-spline feature encoding, integrates the energy by resampled Monte-Carlo quadrature over strata
that follow the crack, and minimizes that energy directly: no mesh, no background integration
grid, no crack tracking. Because every derivative comes from automatic differentiation, the same
code runs the 2nd-order (AT1/AT2) and the 4th-order (Borden) crack functionals -- the latter
needs a C1 field, which a C0 finite-element trial space cannot supply. The geometry is a single
B-spline/NURBS patch, so curved domains are exact.

Six modules, in dependency order:

    utils       device/dtype, the example registry, logging, checkpoints, legacy compatibility
    geometry    the NURBS patch, crack seeds, clamped-corner devices
    config      Cfg and the shared command line
    plots       run-time figures
    solver      encoding, network, energy, quadrature, force, load stepping
    problems    the Problem interface and the three examples (sen, ring, hole)

Runs are launched from the entry scripts in examples/; see README.md.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from . import utils
from .config import Cfg, build_parser, cfg_from_json
from .geometry import (DEFAULT_CRACKS, GRIP_CORNERS, SEN_NOTCH, GeoMap, build_geomap,
                       build_ring_patch, crack_profile, dist_to_notch, gc_bump, grip_corners,
                       parse_cracks, phi0_field, phi_gate, to_phys, update_grip_release)
from .plots import (plot_FD_compare, plot_FD_solo, plot_fields, plot_geometry, plot_geometry_bc,
                    plot_linecut, plot_parametric, plot_phi)
from .solver import (DirField, MultiResEncoding, NetMS, Sampler, _fields, _frac_energy,
                     drive_field, energy_terms, estimate_Pi, kinematics, phi_forward,
                     phi_squash, phi_value, reaction_force, run, run_cli, solve_step, validate)
from .utils import (DTYPE, available, get_logger, get_problem, load_ckpt, problem_of, save_ckpt,
                    set_device)
from .problems import Problem          # last: it imports geometry and solver

__version__ = "1.0.0"

def __getattr__(name):
    """DEVICE is re-read from utils on every access, so pfpiml.set_device('cpu') is visible
       through this name too (a plain import would have frozen a copy at import time)."""
    if name == "DEVICE":
        return utils.DEVICE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "utils", "set_device", "DEVICE", "DTYPE",
    "Cfg", "cfg_from_json", "build_parser", "Problem", "get_problem", "problem_of", "available",
    "GeoMap", "build_geomap", "build_ring_patch",
    "MultiResEncoding", "NetMS", "kinematics", "phi_squash", "phi_forward", "phi_value",
    "drive_field", "_fields",
    "DirField", "energy_terms", "_frac_energy",
    "Sampler", "estimate_Pi", "reaction_force",
    "to_phys", "parse_cracks", "dist_to_notch", "crack_profile", "phi0_field", "phi_gate",
    "grip_corners", "update_grip_release", "gc_bump", "SEN_NOTCH", "DEFAULT_CRACKS",
    "GRIP_CORNERS",
    "save_ckpt", "load_ckpt", "get_logger",
    "plot_phi", "plot_fields", "plot_linecut", "plot_FD_compare", "plot_FD_solo",
    "plot_geometry", "plot_geometry_bc", "plot_parametric",
    "solve_step", "run", "run_cli", "validate",
]
