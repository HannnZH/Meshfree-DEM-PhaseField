"""The multi-crack benchmark of Hamdi and Lejeune as a pfpiml Problem.

Their model, read off the generator (`pfm_dataset-main/src/main.py`): the displacement solve
degrades the FULL isotropic stress by (1-phi)^2 and the energy decomposition enters only through
the driving history field. That is the hybrid formulation, so their `miehe`/`spect` subset is
exactly our `--split hybrid` -- no code change and no modelling ambiguity.

Geometry: the domain is 2 x 2 mm (confirmed from the data, not the paper text), so the map is an
affine degree-2 B-spline patch built exactly like the package's `identity` patch with the control
net scaled by the domain size. |det J| = L^2 handles the quadrature, and the pullback handles the
gradients, so nothing downstream needs to know about the scale.

Boundary conditions (also read off the generator, they are not fully stated in the paper):
    tension (bi-axial)  bottom v=0, top v=delta, left u=0, right u=delta;  no phi condition
    shear               bottom u=v=0, top v=0 and u=delta;  phi=0 on the top AND bottom edges

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import json
import os

import numpy as np
import torch

from pfpiml import utils
from pfpiml.geometry import _greville, _open_uniform_knots, crack_profile, dist_to_notch
from pfpiml.problems import REGISTRY, Problem
from pfpiml.solver import phi_squash

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(HERE, "patches")

def square_patch(L, path):
    """A degree-2 B-spline patch on [0,L]^2, the package's `identity` construction scaled by L.
       Affine, so det J = L^2 is constant and the map is exact."""
    p, m = 2, 7
    knots = _open_uniform_knots(m, p)
    g = _greville(knots, p)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    ctrl = np.stack([GX * L, GY * L], axis=-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"pu": p, "pv": p, "knots_u": list(knots), "knots_v": list(knots),
                   "ctrl": ctrl.tolist()}, f)
    return path

class MultiCrackProblem(Problem):
    """10-20 randomly placed and oriented internal cracks in a square, bi-axial tension or shear."""

    name = "multicrack"
    default_geo = os.path.join(PATCH_DIR, "square_2mm.json")

    # ------------------------------------------------------------------ CLI / config
    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--sample", type=str, required=True,
                        help="converted .npz for one dataset sample (see convert_data.py)")
        ap.add_argument("--case", type=str, default="tension", choices=["tension", "shear"],
                        help="bi-axial tension or shear, matching the dataset sub-directory")
        ap.add_argument("--phi_bc_w", type=float, default=0.01,
                        help="parametric width of the phi=0 boundary layer on the sheared edges; "
                             "0.01 of a 2 mm domain is 2 l at l=0.01")
        ap.add_argument("--seed_band", type=str, default="theirs", choices=["theirs", "at2"],
                        help="crack seeding: 'theirs' reproduces the profile their strain-history "
                             "seeding produces (a saturated core then the AT2 tail), so a Dice "
                             "score at t=0 is ~1 and every later difference is physics; 'at2' "
                             "uses our own optimal profile, which is narrower and costs ~0.23 "
                             "Dice at t=0 for nothing. NOTE 'theirs' carries 1.5x the ideal "
                             "surface energy per unit crack length (the saturated core adds "
                             "seed_plateau*Gc), exactly as their seeding does")
        ap.add_argument("--seed_plateau", type=float, default=0.50,
                        help="half-width of the seeding plateau in units of l. 0.50 is DERIVED "
                             "from their H_init: it saturates phi at phi_c=0.999 within l0/2 of "
                             "the segment and the homogeneous solution decays as exp(-d/l0) "
                             "outside, so their fully damaged core is exactly l wide -- ours is "
                             "too at 0.50. (Do NOT fit this to their phi>0.5 contour on the 128^2 "
                             "grid: the band is ~1.7 pixels there, so that measurement is "
                             "aliasing, and fitting it gives 0.62 and a 24%% too wide core.)")

    def defaults(self):
        # their material and our matching model; the load range is set per case in configure()
        return dict(E=1.0e6, nu=0.3, Gc=1.0, l=0.01, split="hybrid",
                    gc_grip=0.0, grip_l=0.0,           # their model has no corner device
                    levels="48,192,768",               # finest h = 0.26 l on a 2 mm domain
                    N=50000, grid_n=512, nsteps=100, iters=1500, iters0=3000)

    def configure(self, cfg, a):
        d = np.load(getattr(a, "sample"))
        cfg.sample = str(getattr(a, "sample"))
        cfg.case = getattr(a, "case", "tension")
        cfg.phi_bc_w = getattr(a, "phi_bc_w", 0.01)
        cfg.seed_band = getattr(a, "seed_band", "theirs")
        cfg.seed_plateau = getattr(a, "seed_plateau", 0.62)
        cfg.domain = float(d["domain"])
        cfg.cracks = [tuple(float(v) for v in c) for c in d["cracks"]]
        cfg.n_cracks = len(cfg.cracks)
        # the displacement the dataset actually reaches, so u_delta_nd runs over [0, 1]
        cfg.U_ref = float(d["t"][-1])
        cfg.delta_max_data = float(d["t"][-1])
        c0 = cfg.cracks[0]
        cfg.nx0, cfg.nx1, cfg.ny = min(c0[0], c0[2]), max(c0[0], c0[2]), 0.5 * (c0[1] + c0[3])
        square_patch(cfg.domain, self.default_geo)      # (re)write the patch for this domain
        cfg.geo = self.default_geo

    def config_keys(self):
        return ("sample", "case", "phi_bc_w", "seed_band", "seed_plateau", "domain",
                "n_cracks", "cracks")

    # ------------------------------------------------------------------ physics
    def phi0(self, x, y, cfg):
        d = dist_to_notch(x, y, cfg)                    # physical distance, via the patch
        if getattr(cfg, "seed_band", "theirs") == "theirs":
            # their H_init saturates phi at phi_c=0.999 within l0/2 of the segment and decays
            # outside it; a plateau plus our AT2 tail reproduces the band they actually have
            return torch.exp(-torch.clamp(d - cfg.seed_plateau * cfg.l, min=0.0) / cfg.l)
        return crack_profile(d, cfg)

    def phi_envelope(self, x, y, cfg):
        """phi = 0 on the sheared edges, imposed exactly by an envelope that vanishes there
           (they impose it as a Dirichlet condition to stop damage running along the edges).
           Declared through the base-class hook so fields() AND phi_forward/phi_value apply the
           SAME factor: before only fields() had it, and phi_prev (unenveloped) sat
           permanently above the reachable phi in the edge strips - an irremovable e_ir source
           that drove raw -> 1 there, manufacturing the very edge bands under investigation."""
        if cfg.case != "shear":
            return None
        w = max(cfg.phi_bc_w, 1e-6)
        return torch.tanh(y / w) * torch.tanh((1.0 - y) / w)

    def fields(self, net, x, y, cfg, u_delta):
        """(x, y) are PARAMETRIC in [0,1]^2; the patch maps them to [0,L]^2. Both cases impose
           their Dirichlet data exactly through a lift, as everywhere else in the package."""
        out = net(x, y)
        if cfg.case == "tension":                       # bi-axial: delta on the top AND the right
            u = cfg.U_ref * (x * (1.0 - x) * out[:, 0:1] + x * u_delta)
            v = cfg.U_ref * (y * (1.0 - y) * out[:, 1:2] + y * u_delta)
        else:                                           # shear: top slides, bottom clamped
            env = y * (1.0 - y)
            u = cfg.U_ref * (env * out[:, 0:1] + y * u_delta)
            v = cfg.U_ref * (env * out[:, 1:2])
        phi0 = self.phi0(x, y, cfg)
        raw = phi_squash(out[:, 2:3], cfg)
        penv = self.phi_envelope(x, y, cfg)
        if penv is not None:
            raw = penv * raw
        phi = phi0 + (1.0 - phi0) * raw
        return u, v, phi, phi0

MULTICRACK = MultiCrackProblem()
REGISTRY["multicrack"] = MULTICRACK
