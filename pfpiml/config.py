"""
Run configuration: the Cfg object every other module reads, and the command line that fills it.

Cfg holds everything shared by the examples (material, phase-field model, encoding, sampler,
load stepping). Anything specific to one example -- crack seeds, hole radius, ring radii -- is
attached by that example's Problem in ``problem.configure(cfg, args)``.

Cfg accepts any object with the right attributes, not just an argparse.Namespace: missing fields
fall back to the documented defaults, so tools can rebuild a Cfg from a saved config.json.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import argparse
import json
import os

import numpy as np

from .geometry import DEFAULT_CRACKS, SEN_NOTCH      # noqa: F401  (named in --cracks help)
from .utils import get_problem

class _Namespace:
    """argparse.Namespace stand-in, for configs rebuilt from a run directory."""

def _infer_problem(js):
    """Which example a saved config came from. Runs made before the name was recorded are
       identified by what only that example writes: a hole radius, or the ring patch."""
    if js.get("problem"):
        return js["problem"]
    if js.get("hole_r") or js.get("mode") in ("kirsch", "nucleate") \
            or js.get("loading") == "tension_x":
        return "hole"                       # only the plate with a hole is pulled along x
    if js.get("geo") == "ring" or js.get("ring_ri"):
        return "ring"
    return "sen"

def cfg_from_json(source, **overrides):
    """Rebuild a Cfg from a finished run: pass its config.json (as a dict, a path to the file,
       or a path to the run directory holding it).

       This is how the monitors and figure tools reopen a run -- the saved config names the
       example it came from, so the right Problem is restored with it and the fields evaluate
       exactly as they did during training. Keyword overrides are applied last.

       The geometry map is NOT rebuilt here (it costs a patch validation): set cfg.geomap
       yourself with problem.build_geomap(cfg) if you need physical coordinates."""
    if isinstance(source, str):
        path = source
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            js = json.load(f)
    else:
        js = dict(source)

    a = _Namespace()
    for k, v in js.items():
        setattr(a, k, v)
    a.problem = _infer_problem(js)
    # Cfg parses these two from comma-separated strings, but they are saved as lists
    a.levels = ",".join(str(int(s)) for s in js["levels"])
    a.enc_start = ",".join(str(int(s)) for s in js["enc_start"])
    # names that differ between the config file and the command line
    a.reg = js.get("lambda_reg", 0.0)
    a.resample = js.get("resample_every", 1)
    # anything a run predating the field would not have written
    for k, dflt in (("plateau_tol", 2e-4), ("plateau_win", 400), ("grip_release", 0.0),
                    ("gamma_star", 0.0), ("geo", "identity"), ("rho_new", 0.0),
                    ("eta_new", 0.3), ("iters0", 0), ("log_every", 300), ("pf_order", 2),
                    ("fixed_notch", False), ("loading", "shear"), ("gc_grip", 0.0),
                    ("ckpt_every", 0), ("plot_every", 0), ("seed", 0)):
        if not hasattr(a, k):
            setattr(a, k, dflt)
    for k, v in overrides.items():
        setattr(a, k, v)
    return Cfg(a)

# --------------------------------------------------------------------------
# configuration

class Cfg:
    def __init__(self, a, problem=None):
        # which example this run is (the Problem supplies crack seeds, BCs, extra physics)
        if problem is None:
            problem = getattr(a, "problem", None) or "sen"
        self.problem = get_problem(problem) if isinstance(problem, str) else problem
        self.problem_name = self.problem.name
        # material (MPa, mm, N), plane strain
        self.E = getattr(a, "E", 340e3)
        self.nu = getattr(a, "nu", 0.22)
        self.g_floor = 1e-6
        self.Gc = getattr(a, "Gc", 0.04247)       # mJ/mm^2 (= 42.47 J/m^2)
        self.l = a.l
        # displacement scale: the prescribed value is U_ref times a dimensionless factor
        self.U_ref, self.u_delta_nd = getattr(a, "U_ref", 1e-3), 1.0
        # domain / notch
        self.x0 = self.y0 = 0.0
        self.x1 = self.y1 = 1.0
        self.loading = getattr(a, "loading", "shear")   # "shear" (mode II, u-lift) | "tension"
                                                        # (mode I, v-lift) | example-specific
        # legacy single-notch fields (plot / fixed-notch sampler code references them); the
        # Problem overwrites them with the first crack's bounding box or its own notch line.
        self.nx0, self.nx1, self.ny = 0.0, 0.5, 0.5
        # phi ansatz
        self.phi_bias = a.phi_bias                # sigmoid bias init (<0 => bulk phi~0)
        self.gamma_ir = a.gamma_ir                # irreversibility penalty weight
        self.ir_tol = a.ir_tol                    # irreversibility dead-band (anti fog-ratchet)
        self.phi_map = a.phi_map                  # "sigmoid" | "leaky" (Manav eq 23 -- no saturation)
        self.at = a.at                            # 2 = AT2 (w=phi^2, no threshold) | 1 = AT1
                                                  # (w=phi, c_w=8/3, TRUE elastic threshold
                                                  # psi_th = 3Gc/16l ~ 0.8 => bulk/fog/halo
                                                  # hard-zero below threshold)
        self.pf_order = getattr(a, "pf_order", 2)  # 2 = 2nd-order (AT1/AT2) | 4 = Borden 4th-order
                                                  # (AT2-type + (Lap phi)^2 term; identity geo only)
        self.grip_l = a.grip_l                    # corner-local grip BC radius (units of l); 0=off
        self.grip_release = a.grip_release        # crack-proximity gate release (0 = never)
        self.grip_released = (False, False, False, False)
        self.gc_grip = a.gc_grip                  # corner toughening beta (0 = off)
        self.gamma_star = a.gamma_star            # star-convex split parameter (0 = Amor)
        self.split = a.split                      # "amor" | "spectral" | "directional"
        self.dirfield = None                      # per-step frozen orientation field (directional)
        self.geo = a.geo                          # "none" | "identity" | "distort" | patch.json
        self.geomap = None                        # GeoMap instance (set in run(); None = pre-IGA path)
        # multiresolution encoding (replaces the warp)
        self.levels = tuple(int(s) for s in a.levels.split(","))
        self.enc_ch = a.enc_ch                    # feature channels per level
        self.lr_enc = a.lr_enc
        self.enc_reg = a.enc_reg                  # tiny L2 on grid features (drift guard)
        self.enc_start = tuple(int(s) for s in a.enc_start.split(","))
        self.enc_ramp = a.enc_ramp                # step-0 progressive-activation ramp (iters)
        # network
        self.width, self.depth = a.width, a.depth
        # sampling (3-component mixture; damage component upgraded per proposal 3)
        self.N = a.N
        self.w_unif, self.w_notch, self.w_damage = a.w_unif, a.w_notch, a.w_damage
        self.sigma_band_factor = 1.5
        self.eta_damage = a.eta_damage            # band indicator ~ phi(1-phi)+eta*phi0
        self.beta_drive = a.beta_drive            # + beta * normalized g(phi)*psi_plus (prev)
        self.grid_n = a.grid_n                    # density grid (256 => cell ~0.39l)
        # adaptive TOTAL count: base N stays; ADD points for the NEWLY
        # propagated band (phi_prev>0.5 where phi0<0.5), n_add = rho_new * new-band area
        self.rho_new = a.rho_new                  # areal density of added pts (pts per unit^2)
        self.eta_new = a.eta_new                  # new-band indicator ~ phi(1-phi)+eta_new*phi
        self.fixed_notch = getattr(a, "fixed_notch", False)   # middle stratum = OLD fixed symmetric band
        # load stepping / training
        self.nsteps, self.delta_max = a.nsteps, a.delta_max
        self.iters = a.iters                      # inner Adam iters per load step
        self.iters0 = a.iters0 if a.iters0 > 0 else a.iters  # step-0 (cold) gets more
        self.resample_every = a.resample
        self.lr = a.lr
        self.lambda_reg = a.reg
        self.plateau_tol, self.plateau_win = a.plateau_tol, a.plateau_win
        self.seed = a.seed
        # bookkeeping
        self.N_meter = 80000
        self.log_every = a.log_every
        self.ckpt_every = a.ckpt_every            # >0: ALSO save ckpt_stepNNN.pt every k steps
        self.plot_every = a.plot_every            # >0: plot every k steps (0 = legacy ~12 plots)
        self.ud_prev = 0.0                        # prev step's converged u_delta (drive eval)
        # example-specific fields (cracks, hole radius, ring radii, ...)
        self.problem.configure(self, a)

    @property
    def lam(self): return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def K(self): return self.lam + 2.0 * self.mu / 3.0
    @property
    def sigma_band(self): return self.sigma_band_factor * self.l

    def schedule(self):
        s = np.linspace(0.0, 1.0, self.nsteps + 1)[1:]      # skip delta=0 (trivial)
        ud = (self.delta_max / self.U_ref) * s
        return ud.tolist()

    def to_dict(self):
        d = {k: getattr(self, k) for k in
             ("E", "nu", "g_floor", "Gc", "l", "U_ref", "phi_bias", "gamma_ir", "ir_tol",
              "at", "pf_order", "phi_map", "grip_l", "grip_release", "gc_grip", "gamma_star", "split",
              "enc_ch", "lr_enc", "enc_reg", "enc_ramp", "width", "depth", "N",
              "w_unif", "w_notch", "w_damage", "eta_damage", "beta_drive", "grid_n",
              "rho_new", "eta_new",
              "nsteps", "delta_max", "iters", "iters0", "resample_every", "lr",
              "lambda_reg", "seed", "ckpt_every", "plot_every", "geo", "loading",
              "fixed_notch")}
        d["levels"] = list(self.levels); d["enc_start"] = list(self.enc_start)
        d["grip_released"] = list(self.grip_released)
        d["problem"] = self.problem_name
        for k in self.problem.config_keys():
            v = getattr(self, k, None)
            d[k] = [list(t) for t in v] if k == "cracks" and v is not None else v
        return d

# --------------------------------------------------------------------------
# command line

def build_parser(problem, description=None):
    """Shared flags + this example's own flags, with the example's default overrides applied."""
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- material ---------------------------------------------------------------
    ap.add_argument("--E", type=float, default=340e3, help="Young's modulus (MPa)")
    ap.add_argument("--nu", type=float, default=0.22, help="Poisson ratio")
    ap.add_argument("--Gc", type=float, default=0.04247,
                    help="critical energy release rate (mJ/mm^2)")
    ap.add_argument("--U_ref", type=float, default=1e-3,
                    help="displacement scale used to non-dimensionalise the load (mm)")
    ap.add_argument("--l", type=float, default=0.01, help="phase-field length scale (mm)")

    # ---- load stepping / training -----------------------------------------------
    ap.add_argument("--nsteps", type=int, default=18)
    ap.add_argument("--delta_max", type=float, default=2.7e-3)
    ap.add_argument("--iters", type=int, default=2000, help="inner Adam iters per load step")
    ap.add_argument("--iters0", type=int, default=4000, help="step-0 (cold) iters; 0=use --iters")
    ap.add_argument("--resample", type=int, default=1,
                    help="redraw points every k iters; 1 (fresh points EVERY iteration) is the "
                         "validated stability recipe -- the moving target kills point-cheating")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--reg", type=float, default=0.0)

    # ---- phase-field model -------------------------------------------------------
    ap.add_argument("--gamma_ir", type=float, default=1e3, help="irreversibility penalty weight")
    ap.add_argument("--ir_tol", type=float, default=0.0,
                    help="irreversibility dead-band: phi may relax up to tol below phi_prev "
                         "(kills the far-field noise-fog ratchet; ~0.02 recommended; 0 = exact)")
    ap.add_argument("--at", type=int, default=2, choices=[2, 1],
                    help="AT model: 2 = AT2 (no threshold) | 1 = AT1 (w=phi, c_w=8/3, true "
                         "elastic threshold -- kills fog/halo/corner nucleation below psi_th)")
    ap.add_argument("--pf_order", type=int, default=2, choices=[2, 4],
                    help="phase-field ORDER: 2 (DEFAULT = the 2nd-order examples) | 4 (Borden "
                         "CMAME 2014 4th-order, AT2-type: gradient coeff halved + a (Laplacian "
                         "phi)^2 term via one extra autodiff pass; the C1 crack functional a C0 "
                         "finite-element trial space cannot represent). pf_order 4 FORCES --at 2 "
                         "+ identity/none geometry, and uses the 4th-order optimal profile "
                         "phi0=exp(-2d/l)(1+2d/l).")
    ap.add_argument("--phi_map", type=str, default="sigmoid", choices=["sigmoid", "leaky"],
                    help="phi logit map: sigmoid (default) | leaky (Manav eq 23, "
                         "non-saturating -- REQUIRED for --at 1; pair with --phi_bias -2)")
    ap.add_argument("--phi_bias", type=float, default=-4.0, help="phi-channel sigmoid bias init")
    ap.add_argument("--split", type=str, default="hybrid",
                    choices=["amor", "spectral", "directional", "hybrid", "iso"],
                    help="energy split (DEFAULT hybrid = the production config): hybrid (Ambati: "
                         "isotropic-degraded mechanics + spectral-tension driving -- single band + "
                         "second hump, matches the FEM reference) | amor (vol-dev; crack slides "
                         "free, but shear DRIVES everywhere incl. corners) | spectral "
                         "(tension-only driving, but mode-II crack LOCKS -- measured) | directional "
                         "(crack-frame split on the band + spectral in bulk; per-step variational) "
                         "| iso (NO split -- full isotropic energy degrades => the crack BRANCHES)")

    # ---- clamped-corner devices ---------------------------------------------------
    ap.add_argument("--grip_l", type=float, default=3.0,
                    help="CORNER-LOCAL grip BC radius (units of l); 0 = off. Anti-nucleation "
                         "protection with a known conflict: always-on it strangles the FINAL "
                         "ligament -- see --grip_release")
    ap.add_argument("--grip_release", type=float, default=0.0,
                    help="OPT-IN crack-proximity gate release: when phi_prev exceeds this "
                         "threshold in the annulus [1.3,2.5]*grip_l*l outside a corner disc "
                         "(= the main crack has arrived), that corner's gate turns OFF "
                         "permanently, letting the final ligament sever. 0 = legacy "
                         "always-on gate. Suggested value 0.7")
    ap.add_argument("--gc_grip", type=float, default=1.0,
                    help="CORNER TOUGHENING beta: Gc -> Gc*(1+beta*bump) inside the ~grip_l*l "
                         "corner discs -- variational replacement for the phi-gate (which it "
                         "RETIRES when >0): the BC-singularity's weak driving cannot climb the "
                         "energy hill, the arriving crack's G cuts through => final severance "
                         "completes. 0 = off (pre-fix gate behavior returns via "
                         "--grip_l/--grip_release)")
    ap.add_argument("--gamma_star", type=float, default=0.0,
                    help="star-convex split parameter (0 = Amor); for compressive multiaxial "
                         "states -- measured ~0%% effect on the SEN corner")

    # ---- geometry -----------------------------------------------------------------
    ap.add_argument("--geo", type=str, default="identity",
                    help="geometry map (this solver is isogeometric by default): identity "
                         "(DEFAULT -- the unit square as a degree-2 B-spline patch, full "
                         "Cox-de Boor/J/|detJ|/pullback pipeline) | distort (interior-distorted "
                         "square, boundary + notch midline pointwise invariant -- the pullback "
                         "null test) | ring (half-annulus NURBS patch) | path to a patch JSON "
                         "{pu, pv, knots_u, knots_v, ctrl[mu][mv][2], weights?} | none (legacy "
                         "opt-out: bypasses the geometry map, ~10-15%% faster, numerically "
                         "identical on the square -- for timing/ablation only)")

    # ---- multiresolution encoding --------------------------------------------------
    ap.add_argument("--levels", type=str, default="32,128,384",
                    help="feature-grid sides, coarse->fine; the finest spacing is the "
                         "representation's resolution cap (384 => h~0.26l, matched to the "
                         "default band sampling; raise together with --N)")
    ap.add_argument("--enc_ch", type=int, default=2, help="feature channels per level")
    ap.add_argument("--lr_enc", type=float, default=2e-3, help="lr for the feature grids")
    ap.add_argument("--enc_reg", type=float, default=1e-8, help="L2 on grid features (drift guard)")
    ap.add_argument("--enc_start", type=str, default="0,0,0",
                    help="step-0 progressive activation start iters per level; the validated "
                         "recipe is NO hard gating (zero-init is the progressive mechanism)")
    ap.add_argument("--enc_ramp", type=int, default=1, help="gate ramp length (iters)")

    # ---- sampler -------------------------------------------------------------------
    ap.add_argument("--N", type=int, default=12000, help="integration points per iteration")
    ap.add_argument("--w_unif", type=float, default=0.4)
    ap.add_argument("--w_notch", type=float, default=0.3)
    ap.add_argument("--w_damage", type=float, default=0.3)
    ap.add_argument("--eta_damage", type=float, default=0.3)
    ap.add_argument("--beta_drive", type=float, default=0.5,
                    help="driving-force weight in the density indicator (0 = plain indicator)")
    ap.add_argument("--grid_n", type=int, default=256, help="density grid (256 => cell ~0.39l)")
    ap.add_argument("--rho_new", type=float, default=0.0,
                    help="OPTIONAL manual adaptive crack-band. 0 (DEFAULT) = off: the crack "
                         "stratum (max(phi0,phi_prev)*|detJ|) already covers the WHOLE crack "
                         "(notch AND propagated) at ONE physical density. >0 = MANUAL: ADD points "
                         "on the propagated crack at this physical areal density (pts per area) "
                         "-- nucleation / extra concentration.")
    ap.add_argument("--eta_new", type=float, default=0.3, help="new-band indicator phi weight")
    ap.add_argument("--fixed_notch", action="store_true",
                    help="middle (notch) stratum = fixed analytic SYMMETRIC band (Gaussian-in-y "
                         "about ny x uniform-in-x over the notch) instead of the adaptive crack "
                         "grid. phi-INDEPENDENT + symmetric => restores symmetric BRANCHING "
                         "(iso split). Default off (adaptive stratum).")

    # ---- network -------------------------------------------------------------------
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # ---- misc ----------------------------------------------------------------------
    ap.add_argument("--plateau_tol", type=float, default=2e-4)
    ap.add_argument("--plateau_win", type=int, default=400)
    ap.add_argument("--out", type=str, default="runs/Bms")
    ap.add_argument("--fem", type=str, default="NONE",
                    help="optional FEM F-delta file to overlay on the comparison plot; NONE = "
                         "DEM-only (default). References come from the finite element solver in fem/pf_fem.py.")
    ap.add_argument("--log_every", type=int, default=300)
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help=">0: also snapshot ckpt into step_NNN/ every k steps (1 = every step)")
    ap.add_argument("--plot_every", type=int, default=0,
                    help=">0: plot fields + solo FD into step_NNN/ every k steps (0 = legacy ~12)")
    ap.add_argument("--resume", action="store_true")

    problem.add_arguments(ap)
    ap.set_defaults(geo=problem.default_geo)      # the patch this example lives on
    defaults = problem.defaults()
    if defaults:
        ap.set_defaults(**defaults)               # ... which defaults() may still override
    return ap
