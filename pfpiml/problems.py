"""
The examples, and the interface they implement.

The solver core is identical for every example in the paper. What changes from one to the next
is a short list: the geometry patch, where the pre-existing cracks are, how the displacement
boundary conditions are imposed, which part of the parametric square is inside the body, and
whatever extra reporting that example needs. Problem packages exactly those, and the three
classes below are all the differences there are between the paper's six results.

To add an example, subclass Problem, override the few hooks that differ, add it to REGISTRY at
the bottom, and write a ten-line entry script under examples/.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import utils
from .geometry import (SEN_NOTCH, build_geomap, crack_profile, dist_to_notch, parse_cracks,
                       phi_gate)
from .solver import Sampler, _fields, kinematics, phi_squash
from .utils import DTYPE

# --------------------------------------------------------------------------
# the interface

"""
The Problem interface -- everything that differs between examples.

The solver core (encoding, network, energy, sampler, load stepping, force extraction, plots) is
identical for every example in the paper. What changes from one example to the next is a short
list: the geometry patch, where the pre-existing cracks are, how the displacement boundary
conditions are imposed, which part of the parametric square is inside the body, and whatever
extra reporting that example needs. A Problem packages exactly those.

To add an example, subclass Problem, override the few hooks that differ, register it in
``pfpiml.problems``, and write a ten-line entry script under ``examples/``. Hooks left alone keep
the defaults below, which describe the plain unit square with hard boundary conditions.
"""

class Problem:
    """Base class. The defaults describe a unit-square domain on a B-spline patch with no
       extra loss terms and no example-specific reporting."""

    name = "base"
    #: default value of --geo for this problem
    default_geo = "identity"

    # ------------------------------------------------------------------ CLI / config
    @staticmethod
    def add_arguments(ap):
        """Add this example's own command-line flags to an argparse parser."""

    def defaults(self):
        """Shared flags whose defaults this example shifts (geometry, material, step size)."""
        return {}

    def configure(self, cfg, a):
        """Copy this example's parsed flags onto cfg. Called at the end of Cfg.__init__."""

    def config_keys(self):
        """Names of the problem-specific cfg fields to record in config.json."""
        return ()

    # ------------------------------------------------------------------ geometry
    def build_geomap(self, cfg, logger=None):
        """Return the GeoMap for this example (None = no map, the legacy --geo none path)."""
        return build_geomap(cfg.geo, logger)

    def phi0(self, x, y, cfg):
        """The crack seed: phi0 in [0,1] at the (parametric) query points, 1 on a pre-existing
           crack and decaying over the length scale l. Examples without a pre-existing crack
           (nucleation) return zeros."""
        raise NotImplementedError

    # ------------------------------------------------------------------ fields / BCs
    def phi_envelope(self, x, y, cfg):
        """Multiplier in [0,1] on the LEARNABLE phi channel, or None for none. An example that
           imposes phi = 0 on part of the boundary through its fields() ansatz MUST express the
           envelope here and route fields() through it, so that phi_forward/phi_value - and with
           them phi_prev in the irreversibility penalty, the sampler's damage stratum and every
           rendering - see the SAME field the energy sees. Before this hook the shear benchmark
           enveloped phi in fields() only: phi_prev then sat permanently above the largest phi
           the ansatz could reach in the edge strips, an irremovable e_ir source whose only
           relief direction was raw -> 1, i.e. it MANUFACTURED the edge damage bands
           ()."""
        return None

    def fields(self, net, x, y, cfg, u_delta):
        """Map the raw network outputs to (u, v, phi, phi0) with the displacement boundary
           conditions imposed exactly (hard constraints), so no BC loss term is needed."""
        raise NotImplementedError

    # ------------------------------------------------------------------ domain
    def in_domain(self, x, y, cfg):
        """Indicator of the physical body on the parametric square: 1.0 inside, 0.0 outside.
           Only examples with a hole or cut-out need to override this."""
        c = cfg
        return ((x >= c.x0) & (x <= c.x1) & (y >= c.y0) & (y <= c.y1))

    def sample_uniform(self, sampler, n):
        """Draw n points for the uniform (bulk) stratum. Override when part of the parametric
           square is outside the body, so that the rejected points are redrawn."""
        return None      # None = use the sampler's own default (Sobol / |det J|-weighted grid)

    def prepare_sampler(self, sampler):
        """Called once per Sampler, after the |det J| cell weights are built and before any
           stratum uses them. This is where a hole or cut-out removes cells from every stratum
           at once (and rescales sampler.dom_area, the body's share of the parametric square)."""

    def crack_stratum(self, crack, gx, gy, cfg):
        """Adjust the grid that decides where the crack stratum puts its points. It normally
           follows max(phi0, phi_prev); an example with no pre-existing crack has to say where a
           crack is expected instead, or the stratum would have nothing to track."""
        return crack

    # ------------------------------------------------------------------ extra physics
    def reaction_force(self, net, cfg, prev_model, u_delta_nd, seed=999):
        """Override when the load does NOT ride the displacement ansatz, so that differentiating
           the elastic energy with respect to delta would miss it. Return None (the default) to
           use the energy-consistent reaction in pfpiml.force."""
        return None

    def extra_loss(self, net, cfg, u_delta):
        """Optional additional term added to the per-iteration objective, e.g. a penalty for
           boundary conditions that cannot be imposed exactly on a curved patch. Return None
           when the example has none (the default)."""
        return None

    # ------------------------------------------------------------------ reporting
    def render_mask(self, cfg, X, Y):
        """Boolean array marking render pixels OUTSIDE the body, which are drawn blank. Return
           None (the default) when the body fills the domain."""
        return None

    def overlay(self, cfg, ax=None):
        """Draw the body's own outline (a hole edge, a free surface) onto a field figure."""

    def plot_domain(self, cfg, save_dir):
        """One geometry figure per run, written next to the shared patch/parametric views."""

    def setup(self, net, cfg, save_dir, logger):
        """Called once after the network is built and before load stepping. Return True to end
           the run there -- for an example that is a single elastic verification rather than a
           fracture simulation."""
        return False

    def plot(self, net, cfg, save_dir, tag, ud):
        """Example-specific figures, called wherever the shared plots are written."""

    def finalize(self, net, cfg, save_dir, logger):
        """Called once after the last load step (verification runs, summary figures)."""

# --------------------------------------------------------------------------
# plate with a hole: Kirsch verification

def outside_hole(x, y, cfg):
    """1.0 outside the interior circular hole, 0.0 inside. The hole is MASKED OUT of the domain
       (no quadrature inside); its edge is then a natural traction-free boundary. hole_r<=0 => solid."""
    r = getattr(cfg, "hole_r", 0.0)
    if r <= 0:
        return torch.ones_like(x)
    return ((x - cfg.hole_cx) ** 2 + (y - cfg.hole_cy) ** 2 >= r * r).to(x.dtype)

def kirsch_stt(r, theta, sigma, a):
    """infinite-plate Kirsch HOOP stress sigma_tt(r,theta), far-field sigma along +x, hole radius a:
       sigma_tt = (s/2)(1+a^2/r^2) - (s/2)(1+3a^4/r^4) cos2theta.  At r=a: sigma(1-2cos2theta)
       => +3 sigma at theta=+/-90deg (nucleation), -sigma at 0/180deg. sigma_tt DECAYS with r, so a
       DEM ring at r=a+eps must be compared to Kirsch AT THAT r (not to the r=a limit of 3)."""
    ar2 = (a / r) ** 2
    return 0.5 * sigma * (1.0 + ar2) - 0.5 * sigma * (1.0 + 3.0 * ar2 ** 2) * np.cos(2.0 * theta)

def _draw_hole(cfg, ax=None):
    """overlay the circular hole outline (mask edge) on a field plot; no-op if solid."""
    if getattr(cfg, "hole_r", 0.0) <= 0:
        return
    t = np.linspace(0, 2 * np.pi, 200)
    (ax or plt).plot(cfg.hole_cx + cfg.hole_r * np.cos(t), cfg.hole_cy + cfg.hole_r * np.sin(t),
                     "-", c="k", lw=1.0)

def plot_nucleation_geometry(cfg, save_dir):
    """Physical geometry for the plate-with-hole: unit square + circular hole (masked interior
       free surface) + sampled quadrature points (which AVOID the hole) + the tension_x BC and the
       theta=+/-90deg Kirsch nucleation points. This is the mask analogue of the IGA geometry figs."""
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    smp = Sampler(cfg, prev_model=None, N=min(cfg.N, 6000), seed=cfg.seed)
    xy = smp.sample().detach().cpu().numpy()
    ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.3, c="0.62", alpha=0.5, label="quadrature pts")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "-", c="0.2", lw=1.4)
    if cfg.hole_r > 0:
        th = np.linspace(0, 2 * np.pi, 220)
        hx = cfg.hole_cx + cfg.hole_r * np.cos(th); hy = cfg.hole_cy + cfg.hole_r * np.sin(th)
        ax.fill(hx, hy, color="white", zorder=3)
        ax.plot(hx, hy, "-", c="k", lw=1.6, zorder=4, label=f"hole a={cfg.hole_r} (traction-free)")
        ax.plot([cfg.hole_cx, cfg.hole_cx], [cfg.hole_cy + cfg.hole_r, cfg.hole_cy - cfg.hole_r],
                "v", c="tab:red", ms=9, zorder=5, label=r"$\theta=\pm90^\circ$ (SCF=3, nucleation)")
    if cfg.loading == "tension_x":                          # left clamped, right pulled (horizontal)
        ax.plot([0, 0], [0, 1], "-", c="tab:green", lw=3, zorder=2)
        ax.plot([1, 1], [0, 1], "-", c="tab:blue", lw=3, zorder=2)
        for yb in np.linspace(0.12, 0.88, 5):
            ax.annotate("", xy=(1.14, yb), xytext=(1.0, yb),
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.4))
        ax.text(0.5, 1.05, r"left clamped (u=v=0)   |   right $u=\delta$ (pull)   |   top/bottom free",
                ha="center", fontsize=8.5)
    ax.set_xlim(-0.16, 1.28); ax.set_ylim(-0.1, 1.14); ax.set_aspect("equal", "box")
    ax.set_title(f"Nucleation plate: hole + mask + BC (loading={cfg.loading}, l={cfg.l})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(os.path.join(save_dir, "geometry_nucleation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def _elastic_density(exx, eyy, exy, cfg):
    """full isotropic elastic energy density (plane strain), phi=0 (no degradation)."""
    tr = exx + eyy
    return 0.5 * cfg.lam * tr ** 2 + cfg.mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)

def _stress(exx, eyy, exy, cfg):
    """plane-strain Cauchy stress from strain (phi=0): sigma = lam tr(eps) I + 2 mu eps."""
    tr = exx + eyy
    return (cfg.lam * tr + 2.0 * cfg.mu * exx,
            cfg.lam * tr + 2.0 * cfg.mu * eyy,
            2.0 * cfg.mu * exy)

def _far_field_sxx(net, cfg, ud):
    """remote uniaxial stress sigma_inf = mean sigma_xx over bulk points far from the hole."""
    rng = np.random.default_rng(7)
    xs = rng.uniform(0.02, 0.98, 4000); ys = rng.uniform(0.02, 0.98, 4000)
    keep = (xs - cfg.hole_cx) ** 2 + (ys - cfg.hole_cy) ** 2 > 0.35 ** 2
    xx = torch.tensor(xs[keep], dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1)
    yy = torch.tensor(ys[keep], dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1)
    exx, eyy, exy, *_ = kinematics(net, xx, yy, cfg, ud)
    sxx, _, _ = _stress(exx.detach(), eyy.detach(), exy.detach(), cfg)
    return float(sxx.mean().cpu())

def _plot_kirsch(theta_deg, stt_dem, stt_ana, r_over_a, s90d, s90a, s0d, s0a, save_dir, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    # (A) angular profile at the hole edge
    ax1.plot(theta_deg, stt_ana, "k-", lw=2, label=r"Kirsch (r=a+0.1$l$)")
    ax1.plot(theta_deg, stt_dem, "C1.", ms=3.5, label="DEM")
    ax1.axhline(3.0, color="0.7", ls=":", lw=1); ax1.axhline(-1.0, color="0.7", ls=":", lw=1)
    for t in (90, 270):
        ax1.axvline(t, color="0.9", lw=1)
    ax1.set_xlabel(r"$\theta$ (deg)"); ax1.set_ylabel(r"$\sigma_{\theta\theta}/\sigma_\infty$")
    ax1.set_title("angular profile at hole edge"); ax1.set_xlim(0, 360)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    # (B) radial decay: DEM -> SCF=3 at r=a
    ax2.plot(r_over_a, s90a, "k-", lw=2, label=r"Kirsch $\theta{=}90^\circ$")
    ax2.plot(r_over_a, s90d, "C1.", ms=3.5, label=r"DEM $\theta{=}90^\circ$")
    ax2.plot(r_over_a, s0a, "-", c="tab:blue", lw=2, label=r"Kirsch $\theta{=}0^\circ$")
    ax2.plot(r_over_a, s0d, ".", c="tab:cyan", ms=3.5, label=r"DEM $\theta{=}0^\circ$")
    ax2.axhline(3.0, color="0.7", ls=":", lw=1); ax2.axhline(-1.0, color="0.7", ls=":", lw=1)
    ax2.set_xlabel(r"$r/a$"); ax2.set_ylabel(r"$\sigma_{\theta\theta}/\sigma_\infty$")
    ax2.set_title(r"radial decay ($\to$ SCF=3 at $r=a$)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle(f"Kirsch verification  (hole a={cfg.hole_r}, l={cfg.l})")
    fig.tight_layout(); fig.savefig(os.path.join(save_dir, "kirsch_scf.png"), dpi=140)
    plt.close(fig)

def verify_kirsch(net, cfg, save_dir, logger):
    """M1: pure-elastic solve (phi frozen 0) on the plate-with-hole under horizontal far-field
       tension; extract the hole-edge hoop stress sigma_tt(theta) and compare to Kirsch
       sigma_tt = sigma_inf (1 - 2 cos 2theta)  =>  SCF = +3 at theta = +/-90deg, -1 at 0/180deg."""
    ud = torch.tensor(float(cfg.u_delta_nd), device=utils.DEVICE)       # single reference load state
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    smp = Sampler(cfg, prev_model=None, N=cfg.N, seed=cfg.seed)
    n_iter = cfg.iters0 if cfg.iters0 > 0 else cfg.iters
    t0 = time.time()
    for it in range(n_iter):
        net.enc.set_progress(0, it)
        pts = smp.sample()
        x, y = pts[:, 0:1].clone(), pts[:, 1:2].clone()
        exx, eyy, exy, _phi, _gp2, detJ, _lap = kinematics(net, x, y, cfg, ud)
        rho, inom = smp.rho_and_mask(pts)
        w = (inom * detJ / rho.clamp_min(1e-300)).detach()
        Pi = (_elastic_density(exx, eyy, exy, cfg) * w).mean()
        opt.zero_grad(); Pi.backward(); opt.step()
        if it % cfg.log_every == 0 or it == n_iter - 1:
            logger.info(f"  [kirsch] it {it:5d}  Pi_el {float(Pi.detach()):.5e}  {time.time()-t0:5.0f}s")
    net.enc.set_progress(1, 0)
    torch.save({"model": net.state_dict()}, os.path.join(save_dir, "ckpt.pt"))   # for re-extraction
    a, l = cfg.hole_r, cfg.l
    sig = _far_field_sxx(net, cfg, ud)                          # remote uniaxial stress

    def dem_stt(rr, tt):                                        # DEM hoop stress / sigma_inf on a ring
        xx = torch.tensor(cfg.hole_cx + rr * np.cos(tt), dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1)
        yy = torch.tensor(cfg.hole_cy + rr * np.sin(tt), dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1)
        exx, eyy, exy, *_ = kinematics(net, xx, yy, cfg, ud)
        sxx, syy, sxy = _stress(exx.detach(), eyy.detach(), exy.detach(), cfg)
        sxx = sxx.cpu().numpy().ravel(); syy = syy.cpu().numpy().ravel(); sxy = sxy.cpu().numpy().ravel()
        st, ct = np.sin(tt), np.cos(tt)
        return (sxx * st ** 2 + syy * ct ** 2 - 2.0 * sxy * st * ct) / sig

    def band(rr, tt, nr=7, dr=0.05):                           # radial-band average over [r-dr*l, r+dr*l]
        acc = 0.0                                              # (denoise: stress = autodiff grad => noisy)
        for rk in np.linspace(-dr * l, dr * l, nr):
            acc = acc + dem_stt(rr + rk, tt)
        return acc / nr

    # (A) ANGULAR profile at r ~ a + 0.1 l (radial-band averaged), vs Kirsch AT THAT radius
    th = np.linspace(0.0, 2.0 * np.pi, 361)
    r_ang = a + 0.1 * l
    stt_dem = band(np.full_like(th, r_ang), th)
    stt_ana = kirsch_stt(r_ang, th, 1.0, a)
    # (B) RADIAL decay at theta=90 (peak) and 0 (side), r in [a, 3a], band-averaged -> SCF=3 at r=a
    rr = np.linspace(a, 3.0 * a, 80)
    s90d, s90a = band(rr, np.full_like(rr, np.pi / 2)), kirsch_stt(rr, np.pi / 2, 1.0, a)
    s0d, s0a = band(rr, np.zeros_like(rr)), kirsch_stt(rr, 0.0, 1.0, a)
    scf_peak = float(stt_dem.max())                            # band-averaged near-edge peak (~SCF)
    logger.info(f"[kirsch] sigma_inf={sig:.3f}  angular peak DEM={scf_peak:.3f} vs Kirsch(r=a+0.1l)="
                f"{stt_ana.max():.3f}  |  radial at r=a: DEM(90)={float(s90d[0]):.3f} "
                f"DEM(0)={float(s0d[0]):.3f}  (Kirsch limits 3 / -1)")
    np.savetxt(os.path.join(save_dir, "kirsch_scf.txt"),
               np.column_stack([np.degrees(th), stt_dem, stt_ana]),
               header="theta(deg)  DEM_sigma_tt/sig(r=a+0.1l,band-avg)  Kirsch(r=a+0.1l)")
    _plot_kirsch(np.degrees(th), stt_dem, stt_ana, rr / a, s90d, s90a, s0d, s0a, save_dir, cfg)
    return scf_peak, sig

# --------------------------------------------------------------------------
# single-edge-notched plate

class SENProblem(Problem):
    name = "sen"
    default_geo = "identity"

    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--loading", type=str, default="shear", choices=["shear", "tension"],
                        help="shear = mode II (u-lift; SEN shear / branching) | tension = mode I "
                             "(v-lift; SEN tension / coalescence)")
        ap.add_argument("--cracks", type=str, default=SEN_NOTCH,
                        help="physical crack segments 'ax,ay,bx,by;...'. Default = single SEN notch "
                             "(0,0.5)-(0.5,0.5). Coalescence: 3 en-echelon cracks (see "
                             "pfpiml.geometry.DEFAULT_CRACKS).")

    def configure(self, cfg, a):
        assert cfg.loading in ("shear", "tension"), \
            f"the SEN plate is loaded in shear or tension, got {cfg.loading!r}"
        raw = getattr(a, "cracks", SEN_NOTCH)
        cfg.cracks = raw if isinstance(raw, list) else parse_cracks(raw)
        # legacy single-notch fields (plot/notch code references them) = first-crack bbox
        c0 = cfg.cracks[0]
        cfg.nx0, cfg.nx1, cfg.ny = min(c0[0], c0[2]), max(c0[0], c0[2]), 0.5 * (c0[1] + c0[3])

    def config_keys(self):
        return ("cracks",)

    def phi0(self, x, y, cfg):
        return crack_profile(dist_to_notch(x, y, cfg), cfg)

    def fields(self, net, x, y, cfg, u_delta):
        """net sees (x,y) + encoding; hard BC via env = y(1-y). SHEAR (mode II): delta rides u
           (top u=delta, v=0). TENSION (mode I): delta rides v (top v=delta, u=0)."""
        out = net(x, y)
        env = y * (1.0 - y)
        if cfg.loading == "shear":
            u = cfg.U_ref * (env * out[:, 0:1] + y * u_delta)
            v = cfg.U_ref * (env * out[:, 1:2])
        else:                                                   # tension (mode I): delta rides v
            u = cfg.U_ref * (env * out[:, 0:1])
            v = cfg.U_ref * (env * out[:, 1:2] + y * u_delta)
        phi0 = self.phi0(x, y, cfg)
        phi = phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
        return u, v, phi, phi0

SEN = SENProblem()

# --------------------------------------------------------------------------
# notched ring

class RingProblem(Problem):
    name = "ring"
    default_geo = "ring"

    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--ring_ri", type=float, default=5.0, help="ring inner radius (mm)")
        ap.add_argument("--ring_ro", type=float, default=20.0, help="ring outer radius (mm)")
        ap.add_argument("--ring_notch", type=float, default=3.0,
                        help="edge-notch radial depth (mm)")
        ap.add_argument("--k_bc", type=float, default=1e5,
                        help="outer-arc penalty (soft Dirichlet) stiffness; pull v=u_imp on the "
                             "upper arc, clamp u=v=0 on the lower arc")

    def defaults(self):
        # Si et al. 2023 sec 4.5 geometry and material, and the load range that cracks it
        return dict(geo="ring", l=0.2, E=210e3, nu=0.3, Gc=2.7, U_ref=0.033,
                    grip_l=0.0, gc_grip=0.0, nsteps=100, delta_max=0.036)

    def configure(self, cfg, a):
        cfg.ring_ri = getattr(a, "ring_ri", 5.0)
        cfg.ring_ro = getattr(a, "ring_ro", 20.0)
        cfg.ring_notch = getattr(a, "ring_notch", 3.0)
        cfg.k_bc = getattr(a, "k_bc", 1e5)
        # right edge notch: horizontal segment y=0, x in [Ro-notch, Ro], tip at r=Ro-notch
        cfg.nx0, cfg.nx1, cfg.ny = cfg.ring_ro - cfg.ring_notch, cfg.ring_ro, 0.0
        cfg.cracks = [(cfg.nx0, cfg.ny, cfg.nx1, cfg.ny)]

    def config_keys(self):
        return ("ring_ri", "ring_ro", "ring_notch", "k_bc", "nx0", "nx1", "ny", "cracks")

    def build_geomap(self, cfg, logger=None):
        return build_geomap(cfg.geo, logger, ring=(cfg.ring_ri, cfg.ring_ro))

    def phi0(self, x, y, cfg):
        return crack_profile(dist_to_notch(x, y, cfg), cfg)

    def fields(self, net, x, y, cfg, u_delta):
        """net sees the PARAMETRIC (xi,eta) + encoding. Hard u=0 on the x=0 SYMMETRY line (both
           radial edges map to eta=0 and eta=1) via env=eta(1-eta); v is FREE -- the outer-arc
           pull (v=u_imp, upper) and clamp (u=v=0, lower) come from the penalty in extra_loss,
           not from a lift, because the two halves split at eta=0.5 mid-edge."""
        out = net(x, y)
        env = y * (1.0 - y)                              # eta(1-eta): 0 on x=0 symmetry edges
        u = cfg.U_ref * (env * out[:, 0:1])              # u=0 on x=0; free interior
        v = cfg.U_ref * out[:, 1:2]                      # v free everywhere (penalty -> outer arc)
        phi0 = self.phi0(x, y, cfg)
        phi = phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
        return u, v, phi, phi0

    # ---------------------------------------------------------------- outer-arc boundary
    def bc_penalty(self, net, cfg, u_delta, n_bc=4000, seed=None):
        """Outer-arc SOFT DIRICHLET (penalty), a boundary integral over xi=1:
             upper arc (eta>0.5, physical y>0): k*(v - u_imp)^2   [imposed vertical pull]
             lower arc (eta<0.5, physical y<0): k*(u^2 + v^2)     [fully clamped]
           ds = |d phys/d eta| deta (arc length, from the geomap eta-tangent); u_imp =
           U_ref*u_delta. Monte-Carlo over eta~U(0,1) at xi=1 => k * E_eta[pen(eta)*(ds/deta)].
           The gradient flows through the net (u,v); the geometry weight is detached."""
        g = torch.Generator(device=utils.DEVICE)
        if seed is not None:
            g.manual_seed(int(seed))
        eta = torch.rand(n_bc, 1, generator=g, device=utils.DEVICE, dtype=DTYPE)
        xi = torch.ones_like(eta)
        u, v, _, _ = _fields(net, xi, eta, cfg, u_delta)
        _, b, _, d, _ = cfg.geomap.jac(xi, eta)              # b=dx/deta, d=dy/deta (detached)
        L = torch.sqrt(b * b + d * d + 1e-30)                # ds/deta (arc-length element)
        u_imp = cfg.U_ref * (u_delta if torch.is_tensor(u_delta) else float(u_delta))
        upper = (eta > 0.5).to(DTYPE)
        pen = upper * (v - u_imp) ** 2 + (1.0 - upper) * (u ** 2 + v ** 2)
        return cfg.k_bc * (pen * L).mean()

    def extra_loss(self, net, cfg, u_delta):
        return self.bc_penalty(net, cfg, u_delta)

    def reaction_force(self, net, cfg, prev_model, u_delta_nd, seed=999):
        """The load enters through the outer-arc penalty (the ansatz carries no delta), so the
           reaction is dPi_bc/du_delta,nd / U_ref = the vertical force on the pulled arc
           (= 2k int (u_imp - v) ds), not the elastic-energy derivative used elsewhere."""
        ud = torch.tensor(float(u_delta_nd), device=utils.DEVICE, requires_grad=True)
        Pi_bc = self.bc_penalty(net, cfg, ud, n_bc=20000, seed=seed)
        dPi = torch.autograd.grad(Pi_bc, ud)[0]
        return float(dPi) / cfg.U_ref

RING = RingProblem()

# --------------------------------------------------------------------------
# plate with a circular hole

class HolePlateProblem(Problem):
    name = "hole"
    default_geo = "identity"

    @staticmethod
    def add_arguments(ap):
        ap.add_argument("--mode", type=str, default="kirsch", choices=["kirsch", "nucleate"],
                        help="kirsch = elastic-only stress-concentration verification | "
                             "nucleate = the full fracture run (damage starts on its own)")
        ap.add_argument("--loading", type=str, default="tension_x",
                        choices=["shear", "tension", "tension_x"],
                        help="tension_x = horizontal tension (left clamped, right pulled) -- the "
                             "configuration used by the Kirsch check and the nucleation run")
        ap.add_argument("--cracks", type=str, default="none",
                        help="physical crack segments 'ax,ay,bx,by;...', or 'none' (the default) "
                             "for a plate with no pre-existing crack")
        ap.add_argument("--hole_r", type=float, default=0.1, help="hole radius (units of the plate)")
        ap.add_argument("--hole_cx", type=float, default=0.5, help="hole centre x")
        ap.add_argument("--hole_cy", type=float, default=0.5, help="hole centre y")

    def configure(self, cfg, a):
        cfg.hole_r = getattr(a, "hole_r", 0.1)
        cfg.hole_cx = getattr(a, "hole_cx", 0.5)
        cfg.hole_cy = getattr(a, "hole_cy", 0.5)
        cfg.mode = getattr(a, "mode", "nucleate")
        _cr = getattr(a, "cracks", "none")
        if isinstance(_cr, list):
            cfg.cracks = _cr
        elif str(_cr).strip().lower() in ("none", ""):
            cfg.cracks = []                       # NUCLEATION: no pre-crack (phi0 == 0 everywhere)
        else:
            cfg.cracks = parse_cracks(_cr)
        # legacy single-notch fields (plot/notch code references them) = first-crack bbox
        if cfg.cracks:
            c0 = cfg.cracks[0]
            cfg.nx0, cfg.nx1, cfg.ny = min(c0[0], c0[2]), max(c0[0], c0[2]), 0.5 * (c0[1] + c0[3])
        else:
            cfg.nx0 = cfg.nx1 = cfg.ny = 0.5      # no notch (nucleation)

    def config_keys(self):
        return ("mode", "hole_r", "hole_cx", "hole_cy", "cracks")

    # ---------------------------------------------------------------- fields
    def phi0(self, x, y, cfg):
        if not getattr(cfg, "cracks", None):      # NUCLEATION: no pre-crack => phi0 == 0 everywhere
            return torch.zeros_like(x)
        return crack_profile(dist_to_notch(x, y, cfg), cfg)

    def fields(self, net, x, y, cfg, u_delta):
        out = net(x, y)
        if cfg.loading == "tension_x":                          # HORIZONTAL tension (nucleation plate):
            env = x * (1.0 - x)                                 # left x=0 clamped, right x=1 u=delta, v=0
            u = cfg.U_ref * (env * out[:, 0:1] + x * u_delta)
            v = cfg.U_ref * (env * out[:, 1:2])
        elif cfg.loading == "shear":
            env = y * (1.0 - y)
            u = cfg.U_ref * (env * out[:, 0:1] + y * u_delta)
            v = cfg.U_ref * (env * out[:, 1:2])
        else:                                                   # tension (mode I, vertical): delta rides v
            env = y * (1.0 - y)
            u = cfg.U_ref * (env * out[:, 0:1])
            v = cfg.U_ref * (env * out[:, 1:2] + y * u_delta)
        phi0 = self.phi0(x, y, cfg)
        phi = phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
        return u, v, phi, phi0

    # ---------------------------------------------------------------- the hole
    def in_domain(self, x, y, cfg):
        c = cfg
        box = ((x >= c.x0) & (x <= c.x1) & (y >= c.y0) & (y <= c.y1)).to(DTYPE)
        return box * outside_hole(x, y, c)                      # exact-circle hole mask

    def prepare_sampler(self, sampler):
        """Fold the hole into the |det J| cell weights, so EVERY grid stratum excludes it, and
           record the remaining area so the bulk density stays normalised. The cell grid is the
           staircase approximation; the point sampling itself uses the exact circle."""
        c = sampler.cfg
        n = sampler.gn
        r = getattr(c, "hole_r", 0.0)
        if r <= 0:
            return
        xs = (np.arange(n) + 0.5) / n
        GX, GY = np.meshgrid(xs, xs, indexing="xy")
        mask = ((GX - c.hole_cx) ** 2 + (GY - c.hole_cy) ** 2 >= r * r).astype(float)
        sampler.HoleMask = mask
        sampler.Jcell = sampler.Jcell * mask
        sampler.dom_area = float(mask.mean())

    def sample_uniform(self, sampler, n):
        """Parametric-uniform Sobol points with the circular hole rejected (exact circle)."""
        c = sampler.cfg
        if getattr(c, "hole_r", 0.0) <= 0:
            return None
        xc, yc, r2 = c.hole_cx, c.hole_cy, c.hole_r ** 2
        out, need = [], n
        while need > 0:
            p = sampler.sobol.random(max(need * 2, 16))
            keep = p[(p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2 >= r2]
            out.append(keep); need -= len(keep)
        return torch.as_tensor(np.concatenate(out)[:n], dtype=DTYPE).to(utils.DEVICE)

    def crack_stratum(self, crack, gx, gy, cfg):
        """With no pre-existing crack the stratum would have nothing to follow, so concentrate it
           on the annulus just outside the hole edge -- where the stress concentrates and the
           crack will start."""
        if getattr(cfg, "hole_r", 0.0) <= 0:
            return crack
        R = torch.sqrt((gx - cfg.hole_cx) ** 2 + (gy - cfg.hole_cy) ** 2)
        band = torch.exp(-torch.clamp(R - cfg.hole_r, min=0.0) / (3.0 * cfg.l))
        return torch.maximum(crack, band)

    # ---------------------------------------------------------------- reporting
    def render_mask(self, cfg, X, Y):
        if getattr(cfg, "hole_r", 0.0) <= 0:
            return None
        return (X - cfg.hole_cx) ** 2 + (Y - cfg.hole_cy) ** 2 < cfg.hole_r ** 2

    def overlay(self, cfg, ax=None):
        _draw_hole(cfg, ax)

    def plot_domain(self, cfg, save_dir):
        plot_nucleation_geometry(cfg, save_dir)

    def setup(self, net, cfg, save_dir, logger):
        if getattr(cfg, "mode", "nucleate") == "kirsch":
            verify_kirsch(net, cfg, save_dir, logger)
            return True                                          # elastic verification only
        return False

HOLE = HolePlateProblem()

# --------------------------------------------------------------------------
# registry

#: every example the package ships, by the name that appears in --problem and config.json
REGISTRY = {
    "sen": SEN,        # single-edge-notched square: tension, shear, branching, coalescence
    "ring": RING,      # thick-walled ring with two edge notches (curved NURBS patch)
    "hole": HOLE,      # plate with a circular hole: Kirsch check + crack nucleation
}
