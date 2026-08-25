"""
Geometry: the isogeometric patch, the crack seeds and the clamped-corner devices.

The physical domain is ONE B-spline / NURBS patch, so curved bodies are exact and the Jacobian
comes from automatic differentiation rather than a mesh. Fields, encoding and quadrature all
live on the parametric square; everything that has to know where a point really is -- the crack
seed, the corner discs, the plot axes -- goes through to_phys.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

from . import utils
from .utils import DTYPE, problem_of

# --------------------------------------------------------------------------
# B-spline / NURBS patch

def _bspline_basis(t, knots, p):
    """Cox-de Boor, vectorized + autodiff-composable (double-backward safe): t (N,),
       open/clamped knot vector (nk,), degree p -> (N, m) basis values, m = nk-p-1.
       The degree-0 boxcar is piecewise-constant (zero grad a.e. -- same status as the
       detached floor-index in the encoding's _interp_level); all t-dependence that
       carries derivatives is in the recursion's polynomial factors. Right endpoint
       folded into the last span (one-sided derivative there -- measure zero)."""
    nk = knots.numel()
    m = nk - p - 1
    t = t.reshape(-1, 1)
    # last-span-closed membership: t in [k_i, k_{i+1}) except the final nonempty span,
    # which is [k_{m-1}, k_m]. Implemented by nudging queries at/above the domain end
    # into the last span (detached -- index selection only).
    hi = knots[m]
    tq = torch.where(t >= hi, hi - 1e-12 * (1.0 + hi.abs()), t)
    B = ((tq >= knots[:-1]) & (tq < knots[1:])).to(t.dtype)          # (N, nk-1)
    for k in range(1, p + 1):
        d1 = knots[k:nk - 1] - knots[:nk - k - 1]                    # left span lengths
        d2 = knots[k + 1:nk] - knots[1:nk - k]                       # right span lengths
        w1 = torch.where(d1 > 0, (t - knots[:nk - k - 1]) / torch.where(d1 > 0, d1, torch.ones_like(d1)),
                         torch.zeros_like(t * d1))
        w2 = torch.where(d2 > 0, (knots[k + 1:nk] - t) / torch.where(d2 > 0, d2, torch.ones_like(d2)),
                         torch.zeros_like(t * d2))
        B = w1 * B[:, :nk - k - 1] + w2 * B[:, 1:nk - k]
    return B                                                          # (N, m)

class GeoMap(nn.Module):
    """Single-patch B-spline/NURBS geometry map (xi,eta) in [0,1]^2 -> physical (x,y).
       FIXED (buffers, not parameters) -- the geometry is data, not a trainable field.
       IGA convention fields/encoding/sampler live on the parametric
       domain; energy quadrature gains |det J|; physical gradients via J^{-T} (jac() --
       a separate detached leaf graph: J does not depend on net params, so it enters the
       energy as constants; the in-graph forward() is used where the MAP ITSELF must
       chain-rule, i.e. phi0/notch distance through to_phys)."""
    def __init__(self, pu, pv, knots_u, knots_v, ctrl, weights=None):
        super().__init__()
        self.is_identity = False                  # set by build_geomap; skips jac in kinematics
        self.pu, self.pv = int(pu), int(pv)
        ku = torch.as_tensor(knots_u, dtype=DTYPE)
        kv = torch.as_tensor(knots_v, dtype=DTYPE)
        C = torch.as_tensor(ctrl, dtype=DTYPE)                        # (mu, mv, 2)
        W = torch.ones(C.shape[0], C.shape[1], dtype=DTYPE) if weights is None \
            else torch.as_tensor(weights, dtype=DTYPE)
        assert ku.numel() - self.pu - 1 == C.shape[0], "knots_u/ctrl size mismatch"
        assert kv.numel() - self.pv - 1 == C.shape[1], "knots_v/ctrl size mismatch"
        self.register_buffer("ku", ku)
        self.register_buffer("kv", kv)
        self.register_buffer("cx", C[..., 0])
        self.register_buffer("cy", C[..., 1])
        self.register_buffer("w", W)

    def forward(self, xi, eta):
        """(N,1),(N,1) -> physical (N,1),(N,1); rational (NURBS) if weights != 1."""
        Bu = _bspline_basis(xi, self.ku, self.pu)                     # (N, mu)
        Bv = _bspline_basis(eta, self.kv, self.pv)                    # (N, mv)
        Wf = torch.einsum("ni,ij,nj->n", Bu, self.w, Bv)
        X = torch.einsum("ni,ij,nj->n", Bu, self.w * self.cx, Bv) / Wf
        Y = torch.einsum("ni,ij,nj->n", Bu, self.w * self.cy, Bv) / Wf
        return X.reshape(-1, 1), Y.reshape(-1, 1)

    def jac(self, xi, eta):
        """J entries + det at query points, DETACHED (constants in the energy graph).
           Separate leaf graph so it never entangles with the field autodiff."""
        xg = xi.detach().reshape(-1, 1).clone().requires_grad_(True)
        yg = eta.detach().reshape(-1, 1).clone().requires_grad_(True)
        xp, yp = self.forward(xg, yg)
        o = torch.ones_like(xp)
        a, b = torch.autograd.grad(xp, [xg, yg], o, retain_graph=True)   # dx/dxi, dx/deta
        c, d = torch.autograd.grad(yp, [xg, yg], o)                      # dy/dxi, dy/deta
        det = a * d - b * c
        return a.detach(), b.detach(), c.detach(), d.detach(), det.detach()

    @property
    def corners_phys(self):
        with torch.no_grad():
            z = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=DTYPE, device=self.ku.device)
            e = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=DTYPE, device=self.ku.device)
            xp, yp = self.forward(z, e)
        return tuple((float(xp[i]), float(yp[i])) for i in range(4))

def _open_uniform_knots(m, p):
    """clamped/open uniform knot vector for m control points, degree p."""
    ni = m - p                                                        # interior spans
    core = np.linspace(0.0, 1.0, ni + 1)
    return np.concatenate([np.zeros(p), core, np.ones(p)])

def _greville(knots, p):
    """Greville abscissae -- control values that reproduce linear functions exactly
       (=> exact identity patch)."""
    k = np.asarray(knots)
    m = len(k) - p - 1
    return np.array([k[i + 1:i + p + 1].mean() for i in range(m)])

def build_ring_patch(Ri, Ro):
    """Half-annulus (RIGHT half, x>=0) = the Si et al. 2023 sec 4.5 thick-walled ring under
       the y-axis symmetry (upper-pulled / lower-clamped is x->-x symmetric). Single NURBS
       patch: xi (u-dir) = RADIAL (degree 1, Ri->Ro), eta (v-dir) = ANGULAR (degree-2
       rational, TWO 90-deg arcs sweeping -90deg -> 0deg -> +90deg, i.e. the right
       semicircle from (0,-r) through (r,0) to (0,+r)). Consequences used by the BC/notch:
       eta=0.5 == angle 0 == the horizontal axis == the notch/crack line; eta>0.5 (physical
       y>0) = the PULLED arc, eta<0.5 (y<0) = the CLAMPED arc; eta=0 and eta=1 map onto the
       x=0 symmetry line (both radial edges). Returns (pu,pv,knots_u,knots_v,ctrl[mu][mv][2],
       weights[mu][mv])."""
    s = 2.0 ** -0.5
    unit = np.array([[0., -1.], [1., -1.], [1., 0.], [1., 1.], [0., 1.]])   # right-semicircle CPs
    wv = np.array([1.0, s, 1.0, s, 1.0])                                    # rational-quadratic circle
    knots_v = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1], dtype=float)          # two 90-deg Bezier arcs
    pv, pu = 2, 1
    knots_u = np.array([0, 0, 1, 1], dtype=float)                           # linear radial, 2 layers
    radii = np.array([float(Ri), float(Ro)])
    ctrl = np.empty((2, 5, 2)); W = np.empty((2, 5))
    for i in range(2):
        ctrl[i] = radii[i] * unit                                          # exact annulus: r_i * unit circle
        W[i] = wv
    return pu, pv, knots_u, knots_v, ctrl, W

def build_geomap(spec, logger=None, ring=(5.0, 20.0)):
    """--geo builders. 'none' -> None; 'identity' -> Greville identity patch (the i0
       regression gate: full spline machinery, map == id up to float roundoff);
       'distort' -> same unit square with boundary AND the y=0.5 notch midline held
       POINTWISE invariant (end rows/cols zero offset => clamped edges exact; offsets
       antisymmetric about the mid control row => odd map => exact midline), interior
       distorted => J != I with identical physics (the i1 pullback null test);
       'ring' -> half-annulus (Si et al. 2023 sec 4.5), radii from `ring`=(Ri,Ro);
       otherwise a JSON file {pu, pv, knots_u, knots_v, ctrl[mu][mv][2], weights?}."""
    if spec is None or spec == "none":
        return None
    if spec == "ring":
        pu, pv, ku, kv, ctrl, W = build_ring_patch(*ring)
        gm = GeoMap(pu, pv, ku, kv, ctrl, W)
    elif spec in ("identity", "distort"):
        p, m = 2, 7
        knots = _open_uniform_knots(m, p)
        g = _greville(knots, p)
        GX0, GY0 = np.meshgrid(g, g, indexing="ij")                   # ctrl[i,j] at (g_i, g_j)
        CX, CY = GX0.copy(), GY0.copy()
        if spec == "distort":
            # offsets ~ s(g_i)*q(g_j): s vanishes at the ends (=> clamped end rows/cols
            # keep the boundary EXACT), q = sin(2*pi*t) is ODD under g_j -> 1-g_j (the
            # Greville set is symmetric) => the control offsets are antisymmetric about
            # the mid row => the map is odd about eta=0.5 => the y=0.5 notch midline is
            # POINTWISE invariant (keeps the notch stratum + phi0 aligned). Interior
            # J != I everywhere else = the pullback null test.
            q = np.sin(2.0 * np.pi * GY0)
            CX = GX0 + 0.07 * np.sin(np.pi * GX0) * q
            CY = GY0 + 0.05 * np.sin(2.0 * np.pi * GX0) * q
        ctrl = np.stack([CX, CY], axis=-1)
        gm = GeoMap(p, p, knots, knots, ctrl)
    else:
        with open(spec) as f:
            d = json.load(f)
        gm = GeoMap(d["pu"], d["pv"], d["knots_u"], d["knots_v"], d["ctrl"],
                    d.get("weights"))
    gm = gm.to(utils.DEVICE)
    # identity map == physical coords to machine precision => flag it so kinematics skips the
    # (pointless, memory-heavy) Jacobian autodiff pass and runs at the exact pre-IGA cost.
    gm.is_identity = (spec == "identity")
    # validity + distortion report: dets on a Sobol cloud (area = integral of detJ)
    smp = torch.as_tensor(qmc.Sobol(d=2, scramble=True, seed=4).random(1 << 14),
                          dtype=DTYPE, device=utils.DEVICE)
    a, b, c, d_, det = gm.jac(smp[:, 0:1], smp[:, 1:2])
    stretch = torch.sqrt(torch.stack([a, b, c, d_]) ** 2).max()
    msg = (f"geomap '{spec}': det J in [{float(det.min()):.4f}, {float(det.max()):.4f}], "
           f"area~{float(det.mean()):.6f}, max |J| entry {float(stretch):.3f}, "
           f"corners {gm.corners_phys}")
    assert float(det.min()) > 0.0, f"geometry map not invertible: {msg}"
    if logger:
        logger.info(msg)
    return gm

# --------------------------------------------------------------------------
# crack seeds and corner devices

def to_phys(x, y, cfg):
    """parametric -> physical coordinates. THE single choke point: with --geo none the
       parametric coords ARE physical (pre-IGA path, bitwise unchanged); with a GeoMap
       the notch distance / corner discs / plot axes all go through here."""
    gm = getattr(cfg, "geomap", None)
    if gm is None or gm.is_identity:
        return x, y
    return gm(x, y)

SEN_NOTCH = "0,0.5,0.5,0.5"

DEFAULT_CRACKS = "0.25,0.35,0.35,0.45;0.45,0.45,0.55,0.55;0.65,0.55,0.75,0.65"

def parse_cracks(s):
    """"ax,ay,bx,by;..." -> [(ax,ay,bx,by), ...] physical crack segments."""
    out = []
    for seg in str(s).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        v = [float(t) for t in seg.split(",")]
        assert len(v) == 4, f"each crack needs 4 numbers ax,ay,bx,by; got {seg!r}"
        out.append(tuple(v))
    assert out, "no cracks parsed"
    return out

def dist_to_notch(x, y, cfg):
    """M0: MIN point-to-segment distance over ALL cfg.cracks, in PHYSICAL space. Standard t-clamp
       (any orientation); a single horizontal crack reduces to the old clamp-on-x SEN notch."""
    xp, yp = to_phys(x, y, cfg)
    d2 = None
    for (ax, ay, bx, by) in cfg.cracks:
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey + 1e-30
        t = torch.clamp(((xp - ax) * ex + (yp - ay) * ey) / L2, 0.0, 1.0)
        cx, cy = ax + t * ex, ay + t * ey
        dj = (xp - cx) ** 2 + (yp - cy) ** 2
        d2 = dj if d2 is None else torch.minimum(d2, dj)
    return torch.sqrt(d2 + 1e-30)

def crack_profile(d, cfg):
    """The model's OPTIMAL phi profile at distance d from a crack: the shape a fully developed
       crack takes, used to seed pre-existing cracks so that no load step has to grow them."""
    if getattr(cfg, "pf_order", 2) == 4:
        # Borden 4th-order OPTIMAL profile (Eq 13, phi=1-c, l0=l/2): exp(-2d/l)(1+2d/l).
        # C1-smooth at the crack (zero slope at d=0), unlike the 2nd-order exp(-d/l) kink.
        r = d / cfg.l
        return torch.exp(-2.0 * r) * (1.0 + 2.0 * r)
    if getattr(cfg, "at", 2) == 1:
        # AT1 optimal profile: compact support 2l (the AT2 exponential is not AT1-stationary)
        return torch.clamp(1.0 - d / (2.0 * cfg.l), min=0.0) ** 2
    return torch.exp(-d / cfg.l)

def phi0_field(x, y, cfg):
    """The crack seed of whichever example is running (see pfpiml.problem.Problem.phi0)."""
    return problem_of(cfg).phi0(x, y, cfg)

GRIP_CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))

def grip_corners(cfg):
    """PHYSICAL corner points of the clamped edges: patch images of the parametric
       corners when a GeoMap is active, the unit-square corners otherwise."""
    gm = getattr(cfg, "geomap", None)
    if gm is None or gm.is_identity:
        return GRIP_CORNERS
    return gm.corners_phys

def phi_gate(x, y, cfg):
    """CORNER-LOCAL grip BC: suppress the LEARNABLE phi within ~grip_l*l of the four
       clamped-edge CORNERS only (the stress-singular points that spuriously nucleate).
       The edges themselves stay damageable so the crack can LAND on the bottom
       edge like the reference (the reference crack exits the BOTTOM edge --
       the earlier full-width strip outlawed that and deflected the crack to the right
       edge). Smooth (C-inf) so |grad phi|^2 stays well-defined. grip_l<=0 disables.
       Corners with the grip_released flag set (crack-proximity release, --grip_release)
       are skipped: their gate is permanently OFF. When --gc_grip > 0 (corner TOUGHENING,
       the variational replacement) the phi-gate is fully retired."""
    if cfg.grip_l <= 0 or getattr(cfg, "gc_grip", 0.0) > 0:
        return 1.0
    w = cfg.grip_l * cfg.l
    rel = getattr(cfg, "grip_released", (False, False, False, False))
    xp, yp = to_phys(x, y, cfg)
    g = 1.0
    for k, (cx, cy) in enumerate(grip_corners(cfg)):
        if rel[k]:
            continue
        r = torch.sqrt((xp - cx) ** 2 + (yp - cy) ** 2 + 1e-30)
        g = g * torch.tanh(r / w)
    return g

@torch.no_grad()
def update_grip_release(cfg, prev_model, logger):
    """OPT-IN (--grip_release > 0) crack-proximity release of the corner gates: a corner's
       disc stays gated (anti-nucleation) until the MAIN crack band -- phi_prev above the
       threshold in the annulus [1.3, 2.5]*grip_l*l just OUTSIDE the disc -- arrives; then
       that corner's gate switches OFF permanently (physical final severance). Evaluated
       once per load step from the FROZEN prev state => each step remains one variational
       minimization. Released flags persist through ckpt/resume."""
    if cfg.grip_l <= 0 or cfg.grip_release <= 0 or cfg.gc_grip > 0 or prev_model is None:
        return
    from .solver import phi_value          # local: solver imports this module, so not at top
    rel = list(cfg.grip_released)
    w = cfg.grip_l * cfg.l
    for k, (cx, cy) in enumerate(GRIP_CORNERS):
        if rel[k]:
            continue
        n = 4000
        r = (1.3 + 1.2 * torch.rand(n, 1, device=utils.DEVICE)) * w
        th = 2.0 * math.pi * torch.rand(n, 1, device=utils.DEVICE)
        xs = cx + r * torch.cos(th)
        ys = cy + r * torch.sin(th)
        m = ((xs >= 0.0) & (xs <= 1.0) & (ys >= 0.0) & (ys <= 1.0)).reshape(-1)
        if int(m.sum()) == 0:
            continue
        ph = phi_value(prev_model, xs[m].reshape(-1, 1), ys[m].reshape(-1, 1), cfg)
        if float(ph.max()) > cfg.grip_release:
            rel[k] = True
            logger.info(f"  grip gate RELEASED at corner ({cx:.0f},{cy:.0f}): main crack "
                        f"arrived (annulus phi_max {float(ph.max()):.2f} > "
                        f"{cfg.grip_release})")
    cfg.grip_released = tuple(rel)

def gc_bump(x, y, cfg):
    """corner-toughening bump for --gc_grip: 1 at the clamped corners, -> 0 outside the
       ~grip_l*l discs (the complement of the old gate profile; grip_l doubles as the
       toughening radius). Smooth, so e_fr stays C-inf."""
    w = max(cfg.grip_l, 1e-6) * cfg.l
    xp, yp = to_phys(x, y, cfg)
    g = 1.0
    for cx, cy in grip_corners(cfg):
        r = torch.sqrt((xp - cx) ** 2 + (yp - cy) ** 2 + 1e-30)
        g = g * torch.tanh(r / w)
    return 1.0 - g
