"""
The solver: feature encoding, the network ansatz, the energy, the mesh-free quadrature and the
load-stepping loop.

The whole method is here, in the order it runs. A multiresolution B-spline encoding feeds one
network that outputs displacement and damage; hard boundary conditions are imposed on its raw
outputs; strains, grad phi and (for the 4th-order model) the Laplacian of phi come from
automatic differentiation; the energy is integrated by resampled Monte-Carlo quadrature over
strata that follow the crack; and each load step minimizes that energy directly with Adam,
warm-started from the previous step.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

from . import utils
from .config import Cfg, build_parser
from .geometry import (dist_to_notch, gc_bump, phi0_field, phi_gate, to_phys,
                       update_grip_release)
from .plots import (plot_FD_compare, plot_FD_solo, plot_fields, plot_geometry, plot_geometry_bc,
                    plot_linecut, plot_parametric, plot_phi)
from .utils import DTYPE, get_logger, load_ckpt, problem_of, save_ckpt

# --------------------------------------------------------------------------
# multiresolution feature encoding

def _qbspline_w(t):
    """uniform quadratic B-spline weights over 3 consecutive control points, t in [0,1].
       C1 across segments; sum = 1. Returns (w0, w1, w2), each shaped like t."""
    return 0.5 * (1.0 - t) ** 2, (-t * t + t + 0.5), 0.5 * t * t

class MultiResEncoding(nn.Module):
    """Dense multiresolution feature grids with quadratic-B-spline (C1) interpolation.
       Positional encoding only: NOT an FE mesh, NOT integration. All levels cover the whole
       domain (no Gamma / no tracking). Grids ZERO-INIT (residual features on top of raw (x,y));
       fine levels gated coarse-to-fine during step 0 (progressive activation). The finest grid
       spacing is the representation's hard bandwidth cap."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grids = nn.ParameterList(
            [nn.Parameter(torch.zeros(cfg.enc_ch, n, n, dtype=DTYPE)) for n in cfg.levels])
        self.gates = [1.0] * len(cfg.levels)      # plain floats, set per iteration in step 0

    @property
    def n_feat(self):
        return self.cfg.enc_ch * len(self.cfg.levels)

    def set_progress(self, step, it):
        """coarse-to-fine gates: only step 0 ramps; every later (warm) step runs fully on."""
        if step > 0:
            self.gates = [1.0] * len(self.cfg.levels)
            return
        c = self.cfg
        self.gates = [min(1.0, max(0.0, (it - s0) / max(1, c.enc_ramp))) if it < s0 + c.enc_ramp
                      else 1.0 for s0 in c.enc_start]

    def _interp_level(self, G, x, y):
        """G: (C, n, n) control values; x,y: (N,) in [0,1] (slight excursions extrapolate).
           Tensor-product quadratic B-spline: 3x3 gather, all ops autograd-composable
           (double backward safe -- needed for grad-phi energy + loss.backward)."""
        n = G.shape[-1]
        s = float(n - 2)
        ux, uy = x * s, y * s
        ix = ux.detach().floor().clamp(0, n - 3).long()
        iy = uy.detach().floor().clamp(0, n - 3).long()
        tx = ux - ix.to(DTYPE)
        ty = uy - iy.to(DTYPE)
        wx = _qbspline_w(tx)                       # 3 x (N,)
        wy = _qbspline_w(ty)
        out = 0.0
        for b in range(3):
            for a in range(3):
                out = out + (wy[b] * wx[a]).unsqueeze(0) * G[:, iy + b, ix + a]   # (C, N)
        return out

    def forward(self, x, y):
        """x, y: (N,1) -> features (N, n_feat)."""
        xf, yf = x.reshape(-1), y.reshape(-1)
        feats = []
        for k, G in enumerate(self.grids):
            f = self._interp_level(G, xf, yf)                 # (C, N)
            feats.append(self.gates[k] * f)
        return torch.cat(feats, dim=0).transpose(0, 1)        # (N, n_feat)

    def reg(self):
        return sum((G ** 2).sum() for G in self.grids)

    def fine_max(self):
        return float(self.grids[-1].detach().abs().max())

# --------------------------------------------------------------------------
# network ansatz and autodiff kinematics

class NetMS(nn.Module):
    """monolithic [u_hat, v_hat, a_hat] on the encoded input (x, y, F(x,y)). NO warp."""
    def __init__(self, cfg):
        super().__init__()
        self.enc = MultiResEncoding(cfg)
        L = [nn.Linear(2 + self.enc.n_feat, cfg.width), nn.GELU()]
        for _ in range(cfg.depth - 1):
            L += [nn.Linear(cfg.width, cfg.width), nn.GELU()]
        last = nn.Linear(cfg.width, 3)
        L += [last]
        self.mlp = nn.Sequential(*L)
        with torch.no_grad():                       # bias the phi channel negative -> bulk phi~0
            last.bias[2] = cfg.phi_bias

    def forward(self, x, y):
        feats = self.enc(x, y)
        return self.mlp(torch.cat([x, y, feats], dim=1))

def _fields(net, x, y, cfg, u_delta):
    """The physical fields (u, v, phi, phi0) at the query points, with the displacement boundary
       conditions imposed exactly. Which conditions those are is the running example's business,
       so this dispatches to its Problem (see pfpiml.problem.Problem.fields)."""
    return problem_of(cfg).fields(net, x, y, cfg, u_delta)

def kinematics(net, x, y, cfg, u_delta):
    """strains (autodiff), phi and |grad phi|^2 (autodiff), all wrt PHYSICAL coords.
       --geo none: inputs ARE physical (pre-IGA path, unchanged). With a GeoMap the
       inputs are PARAMETRIC; raw autodiff grads are pulled back via J^{-T} (J detached
       -- geometry does not depend on net params) and detJ is returned for quadrature."""
    x = x.requires_grad_(True); y = y.requires_grad_(True)
    u, v, phi, phi0 = _fields(net, x, y, cfg, u_delta)
    ou = torch.ones_like(u)
    ux = torch.autograd.grad(u, x, ou, create_graph=True)[0]
    uy = torch.autograd.grad(u, y, ou, create_graph=True)[0]
    vx = torch.autograd.grad(v, x, ou, create_graph=True)[0]
    vy = torch.autograd.grad(v, y, ou, create_graph=True)[0]
    op = torch.ones_like(phi)
    px = torch.autograd.grad(phi, x, op, create_graph=True)[0]
    py = torch.autograd.grad(phi, y, op, create_graph=True)[0]
    gm = getattr(cfg, "geomap", None)
    lap_phi = None
    if getattr(cfg, "pf_order", 2) == 4:
        # 4th-order: the Laplacian phi = phi_xx + phi_yy via ONE MORE autodiff pass (px,py carry
        # create_graph=True). IDENTITY geometry only (SEN square): J=I => px,py are already the
        # physical grad phi, so d(px)/dx + d(py)/dy is the physical Laplacian. Curved-geo would need
        # the geo-map's 2nd derivatives in the pullback (not the 4th-order examples) -- guarded in main().
        if gm is not None and not gm.is_identity:
            raise ValueError("pf_order 4 requires identity geometry (SEN square); "
                             "curved-geo Laplacian pullback is not implemented")
        pxx = torch.autograd.grad(px, x, torch.ones_like(px), create_graph=True)[0]
        pyy = torch.autograd.grad(py, y, torch.ones_like(py), create_graph=True)[0]
        lap_phi = pxx + pyy
    if gm is None or gm.is_identity:                        # identity: J=I, detJ=1 (pre-IGA cost)
        detJ = torch.ones_like(phi)
    else:
        a, b, c, d, detJ = gm.jac(x, y)                     # J = [[a,b],[c,d]], detached
        ux, uy = (d * ux - c * uy) / detJ, (-b * ux + a * uy) / detJ
        vx, vy = (d * vx - c * vy) / detJ, (-b * vx + a * vy) / detJ
        px, py = (d * px - c * py) / detJ, (-b * px + a * py) / detJ
    exx, eyy, exy = ux, vy, 0.5 * (uy + vx)
    gradphi2 = px ** 2 + py ** 2
    return exx, eyy, exy, phi, gradphi2, detJ, lap_phi

def phi_squash(a_hat, cfg):
    """map the raw phi logit to ~[0,1]. sigmoid = the default; "leaky" = Manav et al. CMAME 2024
       eq (23): linear core (slope 1/4, |a|<=2) + LEAKY tails (slope beta=1e-3) => the gradient
       never vanishes, so AT1's constant downward drive cannot put the phi channel into the
       sigmoid saturation coma (measured: a_hat -19/-27, sigma' ~ 1e-8/1e-11, runs/Bms_at1).
       Slight out-of-[0,1] excursions are INCENTIVE-bounded (g, w, irrev push back), not clamped.
       With leaky, init --phi_bias -2 so f(init)=0."""
    if cfg.phi_map == "leaky":
        b = 1e-3
        core = a_hat / 4.0 + 0.5
        lo = b * (a_hat + 2.0)
        hi = b * (a_hat - 2.0) + 1.0
        return torch.where(a_hat < -2.0, lo, torch.where(a_hat > 2.0, hi, core))
    return torch.sigmoid(a_hat)

def phi_forward(net, x, y, cfg):
    """phi at points, grad-ENABLED (orientation field / structure tensor). net=None -> phi0.
       MUST match the phi that Problem.fields() hands the energy: a problem that envelopes the
       learnable channel (phi=0 edge strips) declares it via Problem.phi_envelope, applied here
       IDENTICALLY (same factor order), so phi_prev/e_ir can never diverge from the energy's
       phi again. env=None keeps the original expression byte-for-byte (anchors)."""
    phi0 = phi0_field(x, y, cfg)
    if net is None:
        return phi0
    out = net(x, y)
    env = problem_of(cfg).phi_envelope(x, y, cfg)
    if env is None:
        return phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * phi_squash(out[:, 2:3], cfg)
    return phi0 + (1.0 - phi0) * phi_gate(x, y, cfg) * (env * phi_squash(out[:, 2:3], cfg))

@torch.no_grad()
def phi_value(net, x, y, cfg):
    """phi at points from a (frozen) net, value only. net=None -> phi0 (step 0 / notch)."""
    return phi_forward(net, x, y, cfg)

def drive_field(net, x, y, cfg, u_delta, chunk=16384):
    """g(phi)*psi_plus_VD of the FROZEN prev net (detached) = crack driving force, for the sampler
       indicator (covers the process zone AHEAD of the tip, which otherwise starves).

       DELIBERATELY the volumetric-deviatoric form below, whatever --split the ENERGY uses
       (hybrid/spectral drive their energies spectrally; this stays vd). It is only a sampling
       proposal density: rho_and_mask divides the estimator by the same density, so E[Pi_hat]
       is unchanged for any strictly positive proposal -- only variance/point placement move.
       vd is chosen since it is smooth (no eigen branch) and BROADER than the spectral psi_+
       in shear-dominated states (measured: spectral is ~53% of Amor's at the SENS corner), and
       for a proposal over-covering is safe while under-covering starves the process zone.
       Do NOT "align" it with the energy split: that narrows shear coverage, moves every MC
       realization, and fails all 12 anchors."""
    outs = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].clone().requires_grad_(True)
        yc = y[i:i + chunk].clone().requires_grad_(True)
        u, v, phi, _ = _fields(net, xc, yc, cfg, u_delta)
        ou = torch.ones_like(u)
        ux = torch.autograd.grad(u, xc, ou, retain_graph=True)[0]
        uy = torch.autograd.grad(u, yc, ou, retain_graph=True)[0]
        vx = torch.autograd.grad(v, xc, ou, retain_graph=True)[0]
        vy = torch.autograd.grad(v, yc, ou)[0]
        gm = getattr(cfg, "geomap", None)
        if gm is not None and not gm.is_identity:
            a, b, c, d, detJ = gm.jac(xc, yc)
            ux, uy = (d * ux - c * uy) / detJ, (-b * ux + a * uy) / detJ
            vx, vy = (d * vx - c * vy) / detJ, (-b * vx + a * vy) / detJ
        exx, eyy, exy = ux, vy, 0.5 * (uy + vx)
        tr = exx + eyy
        tr_p = torch.clamp(tr, min=0.0)
        dev_xx, dev_yy, dev_zz = exx - tr / 3.0, eyy - tr / 3.0, -tr / 3.0
        dev_con = dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * exy ** 2
        psi_p = 0.5 * cfg.K * tr_p ** 2 + cfg.mu * dev_con
        g = (1.0 - phi) ** 2 + cfg.g_floor
        outs.append((g * psi_p).detach())
    return torch.cat(outs, dim=0)

# --------------------------------------------------------------------------
# energy densities

class DirField:
    """per-load-step FROZEN crack-orientation + blend field from phi_prev -- pointwise, NO geometry.
       Structure tensor <grad phi grad phi^T> on a grid (Gaussian-blurred outer products), stored
       via the director-safe DOUBLE-ANGLE (cos2th, sin2th); blend W = coherence * phi-presence:
       W~1 on the established band (directional split: crack slides AND opens), W~0 in the bulk
       (spectral split: tension-only driving => corners starve). Everything frozen from prev_model
       => the per-step problem remains the minimization of ONE energy functional (variational,
       unlike the hybrid). Step 0: prev=None -> phi0 (the notch) supplies the orientation."""
    def __init__(self, prev_model, cfg, n=192, sig_cells=2.0, chunk=16384):
        xs = (torch.arange(n, dtype=DTYPE, device=utils.DEVICE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        gx, gy = GX.reshape(-1, 1), GY.reshape(-1, 1)
        pxs, pys, phs = [], [], []
        for i in range(0, gx.shape[0], chunk):
            xc = gx[i:i + chunk].clone().requires_grad_(True)
            yc = gy[i:i + chunk].clone().requires_grad_(True)
            ph = phi_forward(prev_model, xc, yc, cfg)
            px, py = torch.autograd.grad(ph.sum(), [xc, yc])
            pxs.append(px.detach()); pys.append(py.detach()); phs.append(ph.detach())
        px = torch.cat(pxs).reshape(n, n); py = torch.cat(pys).reshape(n, n)
        phig = torch.cat(phs).reshape(n, n)
        J11, J12, J22 = px * px, px * py, py * py
        # Gaussian blur (separable) = the neighborhood average that makes the director
        # well-defined on the band centerline (grad phi = 0 there, flanks average in)
        k = int(4 * sig_cells) | 1
        t = torch.arange(k, dtype=DTYPE, device=utils.DEVICE) - (k - 1) / 2.0
        ker = torch.exp(-0.5 * (t / sig_cells) ** 2); ker = ker / ker.sum()
        def blur(M):
            M = M.reshape(1, 1, n, n)
            M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, -1, 1), padding=(k // 2, 0))
            M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, 1, -1), padding=(0, k // 2))
            return M.reshape(n, n)
        J11, J12, J22 = blur(J11), blur(J12), blur(J22)
        theta = 0.5 * torch.atan2(2.0 * J12, J11 - J22)
        disc = torch.sqrt(((J11 - J22) * 0.5) ** 2 + J12 ** 2 + 1e-300)
        lam1 = (J11 + J22) * 0.5 + disc
        lam2 = (J11 + J22) * 0.5 - disc
        coh = (lam1 - lam2) / (lam1 + lam2 + 1e-30)
        self.A = torch.cos(2.0 * theta).detach()
        self.B = torch.sin(2.0 * theta).detach()
        self.W = (coh * torch.clamp(phig / 0.5, 0.0, 1.0)).detach()
        self.n = n

    @torch.no_grad()
    def __call__(self, x, y):
        """bilinear lookup of (cos2th, sin2th, W) at query points; detached coefficient fields
           (the energy integrand needs no gradients THROUGH the frozen orientation)."""
        n = self.n
        u = (x.reshape(-1) * n - 0.5).clamp(0.0, n - 1.0)
        v = (y.reshape(-1) * n - 0.5).clamp(0.0, n - 1.0)
        i0 = u.floor().long().clamp(0, n - 2); j0 = v.floor().long().clamp(0, n - 2)
        tu = (u - i0.to(DTYPE)).unsqueeze(-1); tv = (v - j0.to(DTYPE)).unsqueeze(-1)
        out = []
        for M in (self.A, self.B, self.W):
            m00 = M[j0, i0]; m01 = M[j0, i0 + 1]; m10 = M[j0 + 1, i0]; m11 = M[j0 + 1, i0 + 1]
            m0 = m00.unsqueeze(-1) * (1 - tu) + m01.unsqueeze(-1) * tu
            m1 = m10.unsqueeze(-1) * (1 - tu) + m11.unsqueeze(-1) * tu
            out.append(m0 * (1 - tv) + m1 * tv)
        return out[0], out[1], out[2]

def _frac_energy(phi, gradphi2, lap_phi, Gc_loc, cfg):
    """crack-surface energy density. pf_order 2 = AT1/AT2 2nd-order (unchanged); pf_order 4 =
       Borden 4th-order (AT2-type): (Gc/2)[phi^2/l + (l/2)|grad phi|^2 + (l^3/16)(Lap phi)^2].
       Borden CMAME 273 (2014) Eq(15) mapped to our convention (phi=1-c, Miehe l = 2*l0)."""
    l = cfg.l
    if getattr(cfg, "pf_order", 2) == 4:
        return 0.5 * Gc_loc * (phi ** 2 / l + 0.5 * l * gradphi2 + (l ** 3 / 16.0) * lap_phi ** 2)
    if cfg.at == 1:
        return (3.0 / 8.0) * Gc_loc * (phi / l + l * gradphi2)          # AT1: Gc/c_w, c_w=8/3
    return 0.5 * Gc_loc * (phi ** 2 / l + l * gradphi2)                 # AT2: Gc/c_w, c_w=2

def energy_terms(exx, eyy, exy, phi, gradphi2, phi_prev, cfg, x=None, y=None, lap_phi=None):
    Gc_loc = cfg.Gc
    if getattr(cfg, "gc_grip", 0.0) > 0 and x is not None:
        Gc_loc = cfg.Gc * (1.0 + cfg.gc_grip * gc_bump(x, y, cfg))
    tr = exx + eyy
    tr_p = torch.clamp(tr, min=0.0); tr_m = tr - tr_p
    if cfg.split == "hybrid":
        # Ambati-style HYBRID: ISOTROPIC degradation rules the MECHANICS (the crack is soft in
        # ALL modes, incl. mode-II sliding -- what Amor gets right) while phi is DRIVEN by the
        # SPECTRAL tension energy only (starves the compressive/shear lobes: the corner AND the
        # upper-lobe wing -- what Amor gets wrong). NOT a single functional: the two couplings
        # are detached so the gradients equal the staggered hybrid scheme's, folded into one
        # Adam loop. e_el's VALUE = g*psi_full (mechanical energy) => reaction force, logs and
        # the plateau meter stay meaningful. (Ambati's crack-interpenetration guard is omitted:
        # SEN shear is a sliding case, no face closure.)
        psi_full = 0.5 * cfg.lam * tr ** 2 + cfg.mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)
        rad = torch.sqrt(((exx - eyy) * 0.5) ** 2 + exy ** 2 + 1e-30)
        e1 = tr * 0.5 + rad
        e2 = tr * 0.5 - rad
        D = (0.5 * cfg.lam * tr_p ** 2
             + cfg.mu * (torch.clamp(e1, min=0.0) ** 2 + torch.clamp(e2, min=0.0) ** 2))
        g_all = (1.0 - phi) ** 2 + cfg.g_floor
        g_det = (1.0 - phi.detach()) ** 2 + cfg.g_floor
        e_el = g_det * psi_full + (g_all - g_all.detach()) * D.detach()
        e_fr = _frac_energy(phi, gradphi2, lap_phi, Gc_loc, cfg)
        e_ir = cfg.gamma_ir * torch.clamp(phi_prev - cfg.ir_tol - phi, min=0.0) ** 2
        return e_el, e_fr, e_ir
    if cfg.split == "directional":
        # crack-frame split ON the band (opening enn+ AND sliding ent degradable => mode II slides
        # free, unlike spectral) blended by W to SPECTRAL in the bulk (tension-only driving =>
        # clamped corners starve, unlike Amor). Orientation/W frozen per step (cfg.dirfield).
        A, B, W = cfg.dirfield(x, y)
        th = 0.5 * torch.atan2(B, A)
        nx, ny = torch.cos(th), torch.sin(th)
        tx, ty = -ny, nx
        enn = exx * nx * nx + 2.0 * exy * nx * ny + eyy * ny * ny
        ett = exx * tx * tx + 2.0 * exy * tx * ty + eyy * ty * ty
        ent = exx * nx * tx + exy * (nx * ty + ny * tx) + eyy * ny * ty
        enn_p = torch.clamp(enn, min=0.0); enn_m = enn - enn_p
        psi_p_dir = 0.5 * cfg.lam * tr_p ** 2 + cfg.mu * enn_p ** 2 + 2.0 * cfg.mu * ent ** 2
        psi_m_dir = 0.5 * cfg.lam * tr_m ** 2 + cfg.mu * enn_m ** 2 + cfg.mu * ett ** 2
        rad = torch.sqrt(((exx - eyy) * 0.5) ** 2 + exy ** 2 + 1e-30)
        e1 = tr * 0.5 + rad
        e2 = tr * 0.5 - rad
        psi_p_sp = (0.5 * cfg.lam * tr_p ** 2
                    + cfg.mu * (torch.clamp(e1, min=0.0) ** 2 + torch.clamp(e2, min=0.0) ** 2))
        psi_m_sp = (0.5 * cfg.lam * tr_m ** 2
                    + cfg.mu * (torch.clamp(e1, max=0.0) ** 2 + torch.clamp(e2, max=0.0) ** 2))
        psi_p = W * psi_p_dir + (1.0 - W) * psi_p_sp
        psi_m = W * psi_m_dir + (1.0 - W) * psi_m_sp
    elif cfg.split == "spectral":
        # Miehe SPECTRAL split, plane strain (eps_zz = 0 is the third eigenvalue, drops out).
        # Tension-only eigen driving: in pure shear psi_p is HALF of Amor's (e1=-e2) => weaker
        # corner/shear-band driving (measured: 53% of Amor's at the corner). Same total psi.
        rad = torch.sqrt(((exx - eyy) * 0.5) ** 2 + exy ** 2 + 1e-30)
        e1 = tr * 0.5 + rad
        e2 = tr * 0.5 - rad
        psi_p = (0.5 * cfg.lam * tr_p ** 2
                 + cfg.mu * (torch.clamp(e1, min=0.0) ** 2 + torch.clamp(e2, min=0.0) ** 2))
        psi_m = (0.5 * cfg.lam * tr_m ** 2
                 + cfg.mu * (torch.clamp(e1, max=0.0) ** 2 + torch.clamp(e2, max=0.0) ** 2))
    elif cfg.split == "iso":
        # NO split (Manav et al. se_split=None): the FULL isotropic strain energy degrades in
        # EVERY mode -- tension, compression and shear alike -- so damage is driven isotropically
        # and the crack BRANCHES (paper example #2, the native-branching leg; contrast hybrid/amor
        # which protect compression => a single band). Plane strain: eps_zz = 0. No psi_minus.
        psi_p = 0.5 * cfg.lam * tr ** 2 + cfg.mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)
        psi_m = torch.zeros_like(psi_p)
    else:
        dev_xx, dev_yy, dev_zz = exx - tr / 3.0, eyy - tr / 3.0, -tr / 3.0
        dev_con = dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * exy ** 2
        # gamma_star = star-convex-type family (Vicentini et al. 2024): deducts the compressive-
        # volumetric energy from the DRIVING part (and banks it undegraded) => suppresses damage
        # in compression-dominated states. gamma_star=0 == Amor exactly. Sum psi_p+psi_m
        # invariant. (Check the paper's admissible gamma range before large values. Measured on
        # the clamped corner: ~0% leverage there -- kept for compression-dominated problems.)
        psi_p = (0.5 * cfg.K * tr_p ** 2 + cfg.mu * dev_con
                 - cfg.gamma_star * 0.5 * cfg.K * tr_m ** 2)
        psi_m = (1.0 + cfg.gamma_star) * 0.5 * cfg.K * tr_m ** 2
    g = (1.0 - phi) ** 2 + cfg.g_floor
    e_el = g * psi_p + psi_m
    e_fr = _frac_energy(phi, gradphi2, lap_phi, Gc_loc, cfg)
    # ir_tol = irreversibility DEAD-BAND: lets phi relax by up to tol below phi_prev, so the
    # per-iteration MC-noise fog cannot RATCHET up across load steps (measured: far-field phi
    # 0.004 -> 0.013 over 10 steps with tol=0); real cracks (phi~1) are unaffected.
    e_ir = cfg.gamma_ir * torch.clamp(phi_prev - cfg.ir_tol - phi, min=0.0) ** 2
    return e_el, e_fr, e_ir

def _blur_grid(M, sig_cells):
    """separable Gaussian blur of an (n,n) torch grid -- used to widen the new-band stratum
       to the notch stratum's cross profile (sigma_band), so the PROPAGATED crack gets the
       same per-unit-length point density as the notch (the core-only mask
       left the new crack ~2.7x sparser, flanks uncovered)."""
    n = M.shape[-1]
    k = int(4 * sig_cells) | 1
    t = torch.arange(k, dtype=DTYPE, device=M.device) - (k - 1) / 2.0
    ker = torch.exp(-0.5 * (t / max(sig_cells, 1e-6)) ** 2); ker = ker / ker.sum()
    M = M.reshape(1, 1, n, n)
    M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, -1, 1), padding=(k // 2, 0))
    M = torch.nn.functional.conv2d(M, ker.reshape(1, 1, 1, -1), padding=(0, k // 2))
    return M.reshape(n, n)

# --------------------------------------------------------------------------
# mesh-free Monte-Carlo quadrature

class Sampler:
    """Mixture: PHYSICAL-uniform bulk + CRACK stratum (whole crack) + damage/process-zone. Unbiased
       1/rho (exact categorical grid density), every stratum weighted by |detJ| so the sampling is
       PHYSICALLY uniform under a geo map (identity => |detJ|=1, parametric == physical, unchanged).
       B2 TIMING: built from prev_model, frozen all step. IGA-safe (phi0/phi through to_phys)."""
    def __init__(self, cfg, prev_model=None, N=None, seed=0):
        self.cfg = cfg
        self.N = cfg.N if N is None else N
        self.sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
        self.rng = np.random.default_rng(seed + 12345)
        wu, wn, wd = cfg.w_unif, cfg.w_notch, cfg.w_damage
        self.gn = cfg.grid_n
        self.cell = 1.0 / self.gn
        gm = getattr(cfg, "geomap", None)
        self.is_curved = gm is not None and not getattr(gm, "is_identity", False)
        self.Jcell = self._jac_grid()          # |detJ| per cell => PHYSICAL-density weight
        # area of the body as a fraction of the parametric square; examples with a hole or a
        # cut-out shrink it in prepare_sampler so the bulk density stays normalised
        self.dom_area = 1.0
        problem_of(cfg).prepare_sampler(self)
        # PHYSICAL-uniform bulk on a curved patch (parametric-uniform Sobol on identity/none)
        self.UnifGrid = (self.Jcell / self.Jcell.sum()) if self.is_curved else None
        self._build_notch_grid(prev_model)     # CRACK stratum = max(phi0,phi_prev)*|detJ| (whole crack)
        self._build_damage_grid(prev_model)    # Dgrid (process zone) + optional Ngrid_raw (manual band)
        if self.Dgrid is None:
            wd = 0.0
        s = wu + wn + wd
        self.wu, self.wn, self.wd = wu / s, wn / s, wd / s
        self.nu = int(round(self.wu * self.N))
        self.nd = int(round(self.wd * self.N)) if self.Dgrid is not None else 0
        self.nn_ = self.N - self.nu - self.nd
        # The crack stratum (nn_ pts) covers the WHOLE crack (notch AND propagated) at ONE physical
        # density => the notch density is AUTO-synced onto the propagated crack, at a fixed budget.
        # OPTIONAL manual band (rho_new>0): ADD n_add pts on the propagated crack at an explicit
        # physical areal density (pts per physical area) -- for nucleation / extra concentration.
        self.n_add = 0
        if self.Ngrid_raw is not None and self.cfg.rho_new > 0:
            self.n_add = int(round(self.cfg.rho_new * self.A_new / (self.gn ** 2)))
            self.Ngrid = (self.Ngrid_raw / self.Ngrid_raw.sum()) if self.n_add > 0 else None
        else:
            self.Ngrid = None
        self.Nt = self.N + self.n_add
        self.fu = self.nu / self.Nt
        self.fn = self.nn_ / self.Nt
        self.fd = self.nd / self.Nt
        self.fnew = self.n_add / self.Nt

    def _jac_grid(self):
        """|detJ| at the grid cell centers (n x n). 1.0 for identity / no geo map (=> parametric
           == physical: unchanged SEN behavior). On a curved patch this is the physical-area weight
           that makes every grid stratum PHYSICALLY uniform (the |detJ| stretch can be large)."""
        n = self.gn
        gm = getattr(self.cfg, "geomap", None)
        if gm is None or getattr(gm, "is_identity", False):
            return np.ones((n, n))
        xs = (torch.arange(n, dtype=DTYPE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        # gm.jac() runs autograd internally (its own leaf graph) => do NOT wrap in no_grad
        _, _, _, _, det = gm.jac(GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE))
        return det.abs().reshape(n, n).cpu().numpy()

    def _build_notch_grid(self, prev_model):
        """CRACK stratum = max(phi0, phi_prev) * |detJ| categorical grid => a PHYSICALLY-uniform
           density over the WHOLE current crack, NOTCH AND PROPAGATED (auto-syncs the notch density
           onto the crack path). Geometry-agnostic / IGA-safe (phi0/phi go through to_phys, so it
           lives at the physical crack). Step 0 (prev_model=None) => just phi0 (the notch)."""
        c = self.cfg
        n = self.gn
        xs = (torch.arange(n, dtype=DTYPE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        gx, gy = GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE)
        with torch.no_grad():
            crack = phi0_field(gx, gy, c)
            if prev_model is not None:
                crack = torch.maximum(crack, phi_value(prev_model, gx, gy, c))
            # an example with no pre-existing crack still needs points where one will start
            crack = problem_of(c).crack_stratum(crack, gx, gy, c)
        crack = crack.reshape(n, n).cpu().numpy()
        W = np.clip(crack, 1e-12, None) * self.Jcell   # crack * |detJ| => physical density
        self.NotchGrid = W / W.sum()                   # (kept name) whole-crack point distribution
        self.A_notch = float(((crack > 0.5) * self.Jcell).sum())   # physical crack-band area (manual)

    def _build_damage_grid(self, prev_model):
        c = self.cfg
        self.Dgrid = None; self.Ngrid_raw = None; self.A_new = 0.0
        if prev_model is None or (c.w_damage <= 0 and c.rho_new == 0):
            return
        n = self.gn
        xs = (torch.arange(n, dtype=DTYPE) + 0.5) / n
        GX, GY = torch.meshgrid(xs, xs, indexing="xy")
        gx, gy = GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE)
        phi = phi_value(prev_model, gx, gy, cfg=c)
        phi0 = phi0_field(gx, gy, c)
        if c.w_damage > 0:
            D = phi * (1.0 - phi) + c.eta_damage * phi0
            if c.beta_drive > 0:
                drv = drive_field(prev_model, gx, gy, c, c.ud_prev)
                D = D + c.beta_drive * drv / (drv.max() + 1e-30)
            D = D.reshape(n, n).cpu().numpy()
            D = np.clip(D, 1e-12, None) * self.Jcell            # physical density
            self.Dgrid = D / D.sum()
        if c.rho_new != 0:                                      # manual band (rho_new>0)
            mask = ((phi0 < 0.4) & (phi > 0.5)).to(DTYPE)
            mask_np = mask.reshape(n, n).cpu().numpy()          # BINARY new-crack band
            Dn = (phi * (1.0 - phi) + c.eta_new * phi) * mask
            Dn = _blur_grid(Dn.reshape(n, n), c.sigma_band * n).cpu().numpy()
            W = np.clip(Dn, 0.0, None) * self.Jcell
            if W.sum() > 0:
                self.Ngrid_raw = W                              # point distribution (weighted)
                self.A_new = float((mask_np * self.Jcell).sum())  # binary band PHYSICAL AREA

    def _sample_grid(self, grid, m):
        n = self.gn
        idx = self.rng.choice(n * n, size=m, p=grid.ravel())
        iy, ix = np.divmod(idx, n)
        jx = (ix + self.rng.random(m)) * self.cell
        jy = (iy + self.rng.random(m)) * self.cell
        return torch.tensor(np.stack([jx, jy], 1), dtype=DTYPE, device=utils.DEVICE)

    def sample(self):
        parts = []
        # examples whose body does not fill the parametric square (a hole, a cut-out) redraw the
        # bulk stratum themselves so the rejected points are replaced rather than lost
        own = problem_of(self.cfg).sample_uniform(self, self.nu)
        if own is not None:
            parts.append(own)
        elif self.is_curved:                        # PHYSICAL-uniform bulk (|detJ|-weighted grid)
            parts.append(self._sample_grid(self.UnifGrid, self.nu))
        else:                                       # identity/none: parametric-uniform Sobol
            parts.append(torch.as_tensor(self.sobol.random(self.nu), dtype=DTYPE).to(utils.DEVICE))
        if getattr(self.cfg, "fixed_notch", False):    # middle stratum = OLD fixed symmetric band
            c = self.cfg
            sx = torch.rand(self.nn_, 1, device=utils.DEVICE) * (c.nx1 - c.nx0) + c.nx0
            sy = torch.full((self.nn_, 1), c.ny, device=utils.DEVICE) + torch.randn(self.nn_, 1, device=utils.DEVICE) * c.sigma_band
            parts.append(torch.cat([sx, sy], dim=1))
        else:                                           # DEFAULT: adaptive crack stratum (grid)
            parts.append(self._sample_grid(self.NotchGrid, self.nn_))
        if self.nd > 0:
            parts.append(self._sample_grid(self.Dgrid, self.nd))
        if self.n_add > 0:
            parts.append(self._sample_grid(self.Ngrid, self.n_add))
        return torch.cat(parts, dim=0)

    def _grid_density(self, grid, x, y):
        if grid is None:
            return torch.zeros_like(x)
        n = self.gn
        ix = torch.clamp((x / self.cell).long(), 0, n - 1)
        iy = torch.clamp((y / self.cell).long(), 0, n - 1)
        Dg = torch.tensor(grid, dtype=DTYPE, device=utils.DEVICE)
        p = Dg[iy.squeeze(-1), ix.squeeze(-1)].unsqueeze(-1)     # categorical prob of the cell
        return p / (self.cell ** 2)                             # -> density (per area)

    def _notch_density(self, x, y):
        """OLD (pre-overhaul) fixed notch-band density: analytic Gaussian-in-y about ny x uniform-in-x
           over [nx0,nx1]. phi-INDEPENDENT and SYMMETRIC about y=ny => anchors the branching tip
           symmetrically. Used by --fixed_notch to restore SYMMETRIC BIFURCATION (the adaptive crack
           stratum is magnitude-weighted => phi-feedback that starves one branch => single kink)."""
        c = self.cfg
        sig, L = c.sigma_band, c.nx1 - c.nx0
        s2pi = float(np.sqrt(2.0 * np.pi)); s2 = float(np.sqrt(2.0))
        gy = torch.exp(-((y - c.ny) ** 2) / (2 * sig ** 2)) / (sig * s2pi)
        a_lo = (c.nx0 - x) / (sig * s2); a_hi = (c.nx1 - x) / (sig * s2)
        return (1.0 / L) * gy * 0.5 * (torch.erf(a_hi) - torch.erf(a_lo))

    def rho_and_mask(self, pts):
        c = self.cfg
        x, y = pts[:, 0:1], pts[:, 1:2]
        inom = problem_of(c).in_domain(x, y, c).to(DTYPE)
        unif = self._grid_density(self.UnifGrid, x, y) if self.is_curved else inom / self.dom_area
        notch = (self._notch_density(x, y) if getattr(c, "fixed_notch", False)
                 else self._grid_density(self.NotchGrid, x, y))
        rho = (self.fu * unif + self.fn * notch
               + self.fd * self._grid_density(self.Dgrid, x, y)
               + self.fnew * self._grid_density(self.Ngrid, x, y))
        return rho, inom

def estimate_Pi(net, pts, sampler, cfg, phi_prev_vals, u_delta):
    x, y = pts[:, 0:1].clone(), pts[:, 1:2].clone()
    exx, eyy, exy, phi, gradphi2, detJ, lap_phi = kinematics(net, x, y, cfg, u_delta)
    e_el, e_fr, e_ir = energy_terms(exx, eyy, exy, phi, gradphi2, phi_prev_vals, cfg,
                                    x=pts[:, 0:1], y=pts[:, 1:2], lap_phi=lap_phi)
    # Pi = int_param f |detJ| dxi ~ E_rho[f |detJ| / rho]  (detJ==1 with --geo none)
    rho, inom = sampler.rho_and_mask(pts)
    w = (inom * detJ / rho.clamp_min(1e-300)).detach()
    Pi_el = (e_el * w).mean()
    Pi_fr = (e_fr * w).mean()
    Pi_ir = (e_ir * w).mean()
    return Pi_el, Pi_fr, Pi_ir, phi.detach()

# --------------------------------------------------------------------------
# reaction force

def reaction_force(net, cfg, phi_prev_model, u_delta_nd, seed=999):
    """F = dPi_el/du_delta,nd / U_ref (energy-consistent reaction; fracture/irrev terms are
       u_delta-independent so grad of total == grad of elastic).

       Examples whose load does not ride the displacement ansatz (the ring, pulled through a
       boundary penalty) extract it their own way -- see Problem.reaction_force."""
    own = problem_of(cfg).reaction_force(net, cfg, phi_prev_model, u_delta_nd, seed)
    if own is not None:
        return own
    smp = Sampler(cfg, prev_model=phi_prev_model, N=cfg.N_meter, seed=seed)
    pts = smp.sample()
    phi_prev = phi_value(phi_prev_model, pts[:, 0:1], pts[:, 1:2], cfg)
    ud = torch.tensor(float(u_delta_nd), device=utils.DEVICE, requires_grad=True)
    Pi_el, _, _, _ = estimate_Pi(net, pts, smp, cfg, phi_prev, ud)
    dPi = torch.autograd.grad(Pi_el, ud)[0]
    return float(dPi) / cfg.U_ref

# --------------------------------------------------------------------------
# load stepping

def solve_step(net, prev_model, cfg, ud, save_dir, logger, step, n_iter):
    """minimize total Pi over MLP weights + encoding grids at fixed delta (warm-started).
       Adam + plateau stop. Encoding gets its own lr + tiny L2 (unsampled-cell drift guard)."""
    groups = [{"params": net.mlp.parameters(), "lr": cfg.lr},
              {"params": net.enc.parameters(), "lr": cfg.lr_enc}]
    opt = torch.optim.Adam(groups)
    sampler = Sampler(cfg, prev_model=prev_model, seed=cfg.seed + 1000 * step)
    if sampler.n_add > 0:
        logger.info(f"  sampler: +{sampler.n_add} adaptive pts on the new band (total {sampler.Nt})")
    pts = sampler.sample()
    phi_prev = phi_value(prev_model, pts[:, 0:1], pts[:, 1:2], cfg)
    hist = []
    t0 = time.time()
    for it in range(n_iter):
        net.enc.set_progress(step, it)
        if it > 0 and it % cfg.resample_every == 0:
            # redraw POINTS only (Sobol/rng streams advance); the density grid stays the
            # one built from prev_model at step start (B2 timing, and rebuilding the 256^2
            # drive grid every resample would be pure waste at resample_every=1)
            pts = sampler.sample()
            phi_prev = phi_value(prev_model, pts[:, 0:1], pts[:, 1:2], cfg)
        opt.zero_grad()
        Pi_el, Pi_fr, Pi_ir, _ = estimate_Pi(net, pts, sampler, cfg, phi_prev, cfg.u_delta_nd_step)
        Pi = Pi_el + Pi_fr + Pi_ir
        loss = Pi
        # boundary conditions that cannot be imposed exactly on a curved patch enter here
        Pi_bc = cfg.problem.extra_loss(net, cfg, cfg.u_delta_nd_step)
        if Pi_bc is not None:
            loss = loss + Pi_bc
        if cfg.lambda_reg > 0:
            loss = loss + cfg.lambda_reg * sum((p ** 2).sum() for p in net.mlp.parameters())
        if cfg.enc_reg > 0:
            loss = loss + cfg.enc_reg * net.enc.reg()
        loss.backward(); opt.step()
        hist.append(float(Pi.detach()))
        if it % cfg.log_every == 0 or it == n_iter - 1:
            bc = "" if Pi_bc is None else f" bc {float(Pi_bc.detach()):.3e}"
            logger.info(f"  step {step:3d} delta {cfg.U_ref*ud:.3e}  it {it:5d}  "
                        f"Pi {hist[-1]:.5e}  (el {float(Pi_el.detach()):.4e} "
                        f"fr {float(Pi_fr.detach()):.4e} ir {float(Pi_ir.detach()):.3e}{bc})  "
                        f"gfine {net.enc.fine_max():.2e}  gate {net.enc.gates[-1]:.2f}  "
                        f"{time.time()-t0:5.0f}s")
        # plateau early-stop (only once all levels are gated on)
        if len(hist) > cfg.plateau_win and net.enc.gates[-1] >= 1.0:
            recent = hist[-cfg.plateau_win:]
            if (max(recent) - min(recent)) / (abs(np.mean(recent)) + 1e-30) < cfg.plateau_tol:
                logger.info(f"  step {step} plateau @ it {it}")
                break
    net.enc.set_progress(step + 1, 0)              # gates fully on for meters/plots/next step
    return hist

def run_cli(problem_name, argv=None, description=None, defaults=None):
    """Parse a command line for one example and run it -- the whole body of an entry script.

       `defaults` presets the flags for one particular case of that example (the SEN plate runs
       four of them), so the script reproduces the published run with no arguments at all while
       every flag stays overridable on the command line."""
    problem = utils.get_problem(problem_name)
    ap = build_parser(problem, description=description)
    if defaults:
        ap.set_defaults(**defaults)
    return run(ap.parse_args(argv), problem)

def validate(cfg):
    """Guards for combinations the solver does not implement (fail loudly, before the run)."""
    assert len(cfg.enc_start) == len(cfg.levels), "--enc_start must list one start per level"
    if cfg.pf_order == 4:
        assert cfg.at == 2, "--pf_order 4 is the AT2-type 4th-order model (Borden); use --at 2"
        assert cfg.geo in ("identity", "none"), \
            "--pf_order 4 needs identity/none geometry (SEN square); curved-geo Laplacian pullback " \
            "is not implemented"
    if cfg.geo != "none":
        assert cfg.split != "directional", "--geo: directional split not ported (falsified branch)"
        assert cfg.grip_release <= 0, "--geo: --grip_release not ported (physical-annulus inverse map)"
    if getattr(cfg, "fixed_notch", False):
        # the fixed band draws the parametric square straight from nx0/nx1/ny, which are the
        # PHYSICAL notch coordinates for every other example (ring: x in [17,20] => NaN energy;
        # nucleation: a degenerate zero-length notch => division by zero)
        assert cfg.problem_name == "sen", \
            "--fixed_notch is the notched-square band; it does not apply to " \
            f"'{cfg.problem_name}' (its notch line is not in the parametric square)"
    if cfg.problem_name == "ring":
        # the notch line, the symmetry lift and the outer-arc penalty are all written for the
        # annulus parametrization (xi = radial, eta = angular): on a square patch they would be
        # applied to coordinates they do not belong to, silently (phi0 ~ 1e-35, no notch at all)
        assert cfg.geo not in ("identity", "distort", "none"), \
            "the ring example needs its annulus patch: --geo ring, or a patch JSON with the " \
            "same radial/angular parametrization"

def run(a, problem):
    """Run one example end to end. `a` is the parsed command line, `problem` its Problem."""
    cfg = Cfg(a, problem=problem)
    validate(cfg)
    os.makedirs(a.out, exist_ok=True)
    logger = get_logger(a.out)
    logger.info(f"device={utils.DEVICE}  dtype={utils.DTYPE}  problem={cfg.problem_name}  "
                f"cfg={json.dumps(cfg.to_dict())}")
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    cfg.geomap = problem.build_geomap(cfg, logger)
    plot_geometry(cfg, a.out)                               # patch figure (once; fixed geometry)
    plot_geometry_bc(cfg, a.out)                            # physical space + BC / control net
    plot_parametric(cfg, a.out)                             # parametric [0,1]^2 reference domain
    problem.plot_domain(cfg, a.out)                         # example-specific geometry figure
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    net = NetMS(cfg).to(utils.DEVICE)
    n_grid = sum(G.numel() for G in net.enc.grids)
    logger.info(f"encoding: levels={list(cfg.levels)} ch={cfg.enc_ch} "
                f"grid params={n_grid} (finest h={1.0/(cfg.levels[-1]-2):.4f} = "
                f"{1.0/(cfg.levels[-1]-2)/cfg.l:.2f}l)")
    if problem.setup(net, cfg, a.out, logger):              # e.g. an elastic verification run
        logger.info(f"done -> {a.out}")
        return cfg, net, None
    schedule = cfg.schedule()
    FD = []
    prev_model = None
    start_step = 0
    ckpt = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt):
        ck = load_ckpt(ckpt, net)
        start_step, FD = ck["step"] + 1, ck["FD"]
        if ck["prev"] is not None:
            prev_model = NetMS(cfg).to(utils.DEVICE); prev_model.load_state_dict(ck["prev"])
            for p in prev_model.parameters(): p.requires_grad_(False)
            prev_model.eval()
        net.enc.set_progress(1, 0)
        cfg.grip_released = tuple(bool(b) for b in
                                  ck["cfg"].get("grip_released", [False, False, False, False]))
        logger.info(f"RESUMED at step {start_step}  grip_released={cfg.grip_released}")

    for step in range(start_step, len(schedule)):
        ud = schedule[step]
        cfg.u_delta_nd_step = ud                          # current step's top-disp DOF
        cfg.ud_prev = schedule[step - 1] if step > 0 else 0.0   # prev converged load (drive eval)
        update_grip_release(cfg, prev_model, logger)      # opt-in corner-gate release
        if cfg.split == "directional":
            cfg.dirfield = DirField(prev_model, cfg)      # frozen orientation field for this step
        n_iter = cfg.iters0 if step == 0 else cfg.iters
        logger.info(f"=== load step {step}/{len(schedule)-1}  delta={cfg.U_ref*ud:.4e}  "
                    f"iters={n_iter} ===")
        solve_step(net, prev_model, cfg, ud, a.out, logger, step, n_iter)

        F = reaction_force(net, cfg, prev_model, ud)
        FD.append([cfg.U_ref * ud, F])
        logger.info(f"    -> F = {F:.4f} N")

        # freeze current net (MLP + encoding) as phi_prev / sampler source for the NEXT step
        prev_state = copy.deepcopy(net.state_dict())
        prev_model = NetMS(cfg).to(utils.DEVICE); prev_model.load_state_dict(prev_state)
        for p in prev_model.parameters(): p.requires_grad_(False)
        prev_model.eval()
        prev_model.enc.set_progress(1, 0)

        save_ckpt(ckpt, net, prev_state, step, ud, FD, cfg)   # rolling root ckpt (--resume)
        last = (step == len(schedule) - 1)
        step_dir = os.path.join(a.out, f"step_{step:03d}")
        if cfg.ckpt_every > 0 and (step % cfg.ckpt_every == 0 or last):
            os.makedirs(step_dir, exist_ok=True)              # per-step snapshot folder
            save_ckpt(os.path.join(step_dir, "ckpt.pt"), net, prev_state, step, ud, FD, cfg)
        if cfg.plot_every > 0:
            plot_now = (step % cfg.plot_every == 0) or last
        else:
            plot_now = (step % max(1, len(schedule) // 12) == 0) or last
        if plot_now:
            os.makedirs(step_dir, exist_ok=True)
            smp = Sampler(cfg, prev_model=prev_model, seed=cfg.seed + 7)
            plot_phi(net, cfg, step_dir, f"s{step:03d}", ud, sample_pts=smp.sample().cpu().numpy())
            plot_linecut(net, cfg, step_dir, f"s{step:03d}", ud)
            plot_fields(net, cfg, step_dir, f"s{step:03d}", ud)
            plot_FD_solo(FD, cfg, step_dir)                   # NO-compare curve -> step folder
            problem.plot(net, cfg, step_dir, f"s{step:03d}", ud)
        plot_FD_compare(FD, cfg, a.out, a.fem)                # compare curve -> run root only
        np.savetxt(os.path.join(a.out, "FD.txt"), np.array(FD), header="delta(mm)  F(N)")

    logger.info("=" * 70)
    FD = np.array(FD)
    if len(FD):
        ipk = int(np.argmax(FD[:, 1]))
        logger.info(f"DEM peak F {FD[ipk,1]:.3f} N @ delta {FD[ipk,0]:.3e}")
    plot_FD_compare(FD.tolist(), cfg, a.out, a.fem)
    plot_phi(net, cfg, a.out, "final", schedule[-1])
    plot_linecut(net, cfg, a.out, "final", schedule[-1])
    problem.finalize(net, cfg, a.out, logger)
    logger.info(f"done -> {a.out}")
    return cfg, net, FD
