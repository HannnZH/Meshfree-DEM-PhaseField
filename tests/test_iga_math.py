"""IGA layer math verification (no training): identity exactness, distort invariances,
J^{-T} pullback vs analytic gradient, |detJ| quadrature vs analytic integrals.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU: leave a running GPU job alone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, math
import numpy as np
import torch

import pfpiml as S

def mkcfg(geo):
    a = argparse.Namespace(
        l=0.01, nsteps=18, delta_max=2.7e-3, iters=2000, iters0=4000, resample=1,
        lr=5e-4, reg=0.0, gamma_ir=1e3, ir_tol=0.0, at=2, phi_map="sigmoid",
        phi_bias=-4.0, grip_l=3.0, grip_release=0.0, gc_grip=1.0, gamma_star=0.0,
        split="hybrid", geo=geo, levels="32,128,384", enc_ch=2, lr_enc=2e-3,
        enc_reg=1e-8, enc_start="0,0,0", enc_ramp=1, N=12000, w_unif=0.4,
        w_notch=0.3, w_damage=0.3, eta_damage=0.3, beta_drive=0.5, grid_n=256,
        rho_new=2e5, eta_new=0.3, width=128, depth=4, seed=0, plateau_tol=2e-4,
        plateau_win=400, out="", fem="", log_every=300, ckpt_every=0, plot_every=0,
        resume=False)
    cfg = S.Cfg(a)
    cfg.geomap = S.build_geomap(geo)
    cfg.u_delta_nd_step = 0.5
    return cfg

def report(name, val, tol):
    flag = "PASS" if val < tol else "FAIL"
    print(f"  {flag}  {name}: {val:.3e}  (tol {tol:.0e})")
    return val < tol

torch.manual_seed(0); np.random.seed(0)
cfg_n = mkcfg("none")
cfg_i = mkcfg("identity")
cfg_d = mkcfg("distort")

# one net shared by all cfgs (identical weights => field diffs isolate the geo layer)
torch.manual_seed(0)
net = S.NetMS(cfg_n).to(S.DEVICE)

N = 20000
rng = torch.Generator().manual_seed(7)
pts = torch.rand(N, 2, generator=rng, dtype=S.DTYPE, device=S.DEVICE)
x, y = pts[:, 0:1], pts[:, 1:2]
ok = True

print("== [1] identity patch: map + fields + kinematics vs --geo none ==")
with torch.no_grad():
    xp, yp = cfg_i.geomap(x, y)
ok &= report("map |G(xi)-xi|_max", float((xp - x).abs().max().item() + (yp - y).abs().max().item()), 1e-12)
kn = S.kinematics(net, x.clone(), y.clone(), cfg_n, 0.5)
ki = S.kinematics(net, x.clone(), y.clone(), cfg_i, 0.5)
for nm, i_ in [("exx", 0), ("eyy", 1), ("exy", 2), ("phi", 3), ("gradphi2", 4)]:
    d = float((kn[i_] - ki[i_]).abs().max())
    sc = float(kn[i_].abs().max()) + 1e-30
    ok &= report(f"kinematics {nm} rel", d / sc, 1e-9)
ok &= report("detJ-1 (identity)", float((ki[5] - 1.0).abs().max()), 1e-9)

print("== [2] distort patch: invariances ==")
gm = cfg_d.geomap
xs = torch.rand(4096, 1, generator=rng, dtype=S.DTYPE, device=S.DEVICE)
half = torch.full_like(xs, 0.5)
with torch.no_grad():
    mx, my = gm(xs, half)                              # midline eta=0.5
    e0x, e0y = gm(xs, torch.zeros_like(xs))            # bottom edge
    e1x, e1y = gm(xs, torch.ones_like(xs))             # top edge
    s0x, s0y = gm(torch.zeros_like(xs), xs)            # left edge
    s1x, s1y = gm(torch.ones_like(xs), xs)             # right edge
ok &= report("midline invariance |G(xi,.5)-(xi,.5)|", float((mx - xs).abs().max() + (my - 0.5).abs().max()), 1e-12)
ok &= report("bottom edge", float((e0x - xs).abs().max() + e0y.abs().max()), 1e-12)
ok &= report("top edge", float((e1x - xs).abs().max() + (e1y - 1).abs().max()), 1e-12)
ok &= report("left edge", float(s0x.abs().max() + (s0y - xs).abs().max()), 1e-12)
ok &= report("right edge", float((s1x - 1).abs().max() + (s1y - xs).abs().max()), 1e-12)
a_, b_, c_, d_, det = gm.jac(x, y)
print(f"  info  distort detJ range [{float(det.min()):.4f}, {float(det.max()):.4f}], "
      f"max stretch {float(torch.stack([a_,b_,c_,d_]).abs().max()):.3f}")
ok &= report("distort min detJ > 0 (as -min)", -float(det.min()), 0.0)

print("== [3] pullback: physical grad of a known function under distort ==")
# f(x,y) = sin(2*x_phys + 3*y_phys): compute grad via param autodiff + J^{-T}, vs analytic
xg = x.clone().requires_grad_(True); yg = y.clone().requires_grad_(True)
xp, yp = gm(xg, yg)
f = torch.sin(2.0 * xp + 3.0 * yp)
fx_p, fy_p = torch.autograd.grad(f.sum(), [xg, yg])
fx = (d_ * fx_p - c_ * fy_p) / det
fy = (-b_ * fx_p + a_ * fy_p) / det
with torch.no_grad():
    gx_true = 2.0 * torch.cos(2.0 * xp + 3.0 * yp)
    gy_true = 3.0 * torch.cos(2.0 * xp + 3.0 * yp)
ok &= report("pullback df/dx rel", float((fx - gx_true).abs().max() / gx_true.abs().max()), 1e-9)
ok &= report("pullback df/dy rel", float((fy - gy_true).abs().max() / gy_true.abs().max()), 1e-9)

print("== [4] |detJ| quadrature: analytic integrals over the (same) unit square ==")
# distorted patch maps the square onto itself (boundary identity, det>0)
from scipy.stats import qmc
sob = torch.as_tensor(qmc.Sobol(d=2, scramble=True, seed=11).random(1 << 17), dtype=S.DTYPE)
sx, sy = sob[:, 0:1], sob[:, 1:2]
aa, bb, cc, dd, dt = gm.jac(sx, sy)
with torch.no_grad():
    px, py = gm(sx, sy)
for fn, lab, exact in [(lambda X, Y: torch.ones_like(X), "area", 1.0),
                       (lambda X, Y: X, "int x", 0.5),
                       (lambda X, Y: X * X * Y, "int x^2 y", 1.0 / 6.0),
                       (lambda X, Y: torch.sin(math.pi * X) * Y * Y, "int sin(pi x) y^2", 2.0 / (3.0 * math.pi))]:
    est = float((fn(px, py) * dt).mean())
    ok &= report(f"{lab} (exact {exact:.6f})", abs(est - exact) / abs(exact), 5e-4)

print("== [5] phi0/gc_bump consistency: distort evaluates them at PHYSICAL points ==")
with torch.no_grad():
    p0_param_as_phys = S.phi0_field(x, y, cfg_n)       # phi0 at (x,y) as physical
    p0_dist = S.phi0_field(x, y, cfg_d)                # phi0 at G(x,y)
    xpq, ypq = gm(x, y)
    p0_ref = S.phi0_field(xpq, ypq, cfg_n)             # phi0 evaluated at the mapped pts
ok &= report("phi0(G(xi)) == phi0_ref", float((p0_dist - p0_ref).abs().max()), 1e-12)
print(f"  info  phi0 distort-vs-naive max diff {float((p0_dist - p0_param_as_phys).abs().max()):.3e} "
      f"(should be LARGE-ish: proves the mapping is actually applied)")

print("== [6] estimate_Pi end-to-end, all three geos (hybrid split; torch RNG reseeded ==")
print("       per run so none/identity draw IDENTICAL sample points)")
vals = {}
for tag, cfg in [("none", cfg_n), ("identity", cfg_i), ("distort", cfg_d)]:
    torch.manual_seed(42)                              # sampler notch stratum uses global RNG
    smp = S.Sampler(cfg, prev_model=None, N=4096, seed=3)
    p = smp.sample()
    phi_prev = S.phi_value(None, p[:, 0:1], p[:, 1:2], cfg)
    Pi_el, Pi_fr, Pi_ir, _ = S.estimate_Pi(net, p, smp, cfg, phi_prev, 0.5)
    vals[tag] = (float(Pi_el), float(Pi_fr))
    print(f"  info  geo={tag:9s} Pi_el {float(Pi_el):.10e}  Pi_fr {float(Pi_fr):.10e}")
ok &= report("Pi_el none-vs-identity rel", abs(vals["none"][0] - vals["identity"][0]) / abs(vals["none"][0]), 1e-9)
ok &= report("Pi_fr none-vs-identity rel", abs(vals["none"][1] - vals["identity"][1]) / abs(vals["none"][1]), 1e-9)
print("  (distort differs at init by design -- ansatz parameterization differs; the "
      "converged physics is what must match, that is the i1 run)")

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
