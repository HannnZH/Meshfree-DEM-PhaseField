"""
Run-time figures: phi maps, displacement/strain fields, F-delta curves, patch views.

Written during the run so a job can be watched while it is still going. The network evaluators
(kinematics, _fields) are imported inside the functions that need them: solver.py imports this
module, so importing it back at module level would close a cycle.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import utils
from .geometry import phi0_field, to_phys
from .utils import DTYPE, problem_of

def _grid_eval(net, cfg, ng=500, chunk=45000):
    """render grid: ng=500 => 0.2l/pixel at l=0.01 (~1.3 px per 0.26l feature) -- the
       A/B-verified smooth point (kills the aliased 'jagged crack edge'). Chunked so the
       autodiff eval of ng^2 points stays within GPU memory (no OOM ceiling on ng)."""
    from .solver import kinematics
    xs = torch.linspace(0, 1, ng, device=utils.DEVICE)
    GX, GY = torch.meshgrid(xs, xs, indexing="xy")
    xf, yf = GX.reshape(-1, 1), GY.reshape(-1, 1)
    exys, phis = [], []
    for i in range(0, xf.shape[0], chunk):
        exx, eyy, exy, phi, gp2, _, _ = kinematics(net, xf[i:i + chunk].clone(),
                                                   yf[i:i + chunk].clone(), cfg, cfg.u_delta_nd)
        exys.append(exy.detach()); phis.append(phi.detach())
    to = lambda ts: torch.cat(ts).reshape(ng, ng).cpu().numpy()
    with torch.no_grad():
        XP, YP = to_phys(xf, yf, cfg)                # physical axes (curvilinear under --geo)
    XP = XP.reshape(ng, ng).cpu().numpy(); YP = YP.reshape(ng, ng).cpu().numpy()
    out = dict(exy=to(exys), phi=to(phis))
    blank = problem_of(cfg).render_mask(cfg, XP, YP)     # holes / cut-outs render as white
    if blank is not None:
        out = {k: np.where(blank, np.nan, v) for k, v in out.items()}
    return XP, YP, out

def _crack_phys(cfg):
    """The pre-existing cracks as PHYSICAL segments for the white guide lines on phi plots --
       EVERY segment of cfg.cracks, at its true inclination. (Drawing the legacy nx0/nx1/ny
       instead gave the multi-crack cases one horizontal line through the bbox of crack #1
       and nothing for the rest -- a guide line that matched no crack at all.)
       cfg.cracks are PHYSICAL coords, exactly what dist_to_notch measures against, so they are
       drawn directly. (Mapping them as parametric xi/eta through to_phys extrapolates the geo
       map far outside [0,1] for a curved patch whose physical crack coords are not in [0,1]
       => garbage; identity/distort were unaffected only by coincidence.)"""
    return [((ax, bx), (ay, by)) for (ax, ay, bx, by) in (getattr(cfg, "cracks", None) or [])]

def plot_phi(net, cfg, save_dir, tag, ud, sample_pts=None):
    """Always writes a CLEAN phi map (phi_{tag}.png). If sample_pts is given, ALSO writes
       phi_{tag}_pts.png with the integration points overlaid (the project two phi plots,
       one with points one without). ng=500 + shading='gouraud' = the smooth (de-aliased)
       render -- the 'jagged crack edge' was pcolormesh flat/nearest cells, not resolution."""
    GX, GY, F = _grid_eval(net, cfg)
    seeds = _crack_phys(cfg)
    pts_phys = None
    if sample_pts is not None:
        if getattr(cfg, "geomap", None) is not None:
            with torch.no_grad():
                sx, sy = cfg.geomap(torch.as_tensor(sample_pts[:, 0:1], dtype=DTYPE, device=utils.DEVICE),
                                    torch.as_tensor(sample_pts[:, 1:2], dtype=DTYPE, device=utils.DEVICE))
            pts_phys = np.concatenate([sx.cpu().numpy(), sy.cpu().numpy()], axis=1)
        else:
            pts_phys = sample_pts
    for with_pts in ([False, True] if pts_phys is not None else [False]):
        plt.figure(figsize=(7.2, 6))
        m = plt.pcolormesh(GX, GY, F["phi"], cmap="magma", shading="gouraud", vmin=0, vmax=1)
        plt.colorbar(m, label="phi")
        if with_pts:
            plt.scatter(pts_phys[:, 0], pts_phys[:, 1], s=1.0, c="c", alpha=0.25, linewidths=0)
        for sx, sy in seeds:                            # no guide line without a pre-existing crack
            plt.plot(sx, sy, "w--", lw=0.8, alpha=0.6)
        problem_of(cfg).overlay(cfg)
        extra = "  + integration points" if with_pts else ""
        plt.title(f"phi  (delta={cfg.U_ref*ud:.3e})  msenc levels={list(cfg.levels)}{extra}")
        plt.xlabel("x"); plt.ylabel("y")
        plt.gca().set_aspect("equal", "box")
        fn = f"phi_{tag}_pts.png" if with_pts else f"phi_{tag}.png"
        plt.savefig(os.path.join(save_dir, fn), dpi=140, bbox_inches="tight"); plt.close()

def plot_FD_compare(FD, cfg, save_dir, fem_path):
    """DEM vs FEM overlay -- lives at the RUN ROOT, refreshed every step."""
    FD = np.array(FD)
    plt.figure(figsize=(6.5, 5))
    if os.path.exists(fem_path):
        fem = np.loadtxt(fem_path)
        plt.plot(fem[:, 0], fem[:, 1], "-", lw=1.3, alpha=0.55,
                 label=f"{os.path.basename(fem_path)} (FEM)")
    if len(FD):
        plt.plot(FD[:, 0], FD[:, 1], "o-", ms=4, lw=1.6, label="DEM Stage B-MSENC")
    plt.xlabel("Displacement delta (mm)"); plt.ylabel("Force (N)")
    plt.title(f"Stage B-MSENC vs FEM (l={cfg.l})")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "FD_vs_FEM.png"), dpi=150, bbox_inches="tight"); plt.close()

def plot_FD_solo(FD, cfg, save_dir):
    """DEM-only F-delta -- saved INTO the per-step folder on plot steps."""
    FD = np.array(FD)
    plt.figure(figsize=(6.5, 5))
    if len(FD):
        plt.plot(FD[:, 0], FD[:, 1], "o-", ms=4, lw=1.6, color="tab:green")
    plt.xlabel("Displacement delta (mm)"); plt.ylabel("Force (N)")
    plt.title(f"Stage B-MSENC F-delta (l={cfg.l})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "FD_solo.png"), dpi=150, bbox_inches="tight"); plt.close()

def plot_fields(net, cfg, save_dir, tag, ud, ng=500, chunk=45000):
    """displacement u, v and shear strain exy maps with the phi=0.5 crack contour overlaid.
       ng=500 + chunked eval: the A/B-verified smooth rendering point."""
    from .solver import _fields, kinematics
    xs = torch.linspace(0, 1, ng, device=utils.DEVICE)
    GX, GY = torch.meshgrid(xs, xs, indexing="xy")
    xf, yf = GX.reshape(-1, 1), GY.reshape(-1, 1)
    us, vs, exys, phis = [], [], [], []
    for i in range(0, xf.shape[0], chunk):
        xc, yc = xf[i:i + chunk].clone(), yf[i:i + chunk].clone()
        exx, eyy, exy, phi, gp2, _, _ = kinematics(net, xc, yc, cfg, ud)
        with torch.no_grad():
            u, v, _, _ = _fields(net, xf[i:i + chunk].clone(), yf[i:i + chunk].clone(),
                                 cfg, ud)
        us.append(u.detach()); vs.append(v.detach())
        exys.append(exy.detach()); phis.append(phi.detach())
    to = lambda ts: torch.cat(ts).reshape(ng, ng).cpu().numpy()
    U, V, EXY, PHI = to(us), to(vs), to(exys), to(phis)
    with torch.no_grad():
        XP, YP = to_phys(xf, yf, cfg)
    X = XP.reshape(ng, ng).cpu().numpy(); Y = YP.reshape(ng, ng).cpu().numpy()
    blank = problem_of(cfg).render_mask(cfg, X, Y)      # holes / cut-outs render as white
    if blank is not None:
        U, V, EXY, PHI = (np.where(blank, np.nan, A) for A in (U, V, EXY, PHI))
    # FIXED project palette: phi=magma, u/v=viridis, T=coolwarm,
    # exy=viridis with SYMMETRIC +-limits (zero shear = mid-green, sign = dark/bright side).
    # phi=0.5 contour overlay on exy ONLY.
    for F, lab, fn, cmap, sym, ctr in [(U, "u (mm)", f"field_u_{tag}.png", "viridis", False, False),
                                       (V, "v (mm)", f"field_v_{tag}.png", "viridis", False, False),
                                       (EXY, "exy", f"field_exy_{tag}.png", "viridis", True, True)]:
        plt.figure(figsize=(7, 6))
        if sym:
            vv = np.nanpercentile(np.abs(F), 99.5); kw = dict(vmin=-vv, vmax=vv)
        else:
            kw = dict(vmin=np.nanpercentile(F, 0.5), vmax=np.nanpercentile(F, 99.5))
        m = plt.pcolormesh(X, Y, F, cmap=cmap, shading="gouraud", **kw)
        plt.colorbar(m, label=lab)
        if ctr:
            # phi=0.5 crack contour on exy ONLY; u/v stay clean (no phase-band trace)
            plt.contour(X, Y, PHI, levels=[0.5], colors="w", linewidths=0.7)
            plt.title(f"{lab}  (delta={cfg.U_ref*ud:.3e})  white = phi=0.5")
        else:
            plt.title(f"{lab}  (delta={cfg.U_ref*ud:.3e})")
        problem_of(cfg).overlay(cfg)
        plt.xlabel("x"); plt.ylabel("y")
        plt.gca().set_aspect("equal", "box")
        plt.savefig(os.path.join(save_dir, fn), dpi=140, bbox_inches="tight"); plt.close()

def plot_linecut(net, cfg, save_dir, tag, ud):
    """u across the crack at a cracked vs intact x -- is the step there and how sharp?"""
    from .solver import _fields
    yy = torch.linspace(0.30, 0.70, 500, device=utils.DEVICE).reshape(-1, 1)
    plt.figure(figsize=(6.5, 5))
    for xv, cc in [(0.25, "tab:red"), (0.55, "tab:orange"), (0.85, "tab:blue")]:
        xx = torch.full_like(yy, xv)
        with torch.no_grad():
            uu = _fields(net, xx, yy, cfg, ud)[0].cpu().numpy()
        plt.plot(yy.cpu().numpy(), uu, color=cc, label=f"x={xv}")
    plt.axvline(0.5, ls=":", c="k", alpha=0.5, label="notch line y=0.5")
    plt.xlabel("y"); plt.ylabel("u(x,y)")
    plt.title(f"u linecut across the crack ({tag})")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"linecut_{tag}.png"), dpi=150, bbox_inches="tight"); plt.close()

def plot_geometry(cfg, save_dir, ng=400):
    """ONE IGA patch figure (once per run -- geometry is fixed): the physical images of the
       KNOT lines (isoparametric 'element' boundaries) + the CONTROL NET (points + polygon).
       When |det J| VARIES (distorted / curved patches) it is drawn as a SMOOTH continuous
       gouraud field behind the mesh; when |det J| is essentially constant (the identity
       square, detJ==1) the flat fill is skipped (it would just be a solid color block) and
       the mesh sits on a clean white background. No-op under --geo none."""
    gm = getattr(cfg, "geomap", None)
    if gm is None:
        return
    xs = torch.linspace(0, 1, ng, device=utils.DEVICE)
    GXi, GEta = torch.meshgrid(xs, xs, indexing="xy")
    xi, eta = GXi.reshape(-1, 1), GEta.reshape(-1, 1)
    with torch.no_grad():
        xp, yp = gm(xi, eta)
    _, _, _, _, det = gm.jac(xi, eta)
    X = xp.reshape(ng, ng).cpu().numpy(); Y = yp.reshape(ng, ng).cpu().numpy()
    D = det.reshape(ng, ng).cpu().numpy()
    ku = np.unique(gm.ku.cpu().numpy()); kv = np.unique(gm.kv.cpu().numpy())
    cx = gm.cx.cpu().numpy(); cy = gm.cy.cpu().numpy()
    t = torch.linspace(0, 1, 200, device=utils.DEVICE).reshape(-1, 1)
    varies = (D.max() - D.min()) / (abs(D.mean()) + 1e-30) > 1e-3

    fig, ax = plt.subplots(figsize=(7.4, 6))
    if varies:
        m = ax.pcolormesh(X, Y, D, cmap="twilight_shifted", shading="gouraud")   # continuous
        plt.colorbar(m, ax=ax, label="|det J|  (param->physical area factor)")
        detstr = f"detJ in [{D.min():.3f}, {D.max():.3f}]"
    else:
        detstr = f"detJ = {D.mean():.3f} (uniform)"
    if not gm.is_identity:                                   # knot lines only for non-square patches
        with torch.no_grad():
            for kval in ku:
                px, py = gm(torch.full_like(t, float(kval)), t)
                ax.plot(px.cpu().numpy(), py.cpu().numpy(), "-", c="0.15", lw=0.9)
            for kval in kv:
                px, py = gm(t, torch.full_like(t, float(kval)))
                ax.plot(px.cpu().numpy(), py.cpu().numpy(), "-", c="0.15", lw=0.9)
    for i in range(cx.shape[0]):                             # control net
        ax.plot(cx[i, :], cy[i, :], "-", c="tab:red", lw=0.6, alpha=0.6)
    for j in range(cx.shape[1]):
        ax.plot(cx[:, j], cy[:, j], "-", c="tab:red", lw=0.6, alpha=0.6)
    ax.plot(cx.ravel(), cy.ravel(), "o", c="tab:red", ms=3.5, label="control points")
    ax.set_title(f"IGA patch: knot lines + control net  (geo={cfg.geo}, deg {gm.pu}x{gm.pv}, {detstr})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal", "box"); ax.legend(fontsize=8)
    fig.savefig(os.path.join(save_dir, "geometry_patch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_geometry_bc(cfg, save_dir):
    """PHYSICAL-space geometry + iso-parametric mesh + control net (Si Fig 24(a) analogue). The
       `geo=='ring'` branch colour-codes the ring BC (pull/clamp/free/symmetry) + notch; any other
       patch draws the generic boundary + NURBS control net. No-op under --geo none."""
    gm = getattr(cfg, "geomap", None)
    if gm is None:
        return

    def P(xa, ea):
        with torch.no_grad():
            X, Y = gm(torch.as_tensor(xa, dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1),
                      torch.as_tensor(ea, dtype=DTYPE, device=utils.DEVICE).reshape(-1, 1))
        return X.cpu().numpy().ravel(), Y.cpu().numpy().ravel()

    t = np.linspace(0, 1, 240)
    fig, ax = plt.subplots(figsize=(6.4, 7))
    if not gm.is_identity:                                   # iso-mesh only for non-square patches
        for e in np.linspace(0, 1, 19):                      # (a plain square needs no iso lines)
            x, y = P(t, np.full_like(t, e)); ax.plot(x, y, "0.8", lw=0.5)
        for xi in np.linspace(0, 1, 10):
            x, y = P(np.full_like(t, xi), t); ax.plot(x, y, "0.8", lw=0.5)
    if cfg.geo == "ring":
        eu = np.linspace(0.5, 1, 120); x, y = P(np.ones_like(eu), eu)
        ax.plot(x, y, "r", lw=3, label="outer upper: PULL  v=+u_imp")
        el = np.linspace(0, 0.5, 120); x, y = P(np.ones_like(el), el)
        ax.plot(x, y, "b", lw=3, label="outer lower: CLAMP  u=v=0")
        x, y = P(np.zeros_like(t), t); ax.plot(x, y, "g", lw=2, label="inner: free")
        for e in (0.0, 1.0):
            x, y = P(t, np.full_like(t, e)); ax.plot(x, y, "m--", lw=1.4)
        ax.plot([], [], "m--", lw=1.4, label="symmetry x=0  (u_x=0)")
        ax.plot([cfg.nx0, cfg.nx1], [cfg.ny, cfg.ny], "k", lw=4, label="notch")
        ax.plot(cfg.nx0, cfg.ny, "ko", ms=5)
        ax.annotate(f"tip r={cfg.nx0:g}", (cfg.nx0, cfg.ny),
                    (cfg.nx0 - 0.28 * cfg.nx1, cfg.ny + 0.12 * cfg.nx1), fontsize=8)
        cx = gm.cx.cpu().numpy(); cy = gm.cy.cpu().numpy()     # NURBS control net
        for i in range(cx.shape[0]):
            ax.plot(cx[i, :], cy[i, :], "-", c="0.45", lw=0.7, alpha=0.7, zorder=1)
        for j in range(cx.shape[1]):
            ax.plot(cx[:, j], cy[:, j], "-", c="0.45", lw=0.7, alpha=0.7, zorder=1)
        ax.plot(cx.ravel(), cy.ravel(), "s", c="darkorange", ms=5,
                label="control points", zorder=5)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    else:
        for xi in (0.0, 1.0):
            x, y = P(np.full_like(t, xi), t); ax.plot(x, y, "0.15", lw=1.6)
        for e in (0.0, 1.0):
            x, y = P(t, np.full_like(t, e)); ax.plot(x, y, "0.15", lw=1.6)
        cx = gm.cx.cpu().numpy(); cy = gm.cy.cpu().numpy()     # NURBS control net (kept for all)
        for i in range(cx.shape[0]):
            ax.plot(cx[i, :], cy[i, :], "-", c="0.45", lw=0.7, alpha=0.7, zorder=1)
        for j in range(cx.shape[1]):
            ax.plot(cx[:, j], cy[:, j], "-", c="0.45", lw=0.7, alpha=0.7, zorder=1)
        ax.plot(cx.ravel(), cy.ravel(), "s", c="darkorange", ms=4, label="control points", zorder=5)
        ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", "box")
    ax.set_title(f"geometry + boundary conditions  (geo={cfg.geo})")
    ax.set_xlabel("x (physical)"); ax.set_ylabel("y (physical)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "geometry_bc.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_parametric(cfg, save_dir, ng=300):
    """PARAMETRIC (reference) domain [0,1]^2 -- the SQUARE computational domain the network,
       encoding, sampler and quadrature actually live on (the physical shape is the geo map's
       doing). Uniform iso-grid 'mesh' + the notch band (phi0 heatmap) + (ring) BC edge labels."""
    xs = torch.linspace(0, 1, ng)
    GX, GY = torch.meshgrid(xs, xs, indexing="xy")
    with torch.no_grad():
        p0 = phi0_field(GX.reshape(-1, 1).to(utils.DEVICE), GY.reshape(-1, 1).to(utils.DEVICE), cfg)
    p0 = p0.reshape(ng, ng).cpu().numpy()
    from matplotlib.colors import LinearSegmentedColormap
    wcmap = LinearSegmentedColormap.from_list("phi0_white", ["white", "#8000a0", "#1a0030"])
    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.set_facecolor("white")
    ax.pcolormesh(GX.numpy(), GY.numpy(), p0, shading="gouraud", cmap=wcmap, vmin=0, vmax=1)
    for g in np.linspace(0, 1, 13):                          # uniform iso-grid 'mesh'
        ax.axvline(g, c="0.75", lw=0.5); ax.axhline(g, c="0.75", lw=0.5)
    if cfg.geo == "ring":
        ax.plot([1, 1], [0.5, 1], "r", lw=5); ax.plot([1, 1], [0, 0.5], "b", lw=5)
        ax.plot([0, 0], [0, 1], "g", lw=5)
        ax.plot([0, 1], [0, 0], "m", lw=4); ax.plot([0, 1], [1, 1], "m", lw=4)
        ax.text(1.04, 0.75, "PULL", color="r", rotation=270, va="center", fontsize=9)
        ax.text(1.04, 0.25, "CLAMP", color="b", rotation=270, va="center", fontsize=9)
        ax.text(-0.035, 0.5, "inner free", color="g", rotation=90, va="center", ha="right", fontsize=9)
        ax.text(0.5, 0.955, "symmetry x=0 (u_x=0)", color="m", ha="center", va="top", fontsize=8)
        ax.text(0.5, 0.045, "symmetry x=0 (u_x=0)", color="m", ha="center", va="bottom", fontsize=8)
        ax.text(0.86, 0.53, "notch (phi0)", color="#8000a0", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("xi  (radial: 0 = inner, 1 = outer)")
        ax.set_ylabel("eta  (angular: 0.5 = notch axis)")
    else:
        ax.set_xlabel("xi"); ax.set_ylabel("eta")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal", "box")
    ax.set_title(f"parametric domain [0,1]$^2$  (geo={cfg.geo})", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "geometry_parametric.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
