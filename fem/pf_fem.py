"""Local phase-field fracture FEM (scikit-fem) -- the finite-element reference generator and
independent cross-check for the mesh-free solver. Runs locally, no FEniCS/dolfin needed.

Purpose: generate references in the SAME units as the DEM runs (E=340e3, nu=0.22, Gc=0.04247)
for every example, including the branching and coalescence cases that have no published external
reference, and cross-check the DEM at matching l. Deliberately MODULAR (elasticity /
scalar-field / staggered driver) so that a thermo-mechanical extension can reuse it: a
temperature field is another scalar diffusion assembled by the same machinery, and thermo-elastic
coupling adds a thermal-strain term to the elasticity right-hand side.

Model: plane strain, AT2 (w=phi^2, c_w=2), degradation g=(1-phi)^2+g_floor. Splits:
  iso    -- no split: u degrades the FULL isotropic energy, driving psi+ = full energy  => BRANCHES
  hybrid -- Ambati: u degrades the full energy (same u-solve as iso), but driving psi+ = SPECTRAL
            tension energy (Miehe) => single band, matches Tangella.
Irreversibility via the history field H = max_tau psi+ (Miehe). Pre-crack via phi=1 Dirichlet on
the notch segment(s). Staggered u<->phi per load step. Run:
  python pf_fem.py --example bifurcation --split iso --l 0.01 --refine 8 --out runs_fem/bifurc

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from skfem import (MeshQuad, Basis, ElementVector, ElementQuad1, BilinearForm,
                   LinearForm, asm, solve, condense)
from skfem.helpers import grad, sym_grad, ddot, dot, div

# ---- linear solver: prefer MKL PARDISO (multithreaded, honours MKL_NUM_THREADS) over scipy
# SuperLU (serial). PARDISO makes refine=9 (~0.5M dofs) x hundreds of steps tractable on a
# multicore machine; without pypardiso installed this falls back to scipy transparently. Both
# systems (degraded elasticity, AT2 phi) are symmetric, which suits PARDISO.
def _make_linsolve():
    try:
        import pypardiso
        def _f(A, b):
            return pypardiso.spsolve(A.tocsr(), b)
        _f.name = "pardiso"
        return _f
    except Exception:
        from skfem.utils import solver_direct_scipy
        f = solver_direct_scipy()
        try:
            f.name = "scipy"
        except Exception:
            pass
        return f

LINSOLVE = _make_linsolve()

# ------------------------------- config -------------------------------
SEN_NOTCH = [(0.0, 0.5, 0.5, 0.5)]                         # SEN: left-half horizontal notch
COAL_CRACKS = [(0.25, 0.35, 0.35, 0.45),                  # Manav 4.6 en-echelon (our-units domain)
               (0.45, 0.45, 0.55, 0.55),
               (0.65, 0.55, 0.75, 0.65)]

EXAMPLES = {
    # example      loading    split(default)  cracks        delta_max
    "shear":       ("shear",   "hybrid", SEN_NOTCH,   2.7e-3),
    "tension":     ("tension", "hybrid", SEN_NOTCH,   8e-4),
    "bifurcation": ("shear",   "iso",    SEN_NOTCH,   2.7e-3),
    "coalescence": ("tension", "hybrid", COAL_CRACKS, 2.7e-3),
}

class Cfg:
    def __init__(self, a):
        self.E, self.nu, self.Gc, self.l = a.E, a.nu, a.Gc, a.l
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))   # plane strain
        self.mu = self.E / (2 * (1 + self.nu))
        self.g_floor = 1e-6
        loading, split, cracks, dmax = EXAMPLES[a.example]
        self.example = a.example
        self.loading = a.loading or loading
        self.split = a.split or split
        self.cracks = cracks
        self.delta_max = a.delta_max if a.delta_max > 0 else dmax
        self.refine = a.refine
        self.nsteps = a.nsteps
        self.stag_tol, self.stag_max = a.stag_tol, a.stag_max
        self.asm = a.asm
        self.snap = getattr(a, "snap", "")
        self.out = a.out

# ------------------------- geometry / notch ---------------------------
def dist_to_segments(px, py, cracks):
    """min point-to-segment distance over all crack segments (vectorized over points)."""
    d = np.full(px.shape, np.inf)
    for ax, ay, bx, by in cracks:
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy + 1e-30
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0.0, 1.0)
        cx, cy = ax + t * vx, ay + t * vy
        d = np.minimum(d, np.sqrt((px - cx) ** 2 + (py - cy) ** 2))
    return d

def notch_dofs(ip, cfg, band):
    """phi=1 Dirichlet dofs: scalar dofs whose node lies within `band` of any crack segment."""
    return ip.get_dofs(lambda x: dist_to_segments(x[0], x[1], cfg.cracks) < band)

# ------------------------------- forms --------------------------------
def make_forms(cfg):
    lam, mu, gf, Gc, l = cfg.lam, cfg.mu, cfg.g_floor, cfg.Gc, cfg.l

    @BilinearForm
    def Kdeg(u, v, w):                                     # degraded elasticity (full energy)
        g = (1.0 - w["phi"]) ** 2 + gf
        return g * (lam * div(u) * div(v) + 2.0 * mu * ddot(sym_grad(u), sym_grad(v)))

    @BilinearForm
    def Aphi(p, q, w):                                     # AT2 phi-system LHS (dot, scalar grad!)
        return Gc * l * dot(grad(p), grad(q)) + (Gc / l + 2.0 * w["H"]) * p * q

    @LinearForm
    def bphi(q, w):                                        # AT2 phi-system RHS
        return 2.0 * w["H"] * q

    return Kdeg, Aphi, bphi

# ---- fast reassembly (cfg.asm == "fast") --------------------------------------------------------
# The staggered loop reassembles K(phi) and A(H) every iteration; at fine meshes skfem's asm of the
# elasticity operator dominates (~90% of the per-iteration cost). But the integrand's geometry+material
# part (B^T C B x |detJ|) is CONSTANT -- only g(phi) / H change. So precompute the per-(elem,qp) kernels
# ONCE, then each iteration is just a weighted sum + scatter. Reproduces asm(Kdeg)/asm(Aphi)/asm(bphi)
# to fp precision (validated), ~18x faster on the elasticity assembly. The same "assemble-once, rescale"
# pattern extends to any further scalar field (a temperature, say) assembled by the same machinery.
def _coo_template(basis, NB, NE):
    """Fixed (row,col) scatter for a basis: returns (csr template, inverse index into the unique
    entries, n_unique) so per-iteration assembly is bincount(inv, data) into a prebuilt sparsity."""
    ed, N = basis.element_dofs, basis.N
    rows = np.broadcast_to(ed[:, None, :], (NB, NB, NE)).ravel()
    cols = np.broadcast_to(ed[None, :, :], (NB, NB, NE)).ravel()
    keys = rows.astype(np.int64) * N + cols.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    templ = coo_matrix((np.zeros(len(uniq)), (uniq // N, uniq % N)), shape=(N, N)).tocsr()
    return templ, inv, len(uniq)

class FastKdeg:
    """K(phi) = int g(phi) [lam div u div v + 2mu eps(u):eps(v)]. The bracket x |detJ| is precomputed
    per (elem, qp); __call__(phi_qp) weights it by g(phi_qp) = (1-phi_qp)^2 + g_floor and scatters."""
    def __init__(self, iu, cfg):
        self.gf = cfg.g_floor
        NB, NE = iu.Nbfun, iu.nelems
        dx = iu.dx
        grads = [np.asarray(iu.basis[i][0].grad) for i in range(NB)]               # each (2,2,NE,NQ)
        div_i = np.stack([g[0, 0] + g[1, 1] for g in grads], 0)                    # (NB,NE,NQ)
        eps_i = np.stack([0.5 * (g + g.transpose(1, 0, 2, 3)) for g in grads], 0)  # (NB,2,2,NE,NQ)
        divdiv = np.einsum('ieq,jeq->ijeq', div_i, div_i)
        epseps = np.einsum('iabeq,jabeq->ijeq', eps_i, eps_i)
        self.Vqp = dx[None, None] * (cfg.lam * divdiv + 2.0 * cfg.mu * epseps)     # (NB,NB,NE,NQ)
        self._T, self._inv, self._n = _coo_template(iu, NB, NE)

    def __call__(self, phi_qp):
        g = (1.0 - phi_qp) ** 2 + self.gf
        data = np.einsum('eq,ijeq->ije', g, self.Vqp).ravel()
        K = self._T.copy()
        K.data[:] = np.bincount(self._inv, weights=data, minlength=self._n)
        return K

class FastPhi:
    """AT2 phi-system. A(H) = Gc*l (grad,grad) + (Gc/l + 2H)(.,.): stiffness + Gc/l mass are constant
    (precomputed into _const); only the 2H mass and RHS b(H) = int 2H q change with the history field."""
    def __init__(self, ip, cfg):
        Gc, l = cfg.Gc, cfg.l
        NB, NE = ip.Nbfun, ip.nelems
        dx = ip.dx
        p = np.stack([np.asarray(ip.basis[i][0].value) for i in range(NB)], 0)    # (NB,NE,NQ)
        gp = np.stack([np.asarray(ip.basis[i][0].grad) for i in range(NB)], 0)    # (NB,2,NE,NQ)
        self.Wqp = dx[None, None] * np.einsum('ieq,jeq->ijeq', p, p)              # 2H mass kernel
        lap = np.einsum('ideq,jdeq->ijeq', gp, gp)
        Cqp = dx[None, None] * (Gc * l * lap + (Gc / l) * np.einsum('ieq,jeq->ijeq', p, p))
        self._T, self._inv, self._n = _coo_template(ip, NB, NE)
        self._const = np.einsum('ijeq->ije', Cqp).ravel()
        self._Bqp = dx[None] * p                                                  # (NB,NE,NQ) RHS kernel
        self._ed, self._N = ip.element_dofs, ip.N

    def A(self, H):
        data = self._const + np.einsum('eq,ijeq->ije', 2.0 * H, self.Wqp).ravel()
        A = self._T.copy()
        A.data[:] = np.bincount(self._inv, weights=data, minlength=self._n)
        return A

    def b(self, H):
        be = np.einsum('eq,ieq->ie', 2.0 * H, self._Bqp)
        b = np.zeros(self._N)
        np.add.at(b, self._ed, be)
        return b

def psi_plus(iu, u, cfg):
    """tensile driving energy density at quad points. iso = full isotropic energy; hybrid =
       Miehe SPECTRAL tension (eigen-split of the strain)."""
    du = iu.interpolate(u).grad
    exx, eyy, exy = du[0, 0], du[1, 1], 0.5 * (du[0, 1] + du[1, 0])
    tr = exx + eyy
    if cfg.split == "iso":
        return 0.5 * cfg.lam * tr ** 2 + cfg.mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)
    # hybrid: spectral tension. eigenvalues of the 2x2 strain
    rad = np.sqrt(((exx - eyy) * 0.5) ** 2 + exy ** 2)
    e1, e2 = 0.5 * tr + rad, 0.5 * tr - rad
    pos = lambda z: np.maximum(z, 0.0)
    return 0.5 * cfg.lam * pos(tr) ** 2 + cfg.mu * (pos(e1) ** 2 + pos(e2) ** 2)

# ------------------------------- driver -------------------------------
def bc_dofs(iu, cfg):
    bot = iu.get_dofs(lambda x: x[1] < 1e-9)
    top = iu.get_dofs(lambda x: x[1] > 1 - 1e-9)
    return bot, top

def apply_load(iu, bot, top, delta, cfg):
    """returns (x prescribed vector, D constrained dofs, loaded-dof indices for the reaction)."""
    x = iu.zeros()
    if cfg.loading == "shear":                            # bottom clamped, top u=delta v=0
        x[top.all("u^1")] = delta
        D = np.concatenate([bot.all(), top.all("u^1"), top.all("u^2")])
        loaded = top.all("u^1")
    else:                                                 # tension: bottom clamped, top v=delta u=0
        x[top.all("u^2")] = delta
        D = np.concatenate([bot.all(), top.all("u^2"), top.all("u^1")])
        loaded = top.all("u^2")
    return x, D, loaded

def run(cfg):
    os.makedirs(cfg.out, exist_ok=True)
    m = MeshQuad().refined(cfg.refine)
    h = 1.0 / (2 ** cfg.refine)
    iu = Basis(m, ElementVector(ElementQuad1()), intorder=2)
    ip = Basis(m, ElementQuad1(), intorder=2)
    if cfg.asm == "fast":                                 # precomputed-kernel reassembly (default)
        fk, fp = FastKdeg(iu, cfg), FastPhi(ip, cfg)
        asmK = lambda phi: fk(ip.interpolate(phi).value)
        asmA, asmb = fp.A, fp.b
    else:                                                 # slow = skfem asm each iter (reference)
        Kdeg, Aphi, bphi = make_forms(cfg)
        asmK = lambda phi: asm(Kdeg, iu, phi=ip.interpolate(phi))
        asmA = lambda H: asm(Aphi, ip, H=H)
        asmb = lambda H: asm(bphi, ip, H=H)
    notch = notch_dofs(ip, cfg, band=1.5 * h)
    bot, top = bc_dofs(iu, cfg)
    print(f"[pf_fem] {cfg.example}: split={cfg.split} loading={cfg.loading} l={cfg.l} "
          f"h={h:.4f} (h/l={h/cfg.l:.2f})  u-dofs={iu.N} phi-dofs={ip.N} notch-dofs={len(notch.all())} "
          f"solver={getattr(LINSOLVE, 'name', '?')} asm={cfg.asm}")

    phi = ip.zeros(); phi[notch.all()] = 1.0
    H = np.zeros((m.nelements, 4))
    FD = []
    deltas = np.linspace(0.0, cfg.delta_max, cfg.nsteps + 1)[1:]
    inc = deltas[0]
    snap = [float(s) for s in
            str(getattr(cfg, "snap", "")).split(",") if s.strip()]
    snap_done = set()
    for k, delta in enumerate(deltas):
        x, D, loaded = apply_load(iu, bot, top, delta, cfg)
        for it in range(cfg.stag_max):
            Km = asmK(phi)
            u = solve(*condense(Km, iu.zeros(), x=x, D=D), solver=LINSOLVE)
            H = np.maximum(H, psi_plus(iu, u, cfg))       # irreversible history
            phi_new = solve(*condense(asmA(H), asmb(H),
                                      x=ip.ones(), D=notch.all()), solver=LINSOLVE)
            dchg = np.linalg.norm(phi_new - phi) / (np.linalg.norm(phi_new) + 1e-30)
            phi = phi_new
            if dchg < cfg.stag_tol:
                break
        F = (asmK(phi) @ u)[loaded].sum()
        FD.append([delta, F])
        for i, t in enumerate(snap):
            if i not in snap_done and abs(delta - t) <= 0.5 * inc:
                tag = f"{t:.4g}"
                np.save(os.path.join(cfg.out, f"phi_d{tag}.npy"), phi)
                np.save(os.path.join(cfg.out, f"u_d{tag}.npy"),
                        u[iu.nodal_dofs])
                snap_done.add(i)
                print(f"  [snap] fields dumped at delta={delta:.4e} "
                      f"(target {t:.4g})")
        if k % max(1, cfg.nsteps // 20) == 0 or k == len(deltas) - 1:
            print(f"  step {k:3d}/{len(deltas)} delta={delta:.3e}  F={F:8.3f}  "
                  f"stag_it={it}  phi_max={phi.max():.3f}  cracked={(phi>0.9).sum()}")
    FD = np.array(FD)
    np.savetxt(os.path.join(cfg.out, "FD.txt"), FD, header="delta(mm)  F(N)")
    _plot_phi(m, ip, phi, cfg, os.path.join(cfg.out, "phi_final.png"))
    _plot_FD(FD, cfg, os.path.join(cfg.out, "FD.png"))
    np.save(os.path.join(cfg.out, "phi_final.npy"), phi)
    np.save(os.path.join(cfg.out, "u_final.npy"), u[iu.nodal_dofs])
    np.save(os.path.join(cfg.out, "nodes.npy"), m.p)
    print(f"[pf_fem] done -> {cfg.out}")
    return FD, phi, m

def _plot_phi(m, ip, phi, cfg, path):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    tp = ax.tripcolor(m.p[0], m.p[1], phi, shading="gouraud", cmap="magma", vmin=0, vmax=1)
    fig.colorbar(tp, label="phi")
    ax.set_aspect("equal"); ax.set_title(f"FEM phi  {cfg.example} ({cfg.split}, l={cfg.l})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def _plot_FD(FD, cfg, path):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(FD[:, 0], FD[:, 1], "-", lw=1.6)
    ax.set_xlabel("displacement delta (mm)"); ax.set_ylabel("reaction F (N)")
    ax.set_title(f"FEM F-delta  {cfg.example} ({cfg.split}, l={cfg.l})")
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="bifurcation", choices=list(EXAMPLES))
    ap.add_argument("--split", default="", help="iso|hybrid (default per example)")
    ap.add_argument("--loading", default="", help="shear|tension (default per example)")
    ap.add_argument("--E", type=float, default=340e3)
    ap.add_argument("--nu", type=float, default=0.22)
    ap.add_argument("--Gc", type=float, default=0.04247)
    ap.add_argument("--l", type=float, default=0.01)
    ap.add_argument("--refine", type=int, default=8, help="MeshQuad refinements (8 => 256x256)")
    ap.add_argument("--nsteps", type=int, default=120)
    ap.add_argument("--delta_max", type=float, default=0.0, help="0 => per-example default")
    ap.add_argument("--stag_tol", type=float, default=1e-3)
    ap.add_argument("--stag_max", type=int, default=50)
    ap.add_argument("--asm", default="fast", choices=["fast", "slow"],
                    help="fast = precomputed-kernel reassembly (default, ~18x on elasticity); "
                         "slow = skfem asm each iteration (reference path for cross-checking)")
    ap.add_argument("--snap", default="",
                    help="comma list of delta values (mm); at the nearest "
                         "load step the phi and displacement fields are "
                         "dumped as phi_d<delta>.npy / u_d<delta>.npy "
                         "(for the DEM-vs-FEM propagation field figures)")
    ap.add_argument("--out", default="runs_fem/pf")
    run(Cfg(ap.parse_args()))

if __name__ == "__main__":
    main()
