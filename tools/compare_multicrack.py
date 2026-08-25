#!/usr/bin/env python
"""Score a finished benchmark run against the Hamdi & Lejeune ground truth.

    python NEW_BENCHMARK/compare.py --run runs/bench_tension_100192

Reports, at every load step that has a checkpoint:
  * the Dice score of our phase field against theirs at the matching displacement, using THEIR
    metric (`pfm_bench/src/metrics/Dice.py`: threshold 0.5 on both sides, 2|p&g|/(|p|+|g|));
  * which seeded cracks are active and which are dormant, on both sides - the discriminating
    prediction, since their reference leaves several cracks at their initial length;
  * the reaction force against theirs (for bi-axial tension our single dPi/ddelta is the
    work-conjugate of one delta driving two edges, so it is compared with |Fx| + |Fy|).

Their arrays are indexed [x][y], so every picture uses pcolormesh with the real coordinates.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark_problem                                    # noqa: E402,F401  (registers it)
import pfpiml as sb                                         # noqa: E402
from pfpiml import cfg_from_json, utils                     # noqa: E402

def dice(p, g, tp=0.5, tg=0.5):
    """Their metric, verbatim."""
    p, g = p > tp, g > tg
    denom = p.sum() + g.sum()
    return float(2.0 * (p & g).sum() / denom) if denom else float("nan")

def render(net, cfg, X, Y, L):
    """Our phase field on THEIR pixel grid (their coordinates are physical; ours parametric)."""
    xi = torch.tensor(X.ravel() / L, dtype=sb.DTYPE).reshape(-1, 1)
    eta = torch.tensor(Y.ravel() / L, dtype=sb.DTYPE).reshape(-1, 1)
    with torch.no_grad():
        phi = sb.phi_value(net, xi, eta, cfg)
    return phi.numpy().reshape(X.shape)

GROW = 0.05          # mm; the seeded phi>0.5 band is only ~1.3 l = 0.013 mm half-width

def crack_activity(phi, cracks, X, Y, l, thresh=GROW):
    """Assign every damaged pixel to its NEAREST seeded segment; a crack is ACTIVE if any of its
       own pixels sits further from the segment than the seeded band could ever reach.

       Shared with plot_reference.py, which produced the reference active/dormant table. The
       earlier version here measured axial reach among pixels within 3*l PERPENDICULAR of the
       seed axis, which reported 0 active on every run at every step: a crack that kinks away
       from its seed orientation - which is what they all do - leaves that window immediately,
       so the growth being looked for was excluded from the set it was measured on.
    """
    pts = np.stack([X.ravel(), Y.ravel()], 1)[phi.ravel() > 0.5]
    if not len(pts):
        return np.zeros(len(cracks), bool)
    D = np.empty((len(cracks), len(pts)))
    for i, (ax, ay, bx, by) in enumerate(cracks):
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey + 1e-30
        t = np.clip(((pts[:, 0] - ax) * ex + (pts[:, 1] - ay) * ey) / L2, 0.0, 1.0)
        D[i] = np.hypot(pts[:, 0] - (ax + t * ex), pts[:, 1] - (ay + t * ey))
    owner = D.argmin(0)
    return np.array([bool((D[i][owner == i] > thresh).any()) for i in range(len(cracks))])

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory produced by run_benchmark.py")
    ap.add_argument("--out", default=None, help="where to write the report (default: the run)")
    a = ap.parse_args()

    utils.set_device("cpu")
    out = a.out or a.run
    cfg = cfg_from_json(a.run)
    cfg.geomap = cfg.problem.build_geomap(cfg)
    d = np.load(cfg.sample)
    X, Y, L = d["x"], d["y"], float(d["domain"])
    theirs, t_ref, fd, fdy = d["phi"], d["t"], d["fd"], d["fd_y"]
    cracks = [tuple(float(v) for v in c) for c in d["cracks"]]

    ours_fd = np.atleast_2d(np.loadtxt(os.path.join(a.run, "FD.txt")))
    steps = sorted(int(s.split("_")[1]) for s in os.listdir(a.run) if s.startswith("step_"))
    if not steps:
        steps = [len(ours_fd) - 1]

    lines = [f"BENCHMARK COMPARISON  {a.run}",
             f"  sample {os.path.basename(cfg.sample)}   case {cfg.case}   "
             f"{len(cracks)} seeded cracks   seeding '{cfg.seed_band}'", ""]
    lines.append(f"{'step':>5}{'delta':>12}{'Dice':>8}{'F ours':>11}{'F theirs':>11}"
                 f"{'ratio':>8}   active ours/theirs")
    lines.append("-" * 78)

    net = sb.NetMS(cfg)
    panels = []
    for s in steps:
        ck = os.path.join(a.run, f"step_{s:03d}", "ckpt.pt")
        if not os.path.exists(ck):
            continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(c["model"]); net.eval()
        delta = cfg.U_ref * c["ud"]
        ours = render(net, cfg, X, Y, L)
        k = int(np.argmin(np.abs(t_ref - delta)))          # their nearest saved snapshot
        ref = theirs[k]
        j = int(np.argmin(np.abs(fd[:, 1] - delta)))
        F_ref = abs(fd[j, 0]) + (abs(fdy[j, 0]) if len(fdy) else 0.0)
        F_our = float(ours_fd[np.argmin(np.abs(ours_fd[:, 0] - delta)), 1])
        act_o = crack_activity(ours, cracks, X, Y, cfg.l)
        act_t = crack_activity(ref, cracks, X, Y, cfg.l)
        lines.append(f"{s:5d}{delta:12.3e}{dice(ours, ref):8.4f}{F_our:11.1f}{F_ref:11.1f}"
                     f"{F_our/F_ref if F_ref else float('nan'):8.3f}   "
                     f"{act_o.sum():2d}/{act_t.sum():2d}  agree "
                     f"{int((act_o == act_t).sum())}/{len(cracks)}")
        panels.append((delta, ours, ref))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out, "benchmark_comparison.txt"), "w") as f:
        f.write(report + "\n")

    if panels:
        n = len(panels)
        fig, ax = plt.subplots(2, n, figsize=(5.2 * n, 9.4), squeeze=False)
        for i, (delta, ours, ref) in enumerate(panels):
            for r, (A, ttl) in enumerate([(ref, "reference"), (ours, "ours")]):
                ax[r][i].pcolormesh(X, Y, A, cmap="magma", vmin=0, vmax=1, shading="nearest")
                ax[r][i].set_title(f"{ttl}   delta={delta:.2e}")
                ax[r][i].set_aspect("equal", "box")
                ax[r][i].set_xlabel("x (mm)"); ax[r][i].set_ylabel("y (mm)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "benchmark_comparison.png"), dpi=110)
        print(f"\nwrote benchmark_comparison.{{txt,png}} -> {out}")

if __name__ == "__main__":
    main()
