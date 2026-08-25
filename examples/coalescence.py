#!/usr/bin/env python
"""
Coalescence of three en-echelon cracks under tension.

Three short cracks are seeded at 45 degrees in a staggered line. Under vertical tension they
grow towards each other and link into one staircase crack. Merging cracks are the case that
makes explicit crack tracking painful -- here the topology change costs nothing, because the
damage field owns the topology and there is no crack geometry to update.

    python examples/coalescence.py                    # the published run
    python examples/coalescence.py --pf_order 4       # the same case, 4th-order functional

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    loading="tension",
    cracks="0.25,0.35,0.35,0.45;0.45,0.45,0.55,0.55;0.65,0.55,0.75,0.65",
    l=0.01,
    levels="32,128,384",
    N=12000,
    nsteps=100,
    delta_max=1e-3,             # 1e-5 mm per load step
    iters=2000,
    iters0=4000,
    split="hybrid",
    gc_grip=1.0,                # keeps the pulled corners quiet so the cracks link, not the corners
    ckpt_every=1,
    plot_every=10,
    out="runs/coalescence",
)

if __name__ == "__main__":
    run_cli("sen", description=__doc__, defaults=DEFAULTS)
