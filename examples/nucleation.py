#!/usr/bin/env python
"""
Crack nucleation at a hole in a plate under tension.

A square plate with a central circular hole, clamped on the left edge and pulled on the right.
There is NO pre-existing crack: damage has to appear on its own at the two points where the
stress concentrates, and grow from there across the plate. The hole is not meshed -- it is
removed from the quadrature, which leaves its edge traction-free automatically.

Run examples/kirsch.py first to check the elastic field this case starts from.

    python examples/nucleation.py                     # the published run
    python examples/nucleation.py --resume            # continue from a checkpoint

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    mode="nucleate",
    loading="tension_x",        # left edge clamped, right edge pulled horizontally
    cracks="none",              # nothing seeded: the crack has to nucleate
    hole_r=0.1,
    l=0.01,
    levels="32,128,384",
    N=12000,
    rho_new=1.5e5,              # extra points on the band once a crack does appear
    nsteps=110,
    delta_max=1.1e-3,           # 1e-5 mm per load step
    iters=1500,
    iters0=3000,
    split="hybrid",
    gc_grip=1.0,
    at=2,
    ckpt_every=1,
    plot_every=5,
    out="runs/nucleation",
)

if __name__ == "__main__":
    run_cli("hole", description=__doc__, defaults=DEFAULTS)
