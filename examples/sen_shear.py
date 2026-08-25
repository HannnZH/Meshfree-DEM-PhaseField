#!/usr/bin/env python
"""
Single-edge-notched plate in SHEAR (mode II).

The same notched square, clamped top and bottom, with the top edge slid sideways. The crack
leaves the notch at an angle and curves down to the lower edge, and the load-displacement curve
softens, dips and hardens again before it snaps -- the shape that makes this the demanding case.

    python examples/sen_shear.py                      # the published run
    python examples/sen_shear.py --pf_order 4         # the same case, 4th-order functional
    python examples/sen_shear.py --resume             # continue from a checkpoint

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    loading="shear",
    l=0.01,
    levels="32,128,384",        # finest spacing ~0.26 l, matched to the band sampling
    N=12000,
    nsteps=126,
    delta_max=3.15e-3,          # 2.5e-5 mm per load step
    iters=2000,
    iters0=4000,
    split="hybrid",
    gc_grip=1.0,                # corner toughening: lets the final ligament sever cleanly
    ckpt_every=1,
    plot_every=10,
    out="runs/sen_shear",
)

if __name__ == "__main__":
    run_cli("sen", description=__doc__, defaults=DEFAULTS)
