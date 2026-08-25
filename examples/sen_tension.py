#!/usr/bin/env python
"""
Single-edge-notched plate in TENSION (mode I).

A square plate with a horizontal notch reaching the centre, clamped top and bottom and pulled
vertically. The crack runs straight across the remaining ligament. This is the calibration case:
the peak load and the displacement at which it fails are what the finite-element reference is
compared against.

    python examples/sen_tension.py                    # the published run
    python examples/sen_tension.py --pf_order 4       # the same case, 4th-order functional
    python examples/sen_tension.py --resume           # continue from a checkpoint

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    loading="tension",
    l=0.01,
    levels="32,128,384",        # finest spacing ~0.26 l, matched to the band sampling
    N=12000,
    nsteps=80,
    delta_max=8e-4,             # 1e-5 mm per load step
    iters=2000,
    iters0=4000,
    split="hybrid",
    gc_grip=1.0,                # corner toughening: keeps the clamped corners from nucleating
    ckpt_every=1,
    plot_every=10,
    out="runs/sen_tension",
)

if __name__ == "__main__":
    run_cli("sen", description=__doc__, defaults=DEFAULTS)
