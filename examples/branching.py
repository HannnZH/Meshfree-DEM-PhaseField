#!/usr/bin/env python
"""
Crack branching in the notched plate under shear.

Identical to the shear case except that the full isotropic energy is degraded (--split iso), so
damage is driven in every mode rather than in tension only. The crack then forks: two branches
leave the notch and each forks again, giving the symmetric double Y. Nothing in the solver knows
about branching -- no crack is tracked, so a crack that becomes two costs nothing to represent.

The crack stratum normally follows the damage itself, which feeds back and starves one branch;
--fixed_notch keeps that stratum on a fixed symmetric band so both branches are sampled alike.

    python examples/branching.py                      # the published run
    python examples/branching.py --pf_order 4         # does the 4th-order model still branch?

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    loading="shear",
    split="iso",                # no split: damage is driven isotropically => the crack forks
    fixed_notch=True,           # symmetric sampling, so neither branch is starved of points
    rho_new=2e5,                # extra points seeded on the branches as they propagate
    l=0.01,
    levels="32,128,384",
    N=12000,
    nsteps=126,
    delta_max=3.15e-3,          # 2.5e-5 mm per load step
    iters=2000,
    iters0=4000,
    gc_grip=1.0,
    ckpt_every=1,
    plot_every=10,
    out="runs/branching",
)

if __name__ == "__main__":
    run_cli("sen", description=__doc__, defaults=DEFAULTS)
