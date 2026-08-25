#!/usr/bin/env python
"""
Thick-walled ring with two edge notches, pulled apart across the notches
(Si et al., Comput. Methods Appl. Mech. Engrg. 2023, sec 4.5).

The right half is modelled on ONE exact NURBS patch: a curved domain with no faceting and no
multi-patch coupling, which is what the isogeometric geometry map buys. The outer arc is pulled
on its upper half and clamped on its lower half through a boundary penalty -- the one boundary
condition here that cannot be imposed exactly -- and that penalty also supplies the reaction
force.

    python examples/ring.py                           # the published run
    python examples/ring.py --resume                  # continue from a checkpoint

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    geo="ring",
    ring_ri=5.0,                # inner radius (mm)
    ring_ro=20.0,               # outer radius (mm)
    ring_notch=3.0,             # radial notch depth (mm)
    E=210e3, nu=0.3, Gc=2.7,    # steel, in N and mm
    l=0.2,
    U_ref=0.033,                # the ring cracks at about this displacement
    k_bc=1e6,                   # outer-arc penalty stiffness
    grip_l=0.0, gc_grip=0.0,    # no clamped square corners here, so no corner devices
    split="hybrid",
    levels="64,256,1024",       # the ring is 40 mm across at l = 0.2, so it needs finer grids
    enc_start="0,0,0",
    grid_n=512,
    N=18000,
    nsteps=120,
    delta_max=0.036,
    iters=1500,
    iters0=3000,
    ckpt_every=1,
    plot_every=5,
    out="runs/ring",
)

if __name__ == "__main__":
    run_cli("ring", description=__doc__, defaults=DEFAULTS)
