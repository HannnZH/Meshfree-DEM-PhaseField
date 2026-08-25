#!/usr/bin/env python
"""
Kirsch stress-concentration check for the plate with a hole.

An elastic-only run on the same plate as examples/nucleation.py, with damage switched off. It
measures the hoop stress around the hole and compares it with the analytical Kirsch solution,
which peaks at three times the remote stress at the top and bottom of the hole and dips to minus
one at its sides. It verifies two things at once: that the stresses recovered by automatic
differentiation are right, and that masking the hole out of the quadrature really does leave a
traction-free edge.

Writes kirsch_scf.txt (the measured and analytical profiles) and kirsch_scf.png.

    python examples/kirsch.py

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

DEFAULTS = dict(
    mode="kirsch",
    loading="tension_x",        # left edge clamped, right edge pulled horizontally
    cracks="none",
    hole_r=0.1,
    l=0.01,
    levels="32,128,384",
    N=12000,
    iters0=4000,                # the cold solve the published SCF panel was measured from
    out="runs/kirsch",
)

if __name__ == "__main__":
    run_cli("hole", description=__doc__, defaults=DEFAULTS)
