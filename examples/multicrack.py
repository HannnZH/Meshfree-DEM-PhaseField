#!/usr/bin/env python
"""
Random multi-crack plate under biaxial tension or shear.

One sample of the public phase-field benchmark dataset of Hamdi and Lejeune, a 2 mm square
carrying ten to twenty randomly placed cracks. The dataset ships each sample as an .npz file
holding the crack segments, the reference fields and the reference reaction force; point the
script at one of them and choose the loading case.

    python examples/multicrack.py --sample data/tension/100192.npz --case tension
    python examples/multicrack.py --sample data/shear/100192.npz   --case shear

Every solver flag applies, so a cheap smoke test is a matter of shrinking them:

    python examples/multicrack.py --sample data/shear/100192.npz --case shear \
        --nsteps 4 --iters 150 --iters0 300 --N 6000 --levels 32,128,384 --grid_n 128

The material, the hybrid model matching theirs and the 2 mm patch come from
MultiCrackProblem.defaults(); only the load schedule is chosen per run.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
from _entry import run_cli

import pfpiml.multicrack                                        # noqa: F401  (registers it)

if __name__ == "__main__":
    run_cli("multicrack", description=__doc__)
