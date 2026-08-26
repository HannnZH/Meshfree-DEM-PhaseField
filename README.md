# A mesh-free multiresolution deep energy method with phase-field modeling of brittle fracture

Reference implementation for the paper of the same name. One neural network carries the
displacement and phase fields, the coordinates enter it through a multiresolution grid of
C1 quadratic B-spline features, and the incremental energy is minimized directly on integration
points that are redrawn at every optimizer iteration. Essential boundary conditions hold exactly
through lifts, curved domains enter through a NURBS map, interior holes through a domain mask,
and the second- and fourth-order fracture energy densities run on the same discretization.

Preprint: https://arxiv.org/abs/2608.24126

Contact: han.zhang7@unsw.edu.au

## Layout

    pfpiml/        the solver: configuration, geometry and NURBS maps, problems, sampler,
                   energy and training loop, plotting
    examples/      one script per case in the paper; each is runnable on its own
    fem/           the staggered finite element solver used for the reference solutions
    tools/         scoring of a multi-crack run against the reference fields of the dataset

## Requirements

Python 3.10 with the packages in `requirements.txt`; a CUDA device is optional but expected for
the production runs, which use double precision throughout. The finite element reference needs
`scikit-fem` and, for the finer meshes, `pypardiso`.

## Running the examples

Every script takes the solver flags and writes its output to `runs/<name>/`.

    python examples/sen_tension.py             # single-edge-notched tension
    python examples/sen_shear.py               # single-edge-notched shear
    python examples/sen_shear.py --pf_order 4  # the same case, fourth-order density
    python examples/branching.py               # crack branching under isotropic driving
    python examples/coalescence.py             # coalescence of en-echelon cracks
    python examples/nucleation.py              # plate with a circular hole
    python examples/kirsch.py                  # its elastic stage against the Kirsch solution
    python examples/ring.py                    # thick-walled ring on one NURBS patch

Add `--resume` to continue an interrupted run from the rolling checkpoint. The published
settings are the defaults of each script; the table in the appendix of the paper lists them.

## The multi-crack benchmark

`examples/multicrack.py` runs one sample of the public phase-field benchmark dataset of Hamdi
and Lejeune. Download the dataset separately, point the script at one sample and choose the
loading case.

    python examples/multicrack.py --sample <sample>.npz --case tension
    python examples/multicrack.py --sample <sample>.npz --case shear --ir_tol 0.02

`tools/compare_multicrack.py` scores a finished run against the reference fields it carries,
reporting the Dice coefficient and the per-crack active or dormant classification.

## Finite element references

    python fem/pf_fem.py --example shear --split hybrid --l 0.01 --refine 8 --out runs/fem_shear

The same geometry, material, boundary conditions and regularization length as the corresponding
deep energy run; irreversibility through a history field, the crack through a phi = 1 condition
on the notch nodes, and an alternating displacement and phase-field solve per load increment.

## Adding a case

Subclass `Problem` in `pfpiml/problems.py`, override the hooks that differ, and register it in
`REGISTRY`. `pfpiml/multicrack.py` is a complete example, including a non-unit domain, its own
boundary conditions and a phase-field condition on part of the boundary.

## Citation

    @article{zhang2026meshfree,
      author  = {Zhang, Han and Makki Alamdari, Mehrisadat and Shahbodagh, Babak
                 and Vahab, Mohammad and Anitescu, Cosmin and Rabczuk, Timon
                 and Atroshchenko, Elena},
      title   = {A mesh-free multiresolution deep energy method with phase-field
                 modeling of brittle fracture},
      journal = {arXiv preprint arXiv:2608.24126},
      year    = {2026}
    }

## License

A license has not been chosen yet. Please contact the authors before redistributing or
reusing this code.
