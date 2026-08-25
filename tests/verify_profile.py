"""Optimal-profile surface-energy check (Borden design property, Eq 14):
the crack-density functional evaluated on its OWN optimal 1D profile integrates to Gc
per unit crack length. This validates (a) the Borden->our-convention coefficient map and
(b) the autodiff Laplacian used in the 4th-order fork.

Our convention (phi=1-c, Miehe l = profile-decay scale):
  2nd-order AT2 : e = (Gc/2)[ phi^2/l + l phi'^2 ]                 , phi0 = exp(-d/l)
  4th-order     : e = (Gc/2)[ phi^2/l + (l/2) phi'^2 + (l^3/16) phi''^2 ], phi0 = exp(-2d/l)(1+2d/l)
Transverse integral over the full band (both sides of the crack) must equal Gc.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import torch
torch.set_default_dtype(torch.float64)

Gc = 0.04247
l = 0.01

# fine 1D transverse grid, wide enough for the profile to decay (|y| up to 30 l)
y = torch.linspace(-30 * l, 30 * l, 400001, requires_grad=True)
d = torch.abs(y)

def integ(f, x):
    return torch.trapz(f, x)

# ---- 2nd order ----
phi2 = torch.exp(-d / l)
(g2,) = torch.autograd.grad(phi2.sum(), y, create_graph=True)
e2 = 0.5 * Gc * (phi2 ** 2 / l + l * g2 ** 2)
E2 = integ(e2, y).item()

# ---- 4th order ----
r = d / l
phi4 = torch.exp(-2.0 * r) * (1.0 + 2.0 * r)
(g4,) = torch.autograd.grad(phi4.sum(), y, create_graph=True)
(gg4,) = torch.autograd.grad(g4.sum(), y, create_graph=True)   # phi'' via a 2nd autodiff pass
e4 = 0.5 * Gc * (phi4 ** 2 / l + 0.5 * l * g4 ** 2 + (l ** 3 / 16.0) * gg4 ** 2)
E4 = integ(e4, y).item()

print(f"Gc (target per unit crack length) = {Gc:.6f}")
print(f"2nd-order profile surface energy  = {E2:.6f}   err {100*(E2-Gc)/Gc:+.3f}%")
print(f"4th-order profile surface energy  = {E4:.6f}   err {100*(E4-Gc)/Gc:+.3f}%")
# also report the term split for the 4th-order (sanity: all three positive, Laplacian nonzero)
print(f"  4th-order term integrals: bulk {integ(0.5*Gc*phi4**2/l, y).item():.6f}  "
      f"grad {integ(0.5*Gc*0.5*l*g4**2, y).item():.6f}  "
      f"lap {integ(0.5*Gc*(l**3/16.0)*gg4**2, y).item():.6f}")
