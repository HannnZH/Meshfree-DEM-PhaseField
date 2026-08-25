"""
Shared plumbing: the torch backend, the example registry, logging and checkpoints.

DEVICE lives here and nowhere else. Every module reads it as ``utils.DEVICE`` (an attribute
lookup at call time, never ``from .utils import DEVICE``) so that a single set_device call
retargets the whole package -- this is what lets the figure tools render checkpoints on the CPU
after the run was trained on a GPU.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import logging
import os
import sys
import types

import numpy as np
import torch

# --------------------------------------------------------------------------
# torch backend

torch.set_default_dtype(torch.float64)

DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_device(dev):
    """Retarget the package (e.g. set_device('cpu') to render checkpoints on a machine
       without a GPU, or to force deterministic CPU evaluation)."""
    global DEVICE
    DEVICE = str(dev)
    return DEVICE

# --------------------------------------------------------------------------
# example registry

# The examples live in pfpiml.problems, which imports geometry and solver -- both of which ask
# the registry for the running example. Importing it lazily here, inside get_problem, is what
# keeps that from closing an import cycle.

def _registry():
    from . import problems
    return problems.REGISTRY

def available():
    """Names of the registered examples."""
    return tuple(_registry())

def get_problem(name):
    """Return the (singleton) Problem instance registered under this name."""
    reg = _registry()
    key = str(name)
    if key not in reg:
        raise KeyError(f"unknown problem {name!r}; available: {', '.join(reg)}")
    return reg[key]

def problem_of(cfg):
    """The Problem behind a cfg, tolerating configs rebuilt from JSON (where cfg.problem is
       still the plain name) and legacy configs that predate the field (they are all SEN runs)."""
    p = getattr(cfg, "problem", None)
    if p is None or isinstance(p, str):
        p = get_problem(p or "sen")
        try:
            cfg.problem = p
        except AttributeError:              # frozen/duck-typed config: resolve every call
            pass
    return p

# --------------------------------------------------------------------------
# checkpoints and logging

def save_ckpt(path, net, prev_state, step, ud, FD, cfg):
    torch.save({"model": net.state_dict(), "prev": prev_state,
                "step": step, "ud": ud, "FD": FD, "cfg": cfg.to_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state()}, path)

def load_ckpt(path, net):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ck["model"])
    torch.set_rng_state(ck["torch_rng"].to("cpu"))
    np.random.set_state(ck["numpy_rng"])
    return ck

def get_logger(save_dir):
    logger = logging.getLogger("stageBms")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(save_dir, "train.log")); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

# --------------------------------------------------------------------------
# legacy module compatibility

class _LegacyModule(types.ModuleType):
    """Module whose DEVICE attribute is an alias of pfpiml.utils.DEVICE.

       Reading it returns the package's current device and assigning to it retargets the whole
       package, which is what the figure tools do (`sb.DEVICE = "cpu"`) before rendering
       checkpoints from a GPU run on a machine without one."""

    def __getattr__(self, name):                     # only called when normal lookup fails
        if name == "DEVICE":
            return DEVICE
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name == "DEVICE":
            set_device(value)                        # retarget the package, not just this module
            return
        object.__setattr__(self, name, value)

def install_device_proxy(module_name):
    """Turn the named (already imported) module into a DEVICE-forwarding shim.

    The module must NOT bind a plain ``DEVICE`` name of its own, or reads would find that copy
    instead of reaching the proxy."""
    mod = sys.modules[module_name]
    mod.__dict__.pop("DEVICE", None)
    mod.__class__ = _LegacyModule
    return mod
