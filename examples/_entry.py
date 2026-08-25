"""
Shared preamble for the example scripts.

Each example is a standalone script that can be run from anywhere, so it has to put the
repository root on the import path before importing the package. That is all this does.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfpiml import run_cli                                       # noqa: E402,F401
