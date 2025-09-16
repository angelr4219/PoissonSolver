# Backwards-compat shim: geometry._utils -> geometry.helpers
from .helpers import *  # re-export public names
__all__ = [n for n in dir() if not n.startswith("_")]
