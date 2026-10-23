"""DecLib++: discrete exterior calculus on Kokkos and PETSc.

The Python package wraps the C++ library (compiled extension
``declib._core``) and interoperates with :mod:`petsc4py` and NumPy.

Examples
--------
>>> import declib
>>> grid = declib.StructuredGrid((64, 64), lengths=(1.0, 1.0))
>>> u = declib.Cochain(grid, degree=0)
>>> du = declib.d(u)
>>> du.degree
1
"""

from declib.forms import Cochain
from declib.grid import Grid, StructuredGrid, UnstructuredGrid
from declib.operators import d, hodge_star, laplacian

__version__ = "0.1.0"

__all__ = [
    "Cochain",
    "Grid",
    "StructuredGrid",
    "UnstructuredGrid",
    "d",
    "hodge_star",
    "laplacian",
]
