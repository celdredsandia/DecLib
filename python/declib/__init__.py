"""DecLib++: discrete exterior calculus on Kokkos and PETSc.

The Python package mirrors the C++ classes one to one. The compiled
extension ``declib._core`` is built with pybind11; PETSc objects cross the
boundary as :mod:`petsc4py` objects.

Examples
--------
>>> import declib as dl
>>> topo = dl.StagTopology(cells=(64, 64))
>>> geom = dl.UniformGeometry(topo, lengths=(1.0, 1.0))
>>> M = dl.DiscreteManifold(topo, geom)
>>> V0 = M.space(0, dl.Orientation.STRAIGHT)
>>> d0 = dl.ExteriorDerivative(V0)
>>> d0.range.type.degree
1
"""

from declib.core import Complex, Orientation
from declib.forms import Bundle, Form, FormSpace, FormType
from declib.geometry import DualPlacement, UniformGeometry
from declib.manifold import DiscreteManifold
from declib.mapping import GridMapping
from declib.operators import (
    BilinearOperator,
    ExteriorDerivative,
    Functional,
    HodgeScheme,
    HodgeStar,
    InteriorProduct,
    LinearOperator,
    NonlinearOperator,
    Pairing,
    Wedge,
    codifferential,
    hodge_laplacian,
    identity,
    inverse,
    lie_derivative,
    mass,
)
from declib.topology import Boundary, CellSet, DualBoundary, StagTopology

__version__ = "0.1.0"

__all__ = [
    "BilinearOperator",
    "Boundary",
    "Bundle",
    "CellSet",
    "Complex",
    "DiscreteManifold",
    "DualBoundary",
    "DualPlacement",
    "ExteriorDerivative",
    "Form",
    "FormSpace",
    "FormType",
    "Functional",
    "GridMapping",
    "HodgeScheme",
    "HodgeStar",
    "InteriorProduct",
    "LinearOperator",
    "NonlinearOperator",
    "Orientation",
    "Pairing",
    "StagTopology",
    "UniformGeometry",
    "Wedge",
    "codifferential",
    "hodge_laplacian",
    "identity",
    "inverse",
    "lie_derivative",
    "mass",
]
