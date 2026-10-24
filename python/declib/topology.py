"""Cell complexes: the topology of a discrete manifold."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from declib import _core
from declib.core import Complex


@dataclass(frozen=True)
class CellSet:
    """All :math:`k`-cells of one complex. Mirror of :cpp:struct:`declib::CellSet`."""

    complex: Complex
    """Primal or dual complex."""
    k: int
    """Cell dimension."""


class DualBoundary(enum.Enum):
    """How the dual complex is closed on a non-periodic boundary.

    Mirror of :cpp:enum:`declib::DualBoundary`.
    """

    TRUNCATED = "truncated"
    """Dual cells clipped at the boundary; no dual cells on the boundary itself."""
    EXTENDED = "extended"
    """Dual complex includes cells on the boundary (default). Natural boundary
    conditions are imposed through these cells."""


class Boundary(enum.Enum):
    """Boundary type of one axis of a structured grid.

    Mirror of :cpp:enum:`declib::stag::Boundary`.
    """

    PERIODIC = "periodic"
    """Periodic axis."""
    GHOSTED = "ghosted"
    """Non-periodic, with a ghost layer."""
    NONE = "none"
    """Non-periodic; stored like :attr:`GHOSTED`, since operators read one cell past the boundary."""


class Topology:
    """Abstract oriented cell complex and its dual.

    Python mirror of :cpp:class:`declib::Topology`. Knows cell counts,
    incidence and orientations, but no coordinates or measures.
    """

    _impl: _core.Topology

    @property
    def dimension(self) -> int:
        """Topological dimension :math:`n`."""
        return self._impl.dimension()

    @property
    def dual_boundary(self) -> DualBoundary:
        """Closure of the dual complex at non-periodic boundaries."""
        return DualBoundary(self._impl.dual_boundary())

    def num_owned(self, cells: CellSet) -> int:
        """Number of cells of ``cells`` owned by this rank."""
        return self._impl.num_owned(cells.complex.value, cells.k)

    def num_global(self, cells: CellSet) -> int:
        """Number of cells of ``cells`` over all ranks."""
        return self._impl.num_global(cells.complex.value, cells.k)

    def incidence(self, cells: CellSet):
        """Signed incidence matrix :math:`\\partial_k` as a ``petsc4py.PETSc.Mat``.

        Parameters
        ----------
        cells : CellSet
            The :math:`k`-cells, :math:`1 \\le k \\le n`, on either complex.
        """
        return self._impl.incidence(cells.complex.value, cells.k)


class StagTopology(Topology):
    """Topologically structured complex on PETSc ``DMStag``.

    Mirror of :cpp:class:`declib::stag::Topology`; the dimension is the
    length of ``cells``.

    Parameters
    ----------
    cells : sequence of int
        Elements per axis.
    boundary : Boundary or sequence of Boundary, default Boundary.PERIODIC
        Boundary type, for all axes or per axis.
    dual_boundary : DualBoundary, default DualBoundary.EXTENDED
        Dual closure on non-periodic axes.
    stencil_width : int, default 1
        Halo width in elements.
    comm : mpi4py.MPI.Comm or petsc4py.PETSc.Comm, optional
        Communicator; defaults to ``PETSC_COMM_WORLD``.

    Examples
    --------
    >>> topo = StagTopology((32, 32, 32), boundary=Boundary.GHOSTED)
    >>> topo.dimension
    3
    """

    def __init__(self, cells: Sequence[int],
                 boundary: Boundary | Sequence[Boundary] = Boundary.PERIODIC,
                 dual_boundary: DualBoundary = DualBoundary.EXTENDED,
                 stencil_width: int = 1, comm=None):
        cells = tuple(cells)
        if not 1 <= len(cells) <= 3:
            raise ValueError(f"StagTopology supports 1 to 3 dimensions, got {len(cells)}")
        if isinstance(boundary, Boundary):
            boundary = (boundary,) * len(cells)
        boundary = tuple(boundary)
        if len(boundary) != len(cells):
            raise ValueError("boundary must have one entry per axis")
        self._impl = _core.StagTopology(cells, [b.value for b in boundary],
                                        dual_boundary.value, stencil_width, comm)
