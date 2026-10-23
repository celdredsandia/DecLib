"""Grids (cell complexes) backed by PETSc ``DMStag`` and ``DMPlex``."""

from __future__ import annotations

from collections.abc import Sequence

from declib import _core


class Grid:
    """Abstract cell complex.

    Python mirror of :cpp:class:`declib::Grid`. Use
    :class:`StructuredGrid` or :class:`UnstructuredGrid` to create one.
    """

    _impl: _core.Grid

    @property
    def dimension(self) -> int:
        """Topological dimension :math:`n` of the complex."""
        return self._impl.dimension()

    def num_cells(self, k: int) -> int:
        """Number of locally owned :math:`k`-cells.

        Parameters
        ----------
        k : int
            Cell degree, :math:`0 \\le k \\le n`.

        Returns
        -------
        int
            Count of :math:`k`-cells owned by this MPI rank.

        Raises
        ------
        ValueError
            If ``k`` is outside ``[0, dimension]``.
        """
        if not 0 <= k <= self.dimension:
            raise ValueError(f"k must be in [0, {self.dimension}], got {k}")
        return self._impl.num_cells(k)


class StructuredGrid(Grid):
    """Uniform Cartesian grid backed by PETSc ``DMStag``.

    Parameters
    ----------
    cells : sequence of int
        Number of elements per direction; its length sets the dimension.
    lengths : sequence of float, optional
        Physical extent per direction. Defaults to the unit box.

    See Also
    --------
    UnstructuredGrid : Meshes loaded from file via ``DMPlex``.

    Examples
    --------
    >>> grid = StructuredGrid((32, 32, 32))
    >>> grid.dimension
    3
    """

    def __init__(self, cells: Sequence[int], lengths: Sequence[float] | None = None):
        lengths = tuple(lengths) if lengths is not None else (1.0,) * len(cells)
        if len(lengths) != len(cells):
            raise ValueError("cells and lengths must have the same length")
        self._impl = _core.StructuredGrid(tuple(cells), lengths)


class UnstructuredGrid(Grid):
    """Unstructured mesh backed by PETSc ``DMPlex``.

    Parameters
    ----------
    path : str
        Mesh file in any format ``DMPlexCreateFromFile`` reads
        (Gmsh, Exodus II, HDF5, ...).
    interpolate : bool, default True
        Create the intermediate edges and faces.
    """

    def __init__(self, path: str, interpolate: bool = True):
        self._impl = _core.UnstructuredGrid(path, interpolate)
