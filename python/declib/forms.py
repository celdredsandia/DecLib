"""Discrete differential forms (cochains)."""

from __future__ import annotations

from typing import Literal

from declib import _core
from declib.grid import Grid


class Cochain:
    """A :math:`k`-cochain: one value per oriented :math:`k`-cell.

    Python mirror of :cpp:class:`declib::Cochain`. Values live in a PETSc
    ``Vec``, available through :attr:`vec` as a :class:`petsc4py.PETSc.Vec`.

    Parameters
    ----------
    grid : Grid
        Grid the cochain is defined on.
    degree : int
        Form degree :math:`k`.
    complex : {"primal", "dual"}, default "primal"
        Whether values live on primal or dual cells.
    """

    def __init__(self, grid: Grid, degree: int,
                 complex: Literal["primal", "dual"] = "primal"):
        self.grid = grid
        self.degree = degree
        self.complex = complex
        self._impl = _core.Cochain(grid._impl, degree, complex == "dual")

    @property
    def vec(self):
        """The underlying :class:`petsc4py.PETSc.Vec` (shared, not copied)."""
        return self._impl.vec()

    def __len__(self) -> int:
        return self._impl.size()
