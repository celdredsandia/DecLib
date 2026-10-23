"""Assignment of straight and twisted forms to the primal and dual complexes."""

from __future__ import annotations

from declib.core import Complex, Orientation
from declib.topology import CellSet


class GridMapping:
    """Which complex hosts straight forms and which hosts twisted forms.

    Mirror of :cpp:class:`declib::GridMapping`. Create one with
    :meth:`straight_on_primal` or :meth:`twisted_on_primal`.
    """

    def __init__(self, straight_host: Complex):
        self._straight_host = straight_host

    @classmethod
    def straight_on_primal(cls) -> GridMapping:
        """Straight forms on the primal complex, twisted forms on the dual."""
        return cls(Complex.PRIMAL)

    @classmethod
    def twisted_on_primal(cls) -> GridMapping:
        """Twisted forms on the primal complex, straight forms on the dual."""
        return cls(Complex.DUAL)

    def host(self, orientation: Orientation) -> Complex:
        """Complex hosting forms of the given orientation."""
        if orientation is Orientation.STRAIGHT:
            return self._straight_host
        return self._straight_host.opposite()

    def cells(self, k: int, orientation: Orientation) -> CellSet:
        """Cells on which a :math:`k`-form of the given orientation is stored."""
        return CellSet(self.host(orientation), k)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GridMapping) and other._straight_host is self._straight_host

    def __hash__(self) -> int:
        return hash(self._straight_host)

    def __repr__(self) -> str:
        side = "straight" if self._straight_host is Complex.PRIMAL else "twisted"
        return f"GridMapping.{side}_on_primal()"
