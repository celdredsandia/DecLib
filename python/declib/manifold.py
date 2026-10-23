"""Discrete manifolds: topology, grid mapping and geometry combined."""

from __future__ import annotations

from declib import _core
from declib.core import Orientation
from declib.forms import Bundle, FormSpace, FormType
from declib.geometry import Geometry
from declib.mapping import GridMapping
from declib.topology import Topology


class DiscreteManifold:
    """Topology + grid mapping + geometry.

    Mirror of :cpp:class:`declib::DiscreteManifold`, and the factory for
    form spaces (one cached space per form type).

    Parameters
    ----------
    topology : Topology
        Cell complex.
    geometry : Geometry
        Metric data built on ``topology``.
    mapping : GridMapping, optional
        Where straight and twisted forms live; defaults to
        :meth:`~declib.mapping.GridMapping.straight_on_primal`.
    """

    def __init__(self, topology: Topology, geometry: Geometry,
                 mapping: GridMapping | None = None):
        self.topology = topology
        self.geometry = geometry
        self.mapping = mapping if mapping is not None else GridMapping.straight_on_primal()
        straight_on_primal = self.mapping == GridMapping.straight_on_primal()
        self._impl = _core.DiscreteManifold(topology._impl, geometry._impl, straight_on_primal)
        self._spaces: dict[FormType, FormSpace] = {}

    @property
    def dimension(self) -> int:
        """Topological dimension :math:`n`."""
        return self.topology.dimension

    def space(self, degree: int, orientation: Orientation,
              bundle: Bundle | None = None) -> FormSpace:
        """The cached space of forms of the given type.

        Parameters
        ----------
        degree : int
            Form degree :math:`k`, :math:`0 \\le k \\le n`.
        orientation : Orientation
            Straight or twisted.
        bundle : Bundle, optional
            Fiber; defaults to :meth:`~declib.forms.Bundle.scalar`.
        """
        if not 0 <= degree <= self.dimension:
            raise ValueError(f"degree must be in [0, {self.dimension}], got {degree}")
        ftype = FormType(degree, orientation, bundle if bundle is not None else Bundle.scalar())
        if ftype not in self._spaces:
            self._spaces[ftype] = FormSpace._from_impl(self, ftype, self._impl.space(ftype))
        return self._spaces[ftype]
