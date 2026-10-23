"""Metric data attached to a topology."""

from __future__ import annotations

import enum
from collections.abc import Sequence

from declib import _core
from declib.topology import StagTopology


class DualPlacement(enum.Enum):
    """Placement of dual vertices. Mirror of :cpp:enum:`declib::DualPlacement`."""

    CIRCUMCENTRIC = "circumcentric"
    """At circumcentres."""
    BARYCENTRIC = "barycentric"
    """At barycentres."""


class Geometry:
    """Abstract metric data on a topology.

    Python mirror of :cpp:class:`declib::Geometry`: coordinates, dual vertex
    placement and cell measures. Metric-free operators never read it.
    """

    _impl: _core.Geometry

    @property
    def dual_placement(self) -> DualPlacement:
        """Placement of dual vertices."""
        return DualPlacement(self._impl.dual_placement())

    @property
    def is_orthogonal(self) -> bool:
        """Whether the metric is diagonal in the grid frame."""
        return self._impl.is_orthogonal()


class UniformGeometry(Geometry):
    """Uniformly spaced Cartesian geometry on a :class:`~declib.topology.StagTopology`.

    Mirror of :cpp:class:`declib::stag::UniformGeometry`.

    Parameters
    ----------
    topology : StagTopology
        The structured complex.
    lengths : sequence of float, optional
        Domain extent per axis; defaults to the unit box.
    origin : sequence of float, optional
        Low corner; defaults to the origin.
    """

    def __init__(self, topology: StagTopology, lengths: Sequence[float] | None = None,
                 origin: Sequence[float] | None = None):
        n = topology.dimension
        lengths = tuple(lengths) if lengths is not None else (1.0,) * n
        origin = tuple(origin) if origin is not None else (0.0,) * n
        if len(lengths) != n or len(origin) != n:
            raise ValueError(f"lengths and origin need {n} entries")
        self.topology = topology
        self._impl = _core.UniformGeometry(topology._impl, lengths, origin)

    def spacing(self, axis: int) -> float:
        """Grid spacing :math:`h_i = L_i / N_i` along ``axis``."""
        return self._impl.spacing(axis)
