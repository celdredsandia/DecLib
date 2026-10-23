"""Basic enumerations shared by every component."""

from __future__ import annotations

import enum


class Orientation(enum.Enum):
    """Orientation type of a differential form.

    Mirror of :cpp:enum:`declib::Orientation`. Straight (inner oriented)
    forms are integrated over inner-oriented cells, twisted (outer oriented)
    forms over outer-oriented cells. The Hodge star maps one to the other.
    """

    STRAIGHT = "straight"
    """Inner oriented form."""
    TWISTED = "twisted"
    """Outer oriented form (pseudo-form, density)."""

    def opposite(self) -> Orientation:
        """The other orientation."""
        return Orientation.TWISTED if self is Orientation.STRAIGHT else Orientation.STRAIGHT


class Complex(enum.Enum):
    """The primal cell complex or its dual.

    Mirror of :cpp:enum:`declib::Complex`. A dual :math:`k`-cell crosses one
    primal :math:`(n-k)`-cell and inherits an outer orientation from it.
    """

    PRIMAL = "primal"
    """The mesh itself."""
    DUAL = "dual"
    """The dual complex."""

    def opposite(self) -> Complex:
        """The other complex."""
        return Complex.DUAL if self is Complex.PRIMAL else Complex.PRIMAL
