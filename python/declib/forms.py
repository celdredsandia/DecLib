"""Discrete differential forms: types, spaces and elements."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from declib.core import Orientation

if TYPE_CHECKING:
    from declib.manifold import DiscreteManifold


class BundleKind(enum.Enum):
    """Kind of vector bundle. Mirror of :cpp:enum:`declib::BundleKind`."""

    SCALAR = "scalar"
    """Real-valued forms."""
    TRIVIAL = "trivial"
    """:math:`\\mathbb{R}^r` with a global frame."""
    TANGENT = "tangent"
    """Tangent bundle (vector-valued forms)."""
    COTANGENT = "cotangent"
    """Cotangent bundle (covector-valued forms)."""
    CUSTOM = "custom"
    """Any other bundle; needs a connection."""


@dataclass(frozen=True)
class Bundle:
    """The vector bundle a form takes values in. Mirror of :cpp:class:`declib::Bundle`."""

    kind: BundleKind
    """Kind of bundle."""
    rank: int
    """Fiber dimension :math:`r`."""
    fiber_orientation: Orientation = Orientation.STRAIGHT
    """Orientation of the fiber elements."""

    @classmethod
    def scalar(cls) -> Bundle:
        """Real-valued forms."""
        return cls(BundleKind.SCALAR, 1)

    @classmethod
    def trivial(cls, rank: int) -> Bundle:
        """:math:`\\mathbb{R}^r`-valued forms with a global frame."""
        return cls(BundleKind.TRIVIAL, rank)

    @classmethod
    def tangent(cls, n: int) -> Bundle:
        """Tangent-vector valued forms on an :math:`n`-manifold."""
        return cls(BundleKind.TANGENT, n)

    @classmethod
    def cotangent(cls, n: int, fiber_orientation: Orientation = Orientation.STRAIGHT) -> Bundle:
        """Covector valued forms on an :math:`n`-manifold."""
        return cls(BundleKind.COTANGENT, n, fiber_orientation)

    @property
    def is_trivial(self) -> bool:
        """Whether operators can act componentwise."""
        return self.kind in (BundleKind.SCALAR, BundleKind.TRIVIAL)


@dataclass(frozen=True)
class FormType:
    """Degree, orientation and fiber of a form. Mirror of :cpp:struct:`declib::FormType`."""

    degree: int
    """Form degree :math:`k`."""
    orientation: Orientation = Orientation.STRAIGHT
    """Straight or twisted."""
    bundle: Bundle = field(default_factory=Bundle.scalar)
    """Fiber."""


class FormSpace:
    """The space of discrete forms of one :class:`FormType`.

    Mirror of :cpp:class:`declib::FormSpace`. Obtain spaces from
    :meth:`declib.manifold.DiscreteManifold.space`; do not construct them
    directly.
    """

    manifold: DiscreteManifold
    """The manifold this space is defined on."""
    type: FormType
    """Type of the forms in this space."""

    @classmethod
    def _from_impl(cls, manifold: DiscreteManifold, ftype: FormType, impl) -> FormSpace:
        space = cls.__new__(cls)
        space.manifold, space.type, space._impl = manifold, ftype, impl
        return space

    @property
    def dm(self):
        """Layout ``petsc4py.PETSc.DM`` (a ``DMStag`` on structured grids)."""
        return self._impl.dm()

    @property
    def global_size(self) -> int:
        """Number of values over all ranks."""
        return self._impl.global_size()

    def dual_space(self) -> FormSpace:
        """The space paired with this one: degree :math:`n-k`, opposite orientation."""
        n = self.manifold.dimension
        return self.manifold.space(n - self.type.degree, self.type.orientation.opposite(),
                                   self.type.bundle)

    def zeros(self) -> Form:
        """A new form with all values zero."""
        return Form(self)

    def from_function(self, omega: Callable) -> Form:
        """A new form holding the de Rham map of ``omega``.

        Parameters
        ----------
        omega : callable
            ``omega(*x)`` returns the form's components at the point ``x``
            in the basis :math:`dx^{i_1} \\wedge \\dots \\wedge dx^{i_k}`,
            :math:`i_1 < \\dots < i_k`, lexicographically ordered.
        """
        form = Form(self)
        self._impl.de_rham(omega, form._impl)
        return form


class Form:
    """A discrete form: one fiber value per cell of its space.

    Mirror of :cpp:class:`declib::Form`.

    Parameters
    ----------
    space : FormSpace
        Space of the new (zero) form.
    """

    def __init__(self, space: FormSpace):
        self.space = space
        self._impl = space._impl.create_form()

    @property
    def type(self) -> FormType:
        """Shorthand for ``self.space.type``."""
        return self.space.type

    @property
    def vec(self):
        """Owned values as a ``petsc4py.PETSc.Vec`` (shared, not copied)."""
        return self._impl.global_vec()

    def to_numpy(self):
        """Host copy of the owned values, shaped by element index and stratum."""
        return self._impl.to_numpy()

    def copy(self) -> Form:
        """A deep copy in the same space."""
        out = Form(self.space)
        out._impl.copy_from(self._impl)
        return out

    def __iadd__(self, other: Form) -> Form:
        self._impl.axpy(1.0, other._impl)
        return self

    def __imul__(self, a: float) -> Form:
        self._impl.scale(a)
        return self
