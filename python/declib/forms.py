"""Discrete differential forms: types, spaces and elements."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from declib import _core
from declib.core import Complex, Orientation

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


class Side(enum.Enum):
    """Side of a boundary face. Mirror of :cpp:enum:`declib::Side`."""

    LOW = "low"
    """Index 0 along the axis."""
    HIGH = "high"
    """Last index along the axis."""


class BoundaryRegion:
    """A part of the boundary. Mirror of :cpp:class:`declib::BoundaryRegion`.

    Create one with :meth:`all` or :meth:`face`.
    """

    def __init__(self, impl, description: str):
        self._impl, self._description = impl, description

    @classmethod
    def all(cls) -> BoundaryRegion:
        """The whole boundary."""
        return cls(_core.BoundaryRegion.all(), "all")

    @classmethod
    def face(cls, axis: int, side: Side) -> BoundaryRegion:
        """The boundary plane at the low or high end of ``axis``."""
        return cls(_core.BoundaryRegion.face(axis, side is Side.HIGH), f"face({axis}, {side.name})")

    def __repr__(self) -> str:
        return f"BoundaryRegion.{self._description}"


@dataclass(frozen=True)
class EssentialCondition:
    """Prescribed values on a boundary region. Mirror of :cpp:struct:`declib::EssentialCondition`.

    The constrained values become fixed degrees of freedom; see
    :meth:`FormSpace.with_essential` and :func:`declib.operators.apply_essential`.
    """

    region: BoundaryRegion = field(default_factory=BoundaryRegion.all)
    """Where the condition holds."""
    value: Callable | None = None
    """``value(*x)``, as for :meth:`FormSpace.from_function`; ``None`` means zero."""


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

    @property
    def local_size(self) -> int:
        """Number of values owned by this rank."""
        return self._impl.local_size()

    @property
    def complex(self) -> Complex:
        """Complex hosting the forms (see :class:`~declib.mapping.GridMapping`)."""
        return Complex(self._impl.complex())

    def same_type_as(self, other: FormSpace) -> bool:
        """Same manifold and form type; essential conditions may differ."""
        return self.manifold is other.manifold and self.type == other.type

    def dual_space(self) -> FormSpace:
        """The space paired with this one: degree :math:`n-k`, opposite orientation."""
        n = self.manifold.dimension
        return self.manifold.space(n - self.type.degree, self.type.orientation.opposite(),
                                   self.type.bundle)

    def zeros(self) -> Form:
        """A new form with all values zero."""
        return Form(self)

    def with_essential(self, conditions: list[EssentialCondition]) -> FormSpace:
        """A copy of this space with essential (Dirichlet-type) conditions added.

        Only forms hosted on the primal complex take essential conditions;
        forms on the dual complex take boundary data through the extended
        dual cells.
        """
        impl = self._impl.with_essential([(c.region._impl, c.value) for c in conditions])
        space = FormSpace._from_impl(self.manifold, self.type, impl)
        space.essential_conditions = tuple(self.essential_conditions) + tuple(conditions)
        return space

    essential_conditions: tuple[EssentialCondition, ...] = ()
    """Conditions attached with :meth:`with_essential`."""

    def boundary_indices(self, region: BoundaryRegion):
        """Global indices of the values on ``region`` owned by this rank, as a ``petsc4py.PETSc.IS``."""
        return self._impl.boundary_indices(region._impl)

    def from_function(self, omega: Callable, order: int = 4) -> Form:
        """A new form holding the de Rham map of ``omega``.

        Parameters
        ----------
        omega : callable
            ``omega(*x)`` returns the form's components at the point ``x``
            in the basis :math:`dx^{i_1} \\wedge \\dots \\wedge dx^{i_k}`,
            :math:`i_1 < \\dots < i_k`, lexicographically ordered, one
            block of ``rank`` values per basis element for bundle-valued
            forms. A scalar return value is accepted when there is a
            single component.
        order : int, default 4
            Polynomial degree integrated exactly on each cell.
        """
        form = Form(self)
        self._impl.de_rham(omega, form._impl, order)
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
        """Host copy of the owned values, in the global ``DMStag`` ordering."""
        return self.vec.getArray(readonly=True).copy()

    def copy(self) -> Form:
        """A deep copy in the same space."""
        out = Form(self.space)
        out._impl.copy_from(self._impl)
        return out

    def fill(self, a: float) -> Form:
        """Set every value to ``a``."""
        self._impl.fill(a)
        return self

    def axpy(self, a: float, x: Form) -> Form:
        """``self += a * x``."""
        self._impl.axpy(a, x._impl)
        return self

    def norm(self, kind: str = "2") -> float:
        """Euclidean (``"2"``) or max (``"max"``) norm of the coefficient vector."""
        return self._impl.norm2() if kind == "2" else self._impl.norm_max()

    def __iadd__(self, other: Form) -> Form:
        self._impl.axpy(1.0, other._impl)
        return self

    def __isub__(self, other: Form) -> Form:
        self._impl.axpy(-1.0, other._impl)
        return self

    def __imul__(self, a: float) -> Form:
        self._impl.scale(a)
        return self
