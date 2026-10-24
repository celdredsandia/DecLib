"""Linear, bilinear and nonlinear operators on discrete forms."""

from __future__ import annotations

import enum

from declib import _core
from declib.core import Orientation
from declib.forms import Form, FormSpace


class LinearOperator:
    """A linear map :math:`A : V \\to W` between form spaces.

    Mirror of :cpp:class:`declib::LinearOperator`. Call an operator to apply
    it and compose operators with ``@``. Python subclasses may override
    :meth:`apply` to prototype an operator on the host before writing a
    kernel.
    """

    domain: FormSpace
    """Space of inputs."""
    range: FormSpace
    """Space of outputs."""

    def apply(self, x: Form, y: Form) -> None:
        """Compute :math:`y = A x`."""
        self._impl.apply(x._impl, y._impl)

    def __call__(self, x: Form) -> Form:
        y = self.range.zeros()
        self.apply(x, y)
        return y

    def apply_transpose(self, y: Form, x: Form) -> None:
        """Algebraic transpose :math:`x = A^T y` of the coefficient matrix."""
        self._impl.apply_transpose(y._impl, x._impl)

    @property
    def is_diagonal(self) -> bool:
        """Whether the coefficient matrix is diagonal."""
        return self._impl.is_diagonal()

    @property
    def name(self) -> str:
        """Name used in error messages."""
        return self._impl.name()

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        if not other.range.same_type_as(self.domain):
            raise TypeError(f"cannot compose: {other.range.type} does not match {self.domain.type}")
        return _Wrapped(other.domain, self.range, _core.compose(self._impl, other._impl))

    def __add__(self, other: LinearOperator) -> LinearOperator:
        _check_same_spaces(self, other)
        return _Wrapped(self.domain, self.range, _core.add(self._impl, other._impl))

    def __sub__(self, other: LinearOperator) -> LinearOperator:
        _check_same_spaces(self, other)
        return _Wrapped(self.domain, self.range, _core.subtract(self._impl, other._impl))

    def __rmul__(self, a: float) -> LinearOperator:
        return _Wrapped(self.domain, self.range, _core.scale(a, self._impl))

    def __neg__(self) -> LinearOperator:
        return (-1.0) * self

    @property
    def T(self) -> LinearOperator:
        """The algebraic transpose :math:`A^T : W \\to V` as an operator."""
        return _Wrapped(self.range, self.domain, _core.transpose(self._impl))

    def pairing_adjoint(self) -> LinearOperator:
        """The adjoint :math:`A^\\dagger` under the duality pairing.

        Maps ``range.dual_space()`` to ``domain.dual_space()`` with
        :math:`\\langle A x, \\beta \\rangle = \\langle x, A^\\dagger \\beta \\rangle`.
        """
        return _Wrapped(self.range.dual_space(), self.domain.dual_space(),
                        self._impl.pairing_adjoint())

    def shell(self):
        """A matrix-free ``petsc4py.PETSc.Mat``."""
        return self._impl.shell()

    def assemble(self, mat_type: str = "aijkokkos"):
        """An assembled ``petsc4py.PETSc.Mat`` of the given type."""
        return self._impl.assemble(mat_type)


def _check_same_spaces(a: LinearOperator, b: LinearOperator) -> None:
    if not (a.domain.same_type_as(b.domain) and a.range.same_type_as(b.range)):
        raise TypeError("cannot add operators with different domains or ranges")


class _Wrapped(LinearOperator):
    def __init__(self, domain: FormSpace, range: FormSpace, impl):
        self.domain, self.range, self._impl = domain, range, impl


def inverse(A: LinearOperator, rtol: float = 1e-10, prefix: str = "") -> LinearOperator:
    """:math:`A^{-1}`: exact for diagonal operators, otherwise a ``KSP`` solve.

    See :cpp:func:`declib::inverse`.
    """
    return _Wrapped(A.range, A.domain, _core.inverse(A._impl, rtol, prefix))


def identity(space: FormSpace) -> LinearOperator:
    """The identity on ``space``."""
    return _Wrapped(space, space, _core.identity(space._impl))


class ExteriorDerivative(LinearOperator):
    """Exterior derivative :math:`d : \\Lambda^k \\to \\Lambda^{k+1}`.

    Mirror of :cpp:class:`declib::ExteriorDerivative`. Metric-free;
    :math:`d \\circ d = 0` to round-off.

    Parameters
    ----------
    domain : FormSpace
        Space of :math:`k`-forms, :math:`k < n`.
    """

    def __init__(self, domain: FormSpace):
        t = domain.type
        self.domain = domain
        self.range = domain.manifold.space(t.degree + 1, t.orientation, t.bundle)
        self._impl = _core.ExteriorDerivative(domain._impl)


class HodgeScheme(enum.Enum):
    """Discretization of the Hodge star. Mirror of :cpp:enum:`declib::HodgeScheme`."""

    DIAGONAL = "diagonal"
    """Ratio of dual to primal measures; needs an orthogonal geometry."""
    GALERKIN = "galerkin"
    """Whitney-form mass matrix; any mapped geometry."""
    HIGH_ORDER = "high_order"
    """Reserved."""


class HodgeStar(LinearOperator):
    """Hodge star :math:`\\star : \\Lambda^k \\to \\tilde\\Lambda^{n-k}`.

    Mirror of :cpp:class:`declib::HodgeStar`. Flips the orientation.

    Parameters
    ----------
    domain : FormSpace
        Space of :math:`k`-forms of either orientation.
    scheme : HodgeScheme, default HodgeScheme.DIAGONAL
        Discretization.
    """

    def __init__(self, domain: FormSpace, scheme: HodgeScheme = HodgeScheme.DIAGONAL):
        t = domain.type
        n = domain.manifold.dimension
        self.domain = domain
        self.range = domain.manifold.space(n - t.degree, t.orientation.opposite(), t.bundle)
        self.scheme = scheme
        self._impl = _core.HodgeStar(domain._impl, scheme.value)


class Pairing:
    """Metric-free duality pairing between a space and its dual.

    Mirror of :cpp:class:`declib::Pairing`. ``Pairing(V)(a, b)`` is the
    discrete :math:`\\int_M a \\wedge b` for ``a`` in ``V`` and ``b`` in
    ``V.dual_space()``.
    """

    def __init__(self, space: FormSpace):
        self.left = space
        self.right = space.dual_space()
        self._impl = _core.Pairing(space._impl)

    def __call__(self, a: Form, b: Form) -> float:
        return self._impl.pair(a._impl, b._impl)


class BilinearOperator:
    """A bilinear map :math:`B : U \\times V \\to W`.

    Mirror of :cpp:class:`declib::BilinearOperator`. Call it as
    ``B(a, b)``; freeze one argument with :meth:`fix_left` or
    :meth:`fix_right` to get a :class:`LinearOperator`.
    """

    left: FormSpace
    """Space of the first argument."""
    right: FormSpace
    """Space of the second argument."""
    range: FormSpace
    """Space of the result."""

    def __call__(self, a: Form, b: Form) -> Form:
        c = self.range.zeros()
        self._impl.apply(a._impl, b._impl, c._impl)
        return c

    def fix_left(self, a: Form) -> LinearOperator:
        """The linear operator :math:`b \\mapsto B(a, b)`; follows later changes to ``a``."""
        return _Wrapped(self.right, self.range, self._impl.fix_left(a._impl))

    def fix_right(self, b: Form) -> LinearOperator:
        """The linear operator :math:`a \\mapsto B(a, b)`; follows later changes to ``b``."""
        return _Wrapped(self.left, self.range, self._impl.fix_right(b._impl))


class Wedge(BilinearOperator):
    """Wedge product :math:`\\Lambda^k \\times \\Lambda^l \\to \\Lambda^{k+l}`.

    Mirror of :cpp:class:`declib::Wedge`. The scheme is required; concrete
    schemes are not chosen yet.

    Parameters
    ----------
    left, right : FormSpace
        Argument spaces.
    scheme : str
        Name of a registered wedge scheme.
    """

    def __init__(self, left: FormSpace, right: FormSpace, scheme: str):
        raise NotImplementedError("Wedge schemes are planned for phase 2")


class InteriorProduct(BilinearOperator):
    """Interior product :math:`(X, \\alpha) \\mapsto i_X \\alpha`.

    Mirror of :cpp:class:`declib::InteriorProduct`. ``X`` is a twisted
    :math:`(n-1)`-form (flux) or a straight 1-form (circulation). The scheme
    is required; concrete schemes are not chosen yet.

    Parameters
    ----------
    vector_space : FormSpace
        Space holding the vector field.
    form_space : FormSpace
        Space of :math:`k`-forms, :math:`k \\ge 1`.
    scheme : str
        Name of a registered interior-product scheme.
    """

    def __init__(self, vector_space: FormSpace, form_space: FormSpace, scheme: str):
        raise NotImplementedError("interior product schemes are planned for phase 2")

    def adjoint(self) -> BilinearOperator:
        """The pairing adjoint :math:`i_X^\\dagger`.

        :math:`\\langle i_X \\alpha, \\beta \\rangle = \\langle \\alpha, i_X^\\dagger \\beta \\rangle`
        to round-off. See :cpp:class:`declib::InteriorProductAdjoint`.
        """
        adj = BilinearOperator.__new__(BilinearOperator)
        adj._impl = self._impl.adjoint()
        adj.left, adj.right, adj.range = self.left, self.range.dual_space(), self.right.dual_space()
        return adj


class NonlinearOperator:
    """A differentiable nonlinear map :math:`F : V \\to W`.

    Mirror of :cpp:class:`declib::NonlinearOperator`. Subclass it and
    implement :meth:`apply` and :meth:`jacobian`.
    """

    domain: FormSpace
    """Space of inputs."""
    range: FormSpace
    """Space of outputs."""

    def apply(self, x: Form, y: Form) -> None:
        """Compute :math:`y = F(x)`."""
        raise NotImplementedError

    def jacobian(self, x: Form) -> LinearOperator:
        """The linearization :math:`DF(x)`."""
        raise NotImplementedError

    def __call__(self, x: Form) -> Form:
        y = self.range.zeros()
        self.apply(x, y)
        return y


class Functional:
    """A differentiable scalar functional, such as an energy.

    Mirror of :cpp:class:`declib::Functional`. The gradient lives in
    ``domain.dual_space()``.
    """

    domain: FormSpace
    """Space of arguments."""

    def value(self, x: Form) -> float:
        """:math:`\\mathcal{F}(x)`."""
        raise NotImplementedError

    def gradient(self, x: Form, g: Form) -> None:
        """Write :math:`\\delta\\mathcal{F}/\\delta x` into ``g``."""
        raise NotImplementedError


def codifferential(space: FormSpace, scheme: HodgeScheme = HodgeScheme.DIAGONAL) -> LinearOperator:
    """Codifferential :math:`\\delta = (-1)^{n(k+1)+1} \\star d \\star` on ``space``."""
    t = space.type
    target = space.manifold.space(t.degree - 1, t.orientation, t.bundle)
    return _Wrapped(space, target, _core.codifferential(space._impl, scheme.value))


def hodge_laplacian(space: FormSpace, scheme: HodgeScheme = HodgeScheme.DIAGONAL) -> LinearOperator:
    """Hodge Laplacian :math:`\\Delta = d\\delta + \\delta d` on ``space``."""
    return _Wrapped(space, space, _core.hodge_laplacian(space._impl, scheme.value))


def mass(space: FormSpace, scheme: HodgeScheme = HodgeScheme.DIAGONAL) -> LinearOperator:
    """Mass matrix :math:`M : V \\to V^*` of the :math:`L^2` inner product."""
    return _Wrapped(space, space.dual_space(), _core.mass(space._impl, scheme.value))


def lie_derivative(X: Form, space: FormSpace, scheme: str) -> LinearOperator:
    """Lie derivative :math:`L_X = d\\, i_X + i_X\\, d` on ``space``."""
    raise NotImplementedError("the Lie derivative is planned for phase 2")


def apply_essential(space: FormSpace, A, x: Form, b: Form) -> None:
    """Impose the essential conditions of ``space`` on the system :math:`A x = b`.

    Writes the prescribed values into ``x``, zeros the constrained rows and
    columns of the ``petsc4py.PETSc.Mat`` ``A`` (unit diagonal, so ``A``
    stays symmetric if it was) and moves the known values to ``b``. See
    :cpp:func:`declib::apply_essential`.
    """
    _core.apply_essential(space._impl, A, x._impl, b._impl)
