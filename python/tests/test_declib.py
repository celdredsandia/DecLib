"""Python interface tests (serial)."""

import math

import numpy as np
import pytest
from petsc4py import PETSc

import declib as dl

MAPPINGS = [dl.GridMapping.straight_on_primal(), dl.GridMapping.twisted_on_primal()]


def manifold(n, N, boundary=dl.Boundary.PERIODIC, mapping=None):
    topo = dl.StagTopology((N,) * n, boundary=boundary)
    geom = dl.UniformGeometry(topo)
    return dl.DiscreteManifold(topo, geom, mapping)


def random_form(space, seed=0):
    f = space.zeros()
    rng = np.random.default_rng(seed)
    f.vec.setArray(rng.uniform(-1, 1, space.local_size))
    return f


def test_topology_counts():
    topo = dl.StagTopology((4, 4), boundary=dl.Boundary.GHOSTED)
    assert topo.dimension == 2
    assert topo.num_global(dl.CellSet(dl.Complex.PRIMAL, 0)) == 25
    assert topo.num_global(dl.CellSet(dl.Complex.PRIMAL, 1)) == 40
    B = topo.incidence(dl.CellSet(dl.Complex.PRIMAL, 1))
    assert B.getSize() == (25, 40)


def test_spaces_are_cached_and_typed():
    M = manifold(2, 4)
    V = M.space(1, dl.Orientation.STRAIGHT)
    assert M.space(1, dl.Orientation.STRAIGHT) is V
    assert V.dual_space().type == dl.FormType(1, dl.Orientation.TWISTED)
    assert V.complex is dl.Complex.PRIMAL
    M2 = manifold(2, 4, mapping=dl.GridMapping.twisted_on_primal())
    assert M2.space(1, dl.Orientation.STRAIGHT).complex is dl.Complex.DUAL


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_d_squared_is_zero(n, mapping):
    M = manifold(n, 4, boundary=dl.Boundary.GHOSTED, mapping=mapping)
    for k in range(n - 1):
        d0 = dl.ExteriorDerivative(M.space(k, dl.Orientation.STRAIGHT))
        d1 = dl.ExteriorDerivative(d0.range)
        assert (d1 @ d0)(random_form(d0.domain)).norm("max") < 1e-13


@pytest.mark.parametrize("n", [1, 2, 3])
def test_star_star(n):
    M = manifold(n, 4)
    for k in range(n + 1):
        s = dl.HodgeStar(M.space(k, dl.Orientation.STRAIGHT))
        s2 = dl.HodgeStar(s.range)
        x = random_form(s.domain)
        y = s2(s(x))
        y.axpy(-(-1) ** (k * (n - k)), x)
        assert y.norm("max") < 1e-13


def test_shell_and_assembled_agree():
    M = manifold(2, 6, boundary=dl.Boundary.GHOSTED)
    L = dl.hodge_laplacian(M.space(1, dl.Orientation.STRAIGHT))
    x = random_form(L.domain)
    y = L(x)
    A, S = L.assemble(), L.shell()
    ya, ys = A.createVecLeft(), S.createVecLeft()
    A.mult(x.vec, ya)
    S.mult(x.vec, ys)
    assert np.allclose(ya.getArray(), y.to_numpy(), atol=1e-11)
    assert np.allclose(ys.getArray(), y.to_numpy(), atol=1e-11)


def test_pairing_adjoint():
    M = manifold(3, 4, boundary=dl.Boundary.GHOSTED)
    d = dl.ExteriorDerivative(M.space(1, dl.Orientation.STRAIGHT))
    P_range, P_domain = dl.Pairing(d.range), dl.Pairing(d.domain)
    x = random_form(d.domain, 1)
    b = random_form(P_range.right, 2)
    assert P_range(d(x), b) == pytest.approx(P_domain(x, d.pairing_adjoint()(b)), abs=1e-12)


def test_algebra_and_type_errors():
    M = manifold(2, 4)
    V0 = M.space(0, dl.Orientation.STRAIGHT)
    d = dl.ExteriorDerivative(V0)
    with pytest.raises(TypeError):
        d @ d
    I = dl.identity(V0)
    L = dl.hodge_laplacian(V0)
    x = random_form(V0)
    lhs = (L + 2.0 * I)(x)
    rhs = L(x)
    rhs.axpy(2.0, x)
    lhs -= rhs
    assert lhs.norm("max") < 1e-12
    with pytest.raises(NotImplementedError):
        dl.Wedge(V0, V0, "any")


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_hodge_laplacian_second_order(mapping):
    def error(N):
        M = manifold(2, N, mapping=mapping)
        V = M.space(1, dl.Orientation.STRAIGHT)
        lam = 4 * math.pi**2 * 2

        def w(x, y):
            return (math.sin(2 * math.pi * (x + y)), math.cos(2 * math.pi * (x - y)))

        def lw(x, y):
            return tuple(lam * c for c in w(x, y))

        y = dl.hodge_laplacian(V)(V.from_function(w))
        exact = V.from_function(lw)
        e = y.copy()
        e -= exact
        return e.norm("max") / exact.norm("max")

    rate = math.log2(error(16) / error(32))
    assert rate > 1.9


def test_poisson_with_essential_conditions():
    def u(x, y):
        return math.sin(math.pi * x + 0.5) * math.sin(math.pi * y + 0.5)

    def error(N):
        M = manifold(2, N, boundary=dl.Boundary.GHOSTED)
        V = M.space(0, dl.Orientation.STRAIGHT).with_essential([dl.EssentialCondition(dl.BoundaryRegion.all(), u)])
        A = dl.hodge_laplacian(V).assemble()
        b = V.from_function(lambda x, y: 2 * math.pi**2 * u(x, y))
        x = V.zeros()
        dl.apply_essential(V, A, x, b)
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(b.vec, x.vec)
        assert ksp.getConvergedReason() > 0
        exact = V.from_function(u)
        x -= exact
        return x.norm("max")

    assert math.log2(error(16) / error(32)) > 1.8


def test_krylov_inverse():
    M = manifold(2, 8, boundary=dl.Boundary.GHOSTED)
    V = M.space(2, dl.Orientation.STRAIGHT)
    A = dl.hodge_laplacian(V) + dl.identity(V)
    b = random_form(V)
    x = dl.inverse(A, rtol=1e-12)(b)
    r = A(x)
    r -= b
    assert r.norm("max") < 1e-8
