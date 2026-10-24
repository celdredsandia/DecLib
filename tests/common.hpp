// Shared helpers for the C++ tests.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "declib/declib.hpp"

namespace declib::test {

inline constexpr Real pi = std::numbers::pi_v<Real>;

// A structured manifold on [0, L]^n with N cells per axis.
inline std::shared_ptr<const DiscreteManifold> make_manifold(int n, PetscInt N, bool periodic,
                                                             GridMapping mapping = GridMapping::straight_on_primal(),
                                                             Real L = 1, MPI_Comm comm = PETSC_COMM_WORLD)
{
    const auto b = periodic ? stag::Boundary::Periodic : stag::Boundary::Ghosted;
    auto build = [&]<int Dim>() {
        typename stag::Topology<Dim>::Options o;
        o.cells.fill(N);
        o.boundary.fill(b);
        auto topo = stag::Topology<Dim>::create(comm, o);
        typename stag::UniformGeometry<Dim>::Options g;
        g.lengths.fill(L);
        auto geom = stag::UniformGeometry<Dim>::create(topo, g);
        return DiscreteManifold::create(topo, geom, mapping);
    };
    switch (n) {
    case 1: return build.template operator()<1>();
    case 2: return build.template operator()<2>();
    default: return build.template operator()<3>();
    }
}

inline Form random_form(const SpacePtr& space)
{
    Form f = space->zeros();
    PetscRandom rnd;
    PetscRandomCreate(PETSC_COMM_WORLD, &rnd);
    PetscRandomSetInterval(rnd, -1, 1);
    VecSetRandom(f.global_vec(), rnd);
    PetscRandomDestroy(&rnd);
    return f;
}

inline Real dot(const Form& a, const Form& b)
{
    Scalar r;
    VecDot(a.global_vec(), b.global_vec(), &r);
    return r;
}

// Max-norm of a - b.
inline Real diff_max(const Form& a, const Form& b)
{
    Form t = a.copy();
    t.axpy(-1, b);
    return t.norm_max();
}

inline Real mat_diff_norm(Mat A, Mat B)
{
    Mat C;
    MatDuplicate(A, MAT_COPY_VALUES, &C);
    MatAXPY(C, -1, B, DIFFERENT_NONZERO_PATTERN);
    Real r;
    MatNorm(C, NORM_FROBENIUS, &r);
    MatDestroy(&C);
    return r;
}

inline int binomial(int n, int k)
{
    int r = 1;
    for (int i = 0; i < k; ++i) r = r * (n - i) / (i + 1);
    return r;
}

// Bit masks of the basis k-forms dx^I in basis order (lexicographic, which
// for n <= 3 is the order of the masks).
inline std::vector<unsigned> basis(int n, int k)
{
    std::vector<unsigned> b;
    for (unsigned m = 0; m < (1u << n); ++m)
        if (std::popcount(m) == k) b.push_back(m);
    return b;
}

// A smooth periodic k-form w = sum_I f_I dx^I on the unit cube with
// f_I = prod_a sin(2 pi m_a x_a + phase_{I,a}), so -lap f_I = lambda f_I.
struct TrigForm {
    int n;
    int k;
    std::array<int, 3> m{1, 2, 1};

    Real phase(int I, int a) const { return 0.3 + 0.7 * I + 1.1 * a; }

    Real lambda() const
    {
        Real s = 0;
        for (int a = 0; a < n; ++a) s += m[a] * m[a];
        return 4 * pi * pi * s;
    }

    Real f(int I, std::span<const Real> x) const
    {
        Real v = 1;
        for (int a = 0; a < n; ++a) v *= std::sin(2 * pi * m[a] * x[a] + phase(I, a));
        return v;
    }

    Real df(int I, int b, std::span<const Real> x) const
    {
        Real v = 1;
        for (int a = 0; a < n; ++a) {
            const Real t = 2 * pi * m[a] * x[a] + phase(I, a);
            v *= a == b ? 2 * pi * m[a] * std::cos(t) : std::sin(t);
        }
        return v;
    }

    FormFunction value() const
    {
        return [*this](std::span<const Real> x, std::span<Scalar> out) {
            for (int I = 0; I < binomial(n, k); ++I) out[I] = f(I, x);
        };
    }

    FormFunction laplacian() const
    {
        return [*this](std::span<const Real> x, std::span<Scalar> out) {
            for (int I = 0; I < binomial(n, k); ++I) out[I] = lambda() * f(I, x);
        };
    }

    // dw = sum_I sum_b df_I/dx_b dx^b ^ dx^I.
    FormFunction derivative() const
    {
        return [*this](std::span<const Real> x, std::span<Scalar> out) {
            const auto src = basis(n, k);
            const auto dst = basis(n, k + 1);
            for (std::size_t J = 0; J < dst.size(); ++J) out[J] = 0;
            for (std::size_t I = 0; I < src.size(); ++I)
                for (int b = 0; b < n; ++b) {
                    if ((src[I] >> b) & 1u) continue;
                    const unsigned mask = src[I] | (1u << b);
                    const int sign = (std::popcount(src[I] & ((1u << b) - 1u)) % 2) ? -1 : 1;
                    for (std::size_t J = 0; J < dst.size(); ++J)
                        if (dst[J] == mask) out[J] += sign * df(static_cast<int>(I), b, x);
                }
        };
    }
};

inline const char* mapping_name(const GridMapping& m)
{
    return m == GridMapping::straight_on_primal() ? "straight_on_primal" : "twisted_on_primal";
}

} // namespace declib::test
