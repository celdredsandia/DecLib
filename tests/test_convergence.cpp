// Phase 1 acceptance: the Hodge Laplacian converges at second order in 1D-3D
// on both grid mappings.
#include "common.hpp"

using namespace declib;

namespace {

PetscInt coarse(int n) { return n == 3 ? 8 : 16; }

std::string label(int n, int k, const GridMapping& m)
{
    return "n=" + std::to_string(n) + " k=" + std::to_string(k) + " " + test::mapping_name(m);
}

// Relative max-norm error of the Hodge Laplacian applied to the de Rham
// projection of a periodic eigenform.
Real consistency_error(int n, int k, PetscInt N, const GridMapping& m)
{
    const auto M = test::make_manifold(n, N, true, m);
    const auto V = M->space(k, Orientation::Straight);
    const test::TrigForm w{n, k};
    const Form x = V->from_function(w.value());
    const Form exact = V->from_function(w.laplacian());
    const Form y = (*hodge_laplacian(V))(x);
    return test::diff_max(y, exact) / exact.norm_max();
}

Real solve(Mat A, const Form& b, Form& x)
{
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 200);
    KSPSetTolerances(ksp, 1e-12, 1e-14, PETSC_DEFAULT, 5000);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, b.global_vec(), x.global_vec());
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    KSPDestroy(&ksp);
    return reason;
}

// u = prod_a sin(pi x_a + c), -lap u = n pi^2 u.
FormFunction product_sine(int n, Real c, Real factor)
{
    return [=](std::span<const Real> x, std::span<Scalar> v) {
        Real u = factor;
        for (int a = 0; a < n; ++a) u *= std::sin(test::pi * x[a] + c);
        v[0] = u;
    };
}

// Poisson problem for 0-forms on the unit cube.
//  - straight_on_primal: values on primal vertices, Dirichlet data imposed
//    as essential conditions (constrained dofs);
//  - twisted_on_primal: values at element centres (dual vertices), with the
//    homogeneous data carried by the extended dual boundary cells.
Real poisson_error(int n, PetscInt N, const GridMapping& m)
{
    const auto M = test::make_manifold(n, N, false, m);
    const bool primal = m == GridMapping::straight_on_primal();
    const Real c = primal ? 0.5 : 0.0; // nonzero boundary data on the primal side
    const auto V0 = M->space(0, Orientation::Straight);
    const auto V = primal ? V0->with_essential({EssentialCondition{BoundaryRegion::all(), product_sine(n, c, 1)}})
                          : V0;
    const OpPtr L = hodge_laplacian(V);
    Mat A = L->assemble();
    Form b = V->from_function(product_sine(n, c, n * test::pi * test::pi));
    Form x = V->zeros();
    if (primal) apply_essential(*V, A, x, b);
    EXPECT_GT(solve(A, b, x), 0);
    MatDestroy(&A);
    const Form exact = V->from_function(product_sine(n, c, 1));
    return test::diff_max(x, exact) / exact.norm_max();
}

} // namespace

TEST(Convergence, HodgeLaplacianConsistencyPeriodic)
{
    for (int n = 1; n <= 3; ++n)
        for (auto m : {GridMapping::straight_on_primal(), GridMapping::twisted_on_primal()})
            for (int k = 0; k <= n; ++k) {
                const Real e1 = consistency_error(n, k, coarse(n), m);
                const Real e2 = consistency_error(n, k, 2 * coarse(n), m);
                const Real rate = std::log2(e1 / e2);
                PetscPrintf(PETSC_COMM_WORLD, "  consistency %-28s error %.3e -> %.3e  rate %.2f\n",
                            label(n, k, m).c_str(), e1, e2, rate);
                EXPECT_GT(rate, 1.9) << label(n, k, m) << " errors " << e1 << " " << e2;
                EXPECT_LT(e2, 0.05) << label(n, k, m);
            }
}

TEST(Convergence, PoissonDirichlet)
{
    for (int n = 1; n <= 3; ++n)
        for (auto m : {GridMapping::straight_on_primal(), GridMapping::twisted_on_primal()}) {
            const Real e1 = poisson_error(n, coarse(n), m);
            const Real e2 = poisson_error(n, 2 * coarse(n), m);
            const Real rate = std::log2(e1 / e2);
            PetscPrintf(PETSC_COMM_WORLD, "  poisson     %-28s error %.3e -> %.3e  rate %.2f\n",
                        label(n, 0, m).c_str(), e1, e2, rate);
            EXPECT_GT(rate, 1.8) << label(n, 0, m) << " errors " << e1 << " " << e2;
            EXPECT_LT(e2, 0.02) << label(n, 0, m);
        }
}
