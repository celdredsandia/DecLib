#include "common.hpp"

using namespace declib;

namespace {

void expect_agree(const OpPtr& A, const char* what)
{
    const Form x = test::random_form(A->domain());
    const Form y = (*A)(x);
    Form ya = A->range()->zeros(), ys = A->range()->zeros();
    Mat M = A->assemble(), S = A->shell();
    MatMult(M, x.global_vec(), ya.global_vec());
    MatMult(S, x.global_vec(), ys.global_vec());
    const Real scale = std::max<Real>(1, y.norm_max());
    EXPECT_LT(test::diff_max(y, ya), 1e-12 * scale) << what;
    EXPECT_LT(test::diff_max(y, ys), 1e-12 * scale) << what;

    // Euclidean transpose: <A x, z> = <x, A^T z>.
    const Form z = test::random_form(A->range());
    Form t = A->domain()->zeros();
    A->apply_transpose(z, t);
    EXPECT_NEAR(test::dot(y, z), test::dot(x, t), 1e-10 * scale * x.norm2() * z.norm2()) << what;
    MatDestroy(&M);
    MatDestroy(&S);
}

} // namespace

TEST(Operators, CompositesAgree)
{
    for (int n = 1; n <= 3; ++n)
        for (auto m : {GridMapping::straight_on_primal(), GridMapping::twisted_on_primal()})
            for (bool periodic : {true, false}) {
                const auto M = test::make_manifold(n, 4, periodic, m);
                for (int k = 0; k <= n; ++k) {
                    const auto V = M->space(k, Orientation::Straight);
                    expect_agree(hodge_laplacian(V), "hodge_laplacian");
                    if (k > 0) expect_agree(codifferential(V), "codifferential");
                }
            }
}

TEST(Operators, Algebra)
{
    const auto M = test::make_manifold(2, 5, false);
    const auto V = M->space(1, Orientation::Straight);
    const OpPtr L = hodge_laplacian(V);
    const OpPtr I = identity(V);
    const Form x = test::random_form(V);

    Form lhs = (*(L + I))(x);
    Form rhs = (*L)(x);
    rhs.axpy(1, x);
    EXPECT_LT(test::diff_max(lhs, rhs), 1e-12);

    lhs = (*(L - 2.0 * I))(x);
    rhs = (*L)(x);
    rhs.axpy(-2, x);
    EXPECT_LT(test::diff_max(lhs, rhs), 1e-12);

    expect_agree(L + I, "sum");
    expect_agree(3.0 * L, "scaled");
    expect_agree(transpose(L), "transpose");
    expect_agree(I, "identity");

    const OpPtr d = ExteriorDerivative::create(M->space(0, Orientation::Straight));
    EXPECT_THROW((void)(d * d), TypeError);
    EXPECT_THROW((void)(L + d), TypeError);
}

TEST(Operators, KrylovInverse)
{
    const auto M = test::make_manifold(2, 8, false);
    const auto V = M->space(1, Orientation::Straight);
    const OpPtr A = hodge_laplacian(V) + identity(V);
    const OpPtr Ainv = inverse(A, KSPOptions{"", 1e-12, nullptr});
    const Form b = test::random_form(V);
    const Form x = (*Ainv)(b);
    EXPECT_LT(test::diff_max((*A)(x), b), 1e-8 * b.norm_max());

    // Matrix-free operator with an assembled preconditioner.
    const OpPtr Ainv2 = inverse(A, KSPOptions{"", 1e-12, A});
    EXPECT_LT(test::diff_max((*Ainv2)(b), x), 1e-8 * x.norm_max());
    EXPECT_THROW((void)Ainv->assemble(), Error);
}

TEST(Operators, GenericPairingAdjoint)
{
    for (auto m : {GridMapping::straight_on_primal(), GridMapping::twisted_on_primal()}) {
        const auto M = test::make_manifold(2, 4, false, m);
        const auto V = M->space(1, Orientation::Straight);
        const OpPtr A = hodge_laplacian(V);
        const OpPtr Ad = A->pairing_adjoint();
        const auto P = Pairing::create(V);
        const Form x = test::random_form(V);
        const Form b = test::random_form(P->right());
        EXPECT_NEAR((*P)((*A)(x), b), (*P)(x, (*Ad)(b)), 1e-9 * (*A)(x).norm2() * b.norm2());
    }
}

// The same problem on one rank and distributed gives the same numbers.
TEST(Operators, IndependentOfDecomposition)
{
    for (int n = 2; n <= 3; ++n)
        for (bool periodic : {true, false}) {
            const auto Mw = test::make_manifold(n, 6, periodic, GridMapping::straight_on_primal(), 1, PETSC_COMM_WORLD);
            const auto Ms = test::make_manifold(n, 6, periodic, GridMapping::straight_on_primal(), 1, PETSC_COMM_SELF);
            for (int k = 0; k <= n; ++k) {
                const test::TrigForm w{n, k};
                Real norms[2];
                int i = 0;
                for (const auto& M : {Mw, Ms}) {
                    const auto V = M->space(k, Orientation::Straight);
                    const Form x = V->from_function(w.value());
                    norms[i++] = (*hodge_laplacian(V))(x).norm2();
                }
                EXPECT_NEAR(norms[0], norms[1], 1e-12 * norms[1]) << "n=" << n << " k=" << k;
            }
        }
}

TEST(Operators, NotYetImplemented)
{
    const auto M = test::make_manifold(2, 4, true);
    const Form X = M->space(1, Orientation::Straight)->zeros();
    EXPECT_THROW((void)lie_derivative(X, M->space(0, Orientation::Straight), nullptr), Error);
    EXPECT_THROW((void)CovariantDerivative::create(M->space(0, Orientation::Straight)), Error);
}
