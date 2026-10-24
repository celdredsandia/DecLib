#include "common.hpp"

using namespace declib;

namespace {

struct Case {
    int n;
    bool periodic;
    GridMapping mapping;
    Orientation orientation;
};

std::vector<Case> all_cases()
{
    std::vector<Case> cases;
    for (int n = 1; n <= 3; ++n)
        for (bool p : {true, false})
            for (auto m : {GridMapping::straight_on_primal(), GridMapping::twisted_on_primal()})
                for (auto o : {Orientation::Straight, Orientation::Twisted}) cases.push_back({n, p, m, o});
    return cases;
}

std::string describe(const Case& c, int k)
{
    return "n=" + std::to_string(c.n) + " k=" + std::to_string(k) + (c.periodic ? " periodic " : " bounded ") +
           test::mapping_name(c.mapping) + (c.orientation == Orientation::Straight ? " straight" : " twisted");
}

} // namespace

TEST(ExteriorDerivative, SquaresToZero)
{
    for (const auto& c : all_cases()) {
        const auto M = test::make_manifold(c.n, 5, c.periodic, c.mapping);
        for (int k = 0; k + 2 <= c.n; ++k) {
            const auto d0 = ExteriorDerivative::create(M->space(k, c.orientation));
            const auto d1 = ExteriorDerivative::create(d0->range());
            const Form x = test::random_form(d0->domain());
            const Form y = (*d1)((*d0)(x));
            EXPECT_LT(y.norm_max(), 1e-13) << describe(c, k);
        }
    }
}

TEST(ExteriorDerivative, ApplyAssembleShellAgree)
{
    for (const auto& c : all_cases()) {
        const auto M = test::make_manifold(c.n, 4, c.periodic, c.mapping);
        for (int k = 0; k < c.n; ++k) {
            const auto d = ExteriorDerivative::create(M->space(k, c.orientation));
            const Form x = test::random_form(d->domain());
            const Form y = (*d)(x);
            Form ya = d->range()->zeros(), ys = d->range()->zeros();
            Mat A = d->assemble(), S = d->shell();
            MatMult(A, x.global_vec(), ya.global_vec());
            MatMult(S, x.global_vec(), ys.global_vec());
            EXPECT_LT(test::diff_max(y, ya), 1e-14) << describe(c, k);
            EXPECT_LT(test::diff_max(y, ys), 1e-14) << describe(c, k);
            MatDestroy(&A);
            MatDestroy(&S);
        }
    }
}

// Stokes' theorem: d commutes with the de Rham map.
TEST(ExteriorDerivative, CommutesWithDeRham)
{
    for (const auto& c : all_cases()) {
        if (!c.periodic) continue; // trigonometric forms are periodic
        const auto M = test::make_manifold(c.n, 6, true, c.mapping);
        for (int k = 0; k < c.n; ++k) {
            const test::TrigForm w{c.n, k};
            const auto d = ExteriorDerivative::create(M->space(k, c.orientation));
            const Form Rw = d->domain()->from_function(w.value(), Quadrature{14});
            const Form Rdw = d->range()->from_function(w.derivative(), Quadrature{14});
            const Form dRw = (*d)(Rw);
            EXPECT_LT(test::diff_max(dRw, Rdw), 1e-10 * Rdw.norm_max()) << describe(c, k);
        }
    }
}

// The structured adjoint, (-1)^(k+1) times d on the dual complex, equals the
// generic S_V d^T S_W built from the pairing signs.
TEST(ExteriorDerivative, PairingAdjointIsDualDerivative)
{
    for (const auto& c : all_cases()) {
        const auto M = test::make_manifold(c.n, 4, c.periodic, c.mapping);
        for (int k = 0; k < c.n; ++k) {
            const auto d = ExteriorDerivative::create(M->space(k, c.orientation));
            const OpPtr fast = d->pairing_adjoint();
            const OpPtr generic = d->LinearOperator::pairing_adjoint();
            EXPECT_TRUE(fast->domain()->same_type_as(*generic->domain()));
            EXPECT_TRUE(fast->range()->same_type_as(*generic->range()));
            Mat A = fast->assemble(MATAIJ), B = generic->assemble(MATAIJ);
            EXPECT_LT(test::mat_diff_norm(A, B), 1e-14) << describe(c, k);
            MatDestroy(&A);
            MatDestroy(&B);

            // <d x, b> = <x, d^dagger b>
            const auto P = Pairing::create(d->range());
            const auto Q = Pairing::create(d->domain());
            const Form x = test::random_form(d->domain());
            const Form b = test::random_form(P->right());
            EXPECT_NEAR((*P)((*d)(x), b), (*Q)(x, (*fast)(b)), 1e-12) << describe(c, k);
        }
    }
}

TEST(ExteriorDerivative, TypeErrors)
{
    const auto M = test::make_manifold(2, 4, true);
    EXPECT_THROW((void)ExteriorDerivative::create(M->space(2, Orientation::Straight)), TypeError);
    EXPECT_THROW((void)ExteriorDerivative::create(M->space(0, Orientation::Straight, Bundle::tangent(2))),
                 TypeError);
    const auto d = ExteriorDerivative::create(M->space(0, Orientation::Straight));
    const Form wrong = M->space(0, Orientation::Twisted)->zeros();
    EXPECT_THROW((void)(*d)(wrong), TypeError);
}

TEST(ExteriorDerivative, BundleValuedComponentwise)
{
    const auto M = test::make_manifold(2, 4, false);
    const auto V = M->space(0, Orientation::Straight, Bundle::trivial(3));
    const auto d = ExteriorDerivative::create(V);
    const auto d1 = ExteriorDerivative::create(d->range());
    const Form x = test::random_form(V);
    EXPECT_LT((*d1)((*d)(x)).norm_max(), 1e-13);
    EXPECT_EQ(d->range()->global_size(), 3 * 40);
}
