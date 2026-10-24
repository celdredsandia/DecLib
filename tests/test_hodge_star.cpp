#include "common.hpp"

using namespace declib;

TEST(HodgeStar, StarStarIsSign)
{
    for (int n = 1; n <= 3; ++n)
        for (bool periodic : {true, false})
            for (auto m : {GridMapping::straight_on_primal(), GridMapping::twisted_on_primal()})
                for (auto o : {Orientation::Straight, Orientation::Twisted})
                    for (int k = 0; k <= n; ++k) {
                        const auto M = test::make_manifold(n, 4, periodic, m, 1.7);
                        const auto s1 = HodgeStar::create(M->space(k, o));
                        const auto s2 = HodgeStar::create(s1->range());
                        EXPECT_TRUE(s2->range()->same_type_as(*s1->domain()));
                        const Form x = test::random_form(s1->domain());
                        Form y = (*s2)((*s1)(x));
                        y.axpy(((k * (n - k)) % 2) ? 1 : -1, x);
                        EXPECT_LT(y.norm_max(), 1e-13) << "n=" << n << " k=" << k;
                    }
}

// <a, *a> > 0: the pairing signs and the star signs agree.
TEST(HodgeStar, MassIsPositive)
{
    for (int n = 1; n <= 3; ++n)
        for (auto m : {GridMapping::straight_on_primal(), GridMapping::twisted_on_primal()})
            for (auto o : {Orientation::Straight, Orientation::Twisted})
                for (int k = 0; k <= n; ++k) {
                    const auto M = test::make_manifold(n, 4, false, m);
                    const auto s = HodgeStar::create(M->space(k, o));
                    const auto P = Pairing::create(s->domain());
                    for (int trial = 0; trial < 3; ++trial) {
                        const Form x = test::random_form(s->domain());
                        EXPECT_GT((*P)(x, (*s)(x)), 0) << "n=" << n << " k=" << k;
                    }
                }
}

TEST(HodgeStar, ApplyAssembleAgreeAndDiagonal)
{
    const auto M = test::make_manifold(3, 4, false);
    for (int k = 0; k <= 3; ++k) {
        const auto s = HodgeStar::create(M->space(k, Orientation::Straight));
        EXPECT_TRUE(s->is_diagonal());
        const Form x = test::random_form(s->domain());
        Form y = s->range()->zeros();
        Mat A = s->assemble();
        MatMult(A, x.global_vec(), y.global_vec());
        EXPECT_LT(test::diff_max(y, (*s)(x)), 1e-13);
        MatDestroy(&A);
        const OpPtr inv = inverse(s);
        EXPECT_LT(test::diff_max((*inv)((*s)(x)), x), 1e-13);
    }
}

TEST(HodgeStar, UniformValues)
{
    // 2D, h = 0.5: the star of a primal 0-form multiplies by the dual cell area,
    // h^2 inside, h^2/2 on edges, h^2/4 at corners.
    const auto M = test::make_manifold(2, 2, false);
    const auto s = HodgeStar::create(M->space(0, Orientation::Straight));
    Form x = s->domain()->zeros();
    x.fill(1);
    const Form y = (*s)(x);
    Scalar total;
    VecSum(y.global_vec(), &total);
    EXPECT_NEAR(total, 1.0, 1e-14); // the dual cells tile the domain
    EXPECT_NEAR(y.norm_max(), 0.25, 1e-14);
}

TEST(HodgeStar, UnsupportedSchemes)
{
    const auto M = test::make_manifold(2, 4, true);
    EXPECT_THROW((void)HodgeStar::create(M->space(1, Orientation::Straight), HodgeScheme::Galerkin), TypeError);
}
