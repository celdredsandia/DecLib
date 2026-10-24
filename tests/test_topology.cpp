#include "common.hpp"

using namespace declib;

namespace {

PetscInt cells(const std::shared_ptr<const DiscreteManifold>& M, Complex c, int k)
{
    return M->topology().num_global(CellSet{c, k});
}

} // namespace

TEST(Topology, CountsNonPeriodic2D)
{
    const auto M = test::make_manifold(2, 4, false);
    EXPECT_EQ(cells(M, Complex::Primal, 0), 25);
    EXPECT_EQ(cells(M, Complex::Primal, 1), 40);
    EXPECT_EQ(cells(M, Complex::Primal, 2), 16);
    // Dual k-cells correspond to primal (n-k)-cells, boundary ones included.
    EXPECT_EQ(cells(M, Complex::Dual, 0), 16);
    EXPECT_EQ(cells(M, Complex::Dual, 2), 25);
}

TEST(Topology, CountsPeriodic3D)
{
    const auto M = test::make_manifold(3, 3, true);
    EXPECT_EQ(cells(M, Complex::Primal, 0), 27);
    EXPECT_EQ(cells(M, Complex::Primal, 1), 81);
    EXPECT_EQ(cells(M, Complex::Primal, 2), 81);
    EXPECT_EQ(cells(M, Complex::Primal, 3), 27);
}

TEST(Topology, CountsNonPeriodic3D)
{
    const auto M = test::make_manifold(3, 2, false);
    EXPECT_EQ(cells(M, Complex::Primal, 0), 27);
    EXPECT_EQ(cells(M, Complex::Primal, 1), 54);
    EXPECT_EQ(cells(M, Complex::Primal, 2), 36);
    EXPECT_EQ(cells(M, Complex::Primal, 3), 8);
}

TEST(Topology, StrataAreOrderedByDimensionThenMask)
{
    const auto all = stag::Topology<3>::all_strata();
    ASSERT_EQ(all.size(), 8u);
    const unsigned expected[8] = {0u, 1u, 2u, 4u, 3u, 5u, 6u, 7u};
    for (std::size_t i = 0; i < 8; ++i) EXPECT_EQ(all[i].axes, expected[i]);
    EXPECT_EQ(all[4].location, DMSTAG_BACK); // xy-face
    EXPECT_EQ(all[5].location, DMSTAG_DOWN); // xz-face
    EXPECT_EQ(all[6].location, DMSTAG_LEFT); // yz-face
}

TEST(Topology, IncidenceIsTransposeOfD)
{
    for (int n = 1; n <= 3; ++n) {
        const auto M = test::make_manifold(n, 4, false);
        for (int k = 1; k <= n; ++k) {
            Mat B = M->topology().incidence(CellSet{Complex::Primal, k});
            const auto d = ExteriorDerivative::create(M->space(k - 1, Orientation::Straight));
            Mat D = d->assemble(MATAIJ);
            Mat Dt;
            MatTranspose(D, MAT_INITIAL_MATRIX, &Dt);
            EXPECT_LT(test::mat_diff_norm(B, Dt), 1e-14) << "n=" << n << " k=" << k;
            MatDestroy(&B);
            MatDestroy(&D);
            MatDestroy(&Dt);
        }
    }
}

TEST(Topology, RejectsBadOptions)
{
    stag::Topology<2>::Options o;
    o.cells = {0, 3};
    EXPECT_THROW((void)stag::Topology<2>::create(PETSC_COMM_WORLD, o), Error);
}

TEST(Manifold, SpacesAreCached)
{
    const auto M = test::make_manifold(2, 4, true);
    const auto a = M->space(1, Orientation::Straight);
    const auto b = M->space(1, Orientation::Straight);
    EXPECT_EQ(a.get(), b.get());
    const auto c = M->space(1, Orientation::Twisted);
    EXPECT_NE(a.get(), c.get());
    EXPECT_EQ(a->dual_space()->type(), (FormType{1, Orientation::Twisted, Bundle::scalar()}));
    EXPECT_THROW((void)M->space(3, Orientation::Straight), TypeError);
}

TEST(Manifold, MappingPlacesForms)
{
    const auto M = test::make_manifold(2, 4, true, GridMapping::twisted_on_primal());
    EXPECT_EQ(M->space(0, Orientation::Straight)->cells(), (CellSet{Complex::Dual, 0}));
    EXPECT_EQ(M->space(0, Orientation::Twisted)->cells(), (CellSet{Complex::Primal, 0}));
    const auto* s = dynamic_cast<const stag::FormSpace<2>*>(M->space(0, Orientation::Straight).get());
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->storage_dimension(), 2);
}
