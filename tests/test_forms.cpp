#include "common.hpp"

using namespace declib;

TEST(Form, BasicVectorOps)
{
    const auto M = test::make_manifold(2, 5, false);
    const auto V = M->space(1, Orientation::Straight);
    Form a = V->zeros();
    EXPECT_EQ(a.norm_max(), 0);
    a.fill(2);
    Form b = a.copy();
    b.scale(3);
    EXPECT_DOUBLE_EQ(b.norm_max(), 6);
    EXPECT_DOUBLE_EQ(a.norm_max(), 2);
    b.axpy(-3, a);
    EXPECT_EQ(b.norm_max(), 0);
    EXPECT_DOUBLE_EQ(a.norm2(), 2 * std::sqrt(static_cast<Real>(V->global_size())));

    const Form c = M->space(1, Orientation::Twisted)->zeros();
    EXPECT_THROW(a.axpy(1, c), TypeError);
}

TEST(Form, MoveAndView)
{
    const auto M = test::make_manifold(1, 8, true);
    const auto V = M->space(0, Orientation::Straight);
    Form a = V->zeros();
    a.fill(1);
    Form b = std::move(a);
    EXPECT_EQ(b.norm_max(), 1);
    Form v = Form::view_of(V, b.global_vec());
    v.scale(5);
    EXPECT_EQ(b.norm_max(), 5);
    Vec wrong;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 3, &wrong);
    EXPECT_THROW((void)Form::view_of(V, wrong), TypeError);
    VecDestroy(&wrong);
}

TEST(Form, LocalValuesFollowGlobal)
{
    const auto M = test::make_manifold(2, 6, true);
    const auto V = M->space(2, Orientation::Straight);
    Form a = V->zeros();
    a.fill(2);
    Real mx;
    VecMax(a.local_vec(), nullptr, &mx);
    EXPECT_EQ(mx, 2);
    a.fill(3);
    VecMax(a.local_vec(), nullptr, &mx);
    EXPECT_EQ(mx, 3);

    auto view = a.local_values_mut();
    Kokkos::deep_copy(view, 7.0);
    a.commit_local();
    Real mn;
    VecMin(a.global_vec(), nullptr, &mn);
    VecMax(a.global_vec(), nullptr, &mx);
    EXPECT_EQ(mn, 7);
    EXPECT_EQ(mx, 7);
    EXPECT_EQ(a.local_values().extent(0), view.extent(0));
}

TEST(Form, FormViewIndexesStrata)
{
    const auto M = test::make_manifold(2, 4, false);
    const auto V = M->space(1, Orientation::Straight);
    Form a = V->from_function([](std::span<const Real> x, std::span<Scalar> v) {
        v[0] = 1; // dx
        v[1] = 0; // dy
    });
    // Strata are numbered as in all_strata(): 1 = x-edges, 2 = y-edges.
    // x-edges carry the integral of dx, h = 1/4; y-edges carry 0.
    const auto& topo = dynamic_cast<const stag::Topology<2>&>(M->topology());
    const auto box = topo.owned_box();
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible) {
        stag::FormView<const Scalar, 2> fv(a);
        const int i = static_cast<int>(box.start[0]), j = static_cast<int>(box.start[1]);
        EXPECT_NEAR(fv(i, j, 1), 0.25, 1e-14);
        EXPECT_NEAR(fv(i, j, 2), 0.0, 1e-14);
    }
}

TEST(Form, DeRhamOfConstants)
{
    for (int n = 1; n <= 3; ++n) {
        const auto M = test::make_manifold(n, 4, true, GridMapping::straight_on_primal(), 2.0);
        // The volume form integrates to the total volume.
        const Form vol = M->space(n, Orientation::Straight)->from_function(
            [](std::span<const Real>, std::span<Scalar> v) { v[0] = 1; });
        Scalar total;
        VecSum(vol.global_vec(), &total);
        EXPECT_NEAR(total, std::pow(2.0, n), 1e-12);
        // So does the twisted volume form on the dual complex.
        const auto Md = test::make_manifold(n, 4, false, GridMapping::twisted_on_primal(), 2.0);
        const Form dvol = Md->space(n, Orientation::Straight)->from_function(
            [](std::span<const Real>, std::span<Scalar> v) { v[0] = 1; });
        VecSum(dvol.global_vec(), &total);
        EXPECT_NEAR(total, std::pow(2.0, n), 1e-12) << "n=" << n;
    }
}

TEST(Form, NaturalBoundaryDataIsNotImplementedYet)
{
    const auto M = test::make_manifold(2, 4, false);
    Form a = M->space(0, Orientation::Straight)->zeros();
    EXPECT_THROW(a.set_boundary_values(BoundaryRegion::all(), [](std::span<const Real>, std::span<Scalar>) {}),
                 Error);
}

TEST(Essential, BoundaryIndices)
{
    const PetscInt N = 5;
    const auto M = test::make_manifold(2, N, false);
    const auto V = M->space(0, Orientation::Straight);
    PetscInt size;
    IS is = V->boundary_indices(BoundaryRegion::all());
    ISGetSize(is, &size);
    EXPECT_EQ(size, 4 * N);
    ISDestroy(&is);
    is = V->boundary_indices(BoundaryRegion::face(1, Side::High));
    ISGetSize(is, &size);
    EXPECT_EQ(size, N + 1);
    ISDestroy(&is);
    // 1-forms: only the edges lying in the boundary.
    is = M->space(1, Orientation::Straight)->boundary_indices(BoundaryRegion::all());
    ISGetSize(is, &size);
    EXPECT_EQ(size, 4 * N);
    ISDestroy(&is);
}

TEST(Essential, Validation)
{
    const auto P = test::make_manifold(2, 4, true);
    EXPECT_THROW((void)P->space(0, Orientation::Straight)->with_essential({EssentialCondition{}}), TypeError);
    const auto M = test::make_manifold(2, 4, false);
    const auto V = M->space(0, Orientation::Straight);
    EXPECT_THROW((void)V->with_essential({EssentialCondition{BoundaryRegion::label("wall", 1), {}}}), TypeError);
    EXPECT_THROW((void)M->space(0, Orientation::Twisted)->with_essential({EssentialCondition{}}), TypeError);
    const auto Vc = V->with_essential({EssentialCondition{BoundaryRegion::face(0, Side::Low), {}}});
    EXPECT_EQ(Vc->essential_conditions().size(), 1u);
    EXPECT_TRUE(Vc->same_type_as(*V));
    EXPECT_EQ(V->essential_conditions().size(), 0u);
}
