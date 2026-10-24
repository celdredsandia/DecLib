#include "declib/operators/pairing.hpp"

#include <bit>

#include "declib/core/error.hpp"
#include "declib/stag/detail/layout.hpp"
#include "declib/stag/orientation.hpp"
#include "declib/stag/topology.hpp"
#include "declib/topology/topology.hpp"
#include "declib/manifold/discrete_manifold.hpp"
#include "detail/petsc.hpp"

namespace declib {

Pairing::Pairing(SpacePtr space) : left_(std::move(space))
{
    right_ = left_->dual_space();
    signs_ = signs(*left_);
}

std::shared_ptr<const Pairing> Pairing::create(SpacePtr space)
{
    if (!space) throw Error("Pairing::create: null space");
    return std::shared_ptr<const Pairing>(new Pairing(std::move(space)));
}

Pairing::~Pairing() { VecDestroy(&signs_); }

const SpacePtr& Pairing::left() const noexcept { return left_; }

const SpacePtr& Pairing::right() const noexcept { return right_; }

Scalar Pairing::operator()(const Form& alpha, const Form& beta) const
{
    if (!alpha.space().same_type_as(*left_) || !beta.space().same_type_as(*right_))
        throw TypeError("Pairing: forms do not match the paired spaces");
    Vec t;
    DECLIB_PETSC(VecDuplicate(alpha.global_vec(), &t));
    DECLIB_PETSC(VecPointwiseMult(t, signs_, alpha.global_vec()));
    Scalar r;
    const PetscErrorCode ierr = VecDot(t, beta.global_vec(), &r);
    VecDestroy(&t);
    if (ierr) detail::throw_petsc(ierr, "VecDot", __FILE__, __LINE__);
    return r;
}

Vec Pairing::signs(const FormSpace& space)
{
    // alpha ^ beta on a cell pair (sigma, *sigma) is sign * dx^1 ^ ... ^ dx^n,
    // with sign the parity of moving the axes of the left form's cell in front
    // of the complementary axes.
    const int n = space.manifold().dimension();
    if (space.manifold().topology().backend() != Backend::Stag)
        throw Error("Pairing: only structured grids are supported");
    const bool primal = space.cells().complex == Complex::Primal;
    DM dm = space.dm();
    const auto layout = stag::detail::local_layout(dm);
    const auto slots = stag::detail::slots_of(dm, n);
    const int ks = primal ? space.type().degree : n - space.type().degree;
    const int r = space.type().bundle.rank();
    Vec loc, out;
    DECLIB_PETSC(DMCreateLocalVector(dm, &loc));
    DECLIB_PETSC(VecSet(loc, 0));
    PetscScalar* a;
    DECLIB_PETSC(VecGetArray(loc, &a));
    for (unsigned axes = 0; axes < (1u << n); ++axes) {
        if (std::popcount(axes) != ks) continue;
        const unsigned form_axes = primal ? axes : stag::complement(axes, n);
        const Scalar s = stag::complement_sign(form_axes, n);
        const int si = stag::detail::stratum_index(axes, n);
        const auto box = stag::detail::owned_cells(dm, axes);
        for (int k = box.lo[2]; k < box.hi[2]; ++k)
            for (int j = box.lo[1]; j < box.hi[1]; ++j)
                for (int i = box.lo[0]; i < box.hi[0]; ++i)
                    for (int c = 0; c < r; ++c) a[layout.index(i, j, k, slots.s[si][c])] = s;
    }
    DECLIB_PETSC(VecRestoreArray(loc, &a));
    DECLIB_PETSC(DMCreateGlobalVector(dm, &out));
    DECLIB_PETSC(DMLocalToGlobal(dm, loc, INSERT_VALUES, out));
    DECLIB_PETSC(VecDestroy(&loc));
    return out;
}

} // namespace declib
