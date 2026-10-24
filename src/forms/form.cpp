// petscvec_kokkos.hpp must precede every other PETSc header.
#include <petscvec_kokkos.hpp>

#include "declib/forms/form.hpp"


#include <petsc/private/petscimpl.h> // PetscObjectStateGet

#include "declib/core/error.hpp"
#include "detail/petsc.hpp"

namespace declib {

Form::Form(SpacePtr space) : space_(std::move(space))
{
    if (!space_) throw Error("Form: null space");
    DECLIB_PETSC(DMCreateGlobalVector(space_->dm(), &global_));
    DECLIB_PETSC(VecSet(global_, 0));
}

Form::Form(SpacePtr space, Vec vec, bool owned) : space_(std::move(space)), global_(vec), owns_global_(owned) {}

Form Form::view_of(SpacePtr space, Vec vec)
{
    if (!space || !vec) throw Error("Form::view_of: null space or vector");
    PetscInt n, nl;
    DECLIB_PETSC(VecGetSize(vec, &n));
    DECLIB_PETSC(VecGetLocalSize(vec, &nl));
    if (n != space->global_size() || static_cast<std::size_t>(nl) != space->local_size())
        throw TypeError("Form::view_of: vector layout does not match the space");
    return Form(std::move(space), vec, false);
}

Form::Form(Form&& other) noexcept
    : space_(std::move(other.space_)), global_(other.global_), owns_global_(other.owns_global_),
      local_(other.local_), local_state_(other.local_state_)
{
    other.global_ = nullptr;
    other.local_ = nullptr;
    other.local_state_ = -1;
}

Form& Form::operator=(Form&& other) noexcept
{
    if (this != &other) {
        VecDestroy(&local_);
        if (owns_global_) VecDestroy(&global_);
        space_ = std::move(other.space_);
        global_ = other.global_;
        owns_global_ = other.owns_global_;
        local_ = other.local_;
        local_state_ = other.local_state_;
        other.global_ = nullptr;
        other.local_ = nullptr;
        other.local_state_ = -1;
    }
    return *this;
}

Form::~Form()
{
    VecDestroy(&local_);
    if (owns_global_) VecDestroy(&global_);
}

const FormSpace& Form::space() const noexcept { return *space_; }

const SpacePtr& Form::space_ptr() const noexcept { return space_; }

const FormType& Form::type() const noexcept { return space_->type(); }

Vec Form::global_vec() noexcept { return global_; }

Vec Form::global_vec() const noexcept { return global_; }

Vec Form::local_vec() const
{
    if (!local_) {
        DECLIB_PETSC(DMCreateLocalVector(space_->dm(), &local_));
        // Points outside the domain and DMStag's dummy points are never
        // written by a scatter, so they stay zero.
        DECLIB_PETSC(VecSet(local_, 0));
    }
    PetscObjectState state;
    DECLIB_PETSC(PetscObjectStateGet(reinterpret_cast<PetscObject>(global_), &state));
    if (state != local_state_) {
        DECLIB_PETSC(DMGlobalToLocal(space_->dm(), global_, INSERT_VALUES, local_));
        local_state_ = state;
    }
    return local_;
}

void Form::update_ghosts() const
{
    local_state_ = -1;
    (void)local_vec();
}

Kokkos::View<const Scalar*, MemSpace> Form::local_values() const
{
    Vec l = local_vec();
    Kokkos::View<const Scalar*, MemSpace> v;
    DECLIB_PETSC(VecGetKokkosView(l, &v));
    DECLIB_PETSC(VecRestoreKokkosView(l, &v));
    return v;
}

Kokkos::View<Scalar*, MemSpace> Form::local_values_mut()
{
    Vec l = local_vec();
    Kokkos::View<Scalar*, MemSpace> v;
    // Restoring right away marks the device copy as the current one; the
    // view aliases it until commit_local() scatters it back.
    DECLIB_PETSC(VecGetKokkosView(l, &v));
    DECLIB_PETSC(VecRestoreKokkosView(l, &v));
    return v;
}

void Form::commit_local()
{
    if (!local_) return;
    DECLIB_PETSC(DMLocalToGlobal(space_->dm(), local_, INSERT_VALUES, global_));
    local_state_ = -1;
}

void Form::set_boundary_values(const BoundaryRegion&, const FormFunction&)
{
    throw Error("Form::set_boundary_values: natural boundary data is not implemented yet (phase 2); "
                "operators currently use homogeneous data on the extended dual cells");
}

Form& Form::axpy(Scalar a, const Form& x)
{
    if (!x.space().same_type_as(*space_)) throw TypeError("Form::axpy: forms of different types");
    DECLIB_PETSC(VecAXPY(global_, a, x.global_));
    return *this;
}

Form& Form::scale(Scalar a)
{
    DECLIB_PETSC(VecScale(global_, a));
    return *this;
}

Form& Form::fill(Scalar a)
{
    DECLIB_PETSC(VecSet(global_, a));
    return *this;
}

Form Form::copy() const
{
    Form f(space_);
    DECLIB_PETSC(VecCopy(global_, f.global_));
    return f;
}

Real Form::norm2() const
{
    Real r;
    DECLIB_PETSC(VecNorm(global_, NORM_2, &r));
    return r;
}

Real Form::norm_max() const
{
    Real r;
    DECLIB_PETSC(VecNorm(global_, NORM_INFINITY, &r));
    return r;
}

} // namespace declib
