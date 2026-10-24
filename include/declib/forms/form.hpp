/**
 * @file form.hpp
 * @brief A discrete differential form: an element of a FormSpace.
 */
#pragma once

#include <memory>

#include <petscvec.h>

#include "declib/core/types.hpp"
#include "declib/forms/boundary_condition.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/forms/form_type.hpp"

namespace declib {

/**
 * @brief A discrete form (cochain): one fiber value per cell of its space.
 *
 * Values live in a PETSc `Vec` of type `VECKOKKOS` laid out by the space's
 * `DM`. A Form owns its vectors: copying is explicit (copy()), moving is
 * cheap. Ghost values are refreshed lazily: writing through global_vec()
 * or the vector operations marks them stale, and the next read-only local
 * access exchanges them.
 *
 * Backends provide structured accessors on top of local_values(), for
 * example declib::stag::FormView.
 */
class Form {
public:
    /// @brief A zero form in @p space.
    explicit Form(SpacePtr space);

    /**
     * @brief A form that uses @p vec as its global vector without owning it.
     *
     * @p vec must have the layout of @p space and outlive the form. Used to
     * hand PETSc vectors (for example inside a `MATSHELL` or a `KSP`) to
     * operators without copying.
     */
    [[nodiscard]] static Form view_of(SpacePtr space, Vec vec);

    Form(Form&&) noexcept;            ///< Move; leaves the source empty.
    Form& operator=(Form&&) noexcept; ///< Move assignment.
    Form(const Form&) = delete;       ///< Use copy() instead.
    Form& operator=(const Form&) = delete;
    ~Form();

    /// @brief The space this form belongs to.
    [[nodiscard]] const FormSpace& space() const noexcept;

    /// @brief Shared handle to the space.
    [[nodiscard]] const SpacePtr& space_ptr() const noexcept;

    /// @brief Shorthand for `space().type()`.
    [[nodiscard]] const FormType& type() const noexcept;

    /// @brief Owned values, for PETSc solvers. Marks ghosts stale.
    [[nodiscard]] Vec global_vec() noexcept;

    /// @brief Owned values, read-only.
    [[nodiscard]] Vec global_vec() const noexcept;

    /// @brief Ghosted local vector, refreshed if stale.
    [[nodiscard]] Vec local_vec() const;

    /// @brief Exchange ghost values now (`DMGlobalToLocal`).
    void update_ghosts() const;

    /// @brief Ghosted local values on device, refreshed if stale.
    [[nodiscard]] Kokkos::View<const Scalar*, MemSpace> local_values() const;

    /**
     * @brief Ghosted local values on device, writable.
     *
     * After writing, call commit_local() to copy the owned values to the
     * global vector.
     */
    [[nodiscard]] Kokkos::View<Scalar*, MemSpace> local_values_mut();

    /// @brief Copy owned local values to the global vector (`DMLocalToGlobal`).
    void commit_local();

    /**
     * @brief Prescribe natural boundary data.
     *
     * Sets the values on the extended dual boundary cells in @p region to
     * the de Rham map of @p omega. Valid only for forms hosted on the dual
     * complex of a topology with DualBoundary::Extended.
     *
     * @warning Not implemented yet: operators treat natural boundary data as
     *          zero. Calling this throws Error.
     */
    void set_boundary_values(const BoundaryRegion& region, const FormFunction& omega);

    /// @brief \f$\mathrm{this} \leftarrow \mathrm{this} + a x\f$. Requires the same layout.
    Form& axpy(Scalar a, const Form& x);

    /// @brief \f$\mathrm{this} \leftarrow a \cdot \mathrm{this}\f$.
    Form& scale(Scalar a);

    /// @brief Set every value to @p a.
    Form& fill(Scalar a);

    /// @brief A deep copy in the same space.
    [[nodiscard]] Form copy() const;

    /// @brief Euclidean 2-norm of the coefficient vector.
    [[nodiscard]] Real norm2() const;

    /// @brief Maximum absolute coefficient.
    [[nodiscard]] Real norm_max() const;

private:
    Form(SpacePtr space, Vec vec, bool owned);

    SpacePtr space_;
    Vec global_ = nullptr;
    bool owns_global_ = true;
    mutable Vec local_ = nullptr;
    mutable PetscObjectState local_state_ = -1;
};

} // namespace declib
