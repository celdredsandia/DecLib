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
 * cheap. Ghost values are refreshed lazily: writing through
 * local_values_mut() or global_vec() marks them stale, and the next
 * read-only access exchanges them.
 *
 * Backends provide structured accessors on top of local_values(), for
 * example declib::stag::FormView.
 */
class Form {
public:
    /// @brief A zero form in @p space.
    explicit Form(SpacePtr space);

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

    /// @brief Exchange ghost values now (`DMGlobalToLocal`).
    void update_ghosts() const;

    /// @brief Ghosted local values on device, refreshed if stale.
    [[nodiscard]] Kokkos::View<const Scalar*, MemSpace> local_values() const;

    /// @brief Ghosted local values on device, writable. Marks ghosts stale.
    [[nodiscard]] Kokkos::View<Scalar*, MemSpace> local_values_mut();

    /**
     * @brief Prescribe natural boundary data.
     *
     * Sets the values on the extended dual boundary cells in @p region to
     * the de Rham map of @p omega. Valid only for forms hosted on the dual
     * complex of a topology with DualBoundary::Extended.
     *
     * @throws TypeError if the form is not hosted on an extended dual complex.
     */
    void set_boundary_values(const BoundaryRegion& region, const FormFunction& omega);

    /// @brief \f$\mathrm{this} \leftarrow \mathrm{this} + a x\f$. Requires the same space.
    Form& axpy(Scalar a, const Form& x);

    /// @brief \f$\mathrm{this} \leftarrow a \cdot \mathrm{this}\f$.
    Form& scale(Scalar a);

    /// @brief Set every value to @p a.
    Form& fill(Scalar a);

    /// @brief A deep copy in the same space.
    [[nodiscard]] Form copy() const;

private:
    SpacePtr space_;
    Vec global_ = nullptr;
    mutable Vec local_ = nullptr;
    mutable bool ghosts_stale_ = true;
};

} // namespace declib
