/**
 * @file form_view.hpp
 * @brief Kokkos access to forms on structured grids.
 */
#pragma once

#include <array>
#include <type_traits>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"
#include "declib/stag/detail/layout.hpp"
#include "declib/stag/form_space.hpp"

namespace declib::stag {

/**
 * @brief Device view of a Form indexed by element, stratum and component.
 *
 * Wraps the ghosted local vector of the form. Indices are global element
 * indices; a kernel may read up to the stencil width outside the owned box.
 * Strata are numbered as in Topology::all_strata(). Slots are resolved once,
 * when the view is built.
 *
 * A read-only view refreshes ghost values first. After writing through a
 * writable view, call Form::commit_local() to copy owned values back to the
 * global vector.
 *
 * @tparam T   `Scalar` for a writable view, `const Scalar` for read-only.
 * @tparam Dim Grid dimension (1, 2 or 3).
 *
 * @code{.cpp}
 * declib::stag::FormView<const declib::Scalar, 2> u(form);
 * Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j) {
 *     auto du_x = u(i + 1, j, 0) - u(i, j, 0); // vertex values along x
 * });
 * @endcode
 */
template <class T, int Dim>
class FormView {
    static_assert(std::is_same_v<std::remove_const_t<T>, Scalar>, "FormView holds Scalar values");

public:
    /// @brief Read-only view; refreshes ghosts if stale.
    explicit FormView(const Form& form)
        requires std::is_const_v<T>
        : FormView(form.local_values(), form)
    {}

    /// @brief Writable view.
    explicit FormView(Form& form)
        requires(!std::is_const_v<T>)
        : FormView(form.local_values_mut(), form)
    {}

    /// @brief Value of component @p comp on stratum @p s of element \f$(i)\f$ (1D).
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int s, int comp = 0) const
        requires(Dim == 1)
    {
        return data_(layout_.index(i, 0, 0, slots_.s[s][comp]));
    }

    /// @brief Value of component @p comp on stratum @p s of element \f$(i, j)\f$ (2D).
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int j, int s, int comp = 0) const
        requires(Dim == 2)
    {
        return data_(layout_.index(i, j, 0, slots_.s[s][comp]));
    }

    /// @brief Value of component @p comp on stratum @p s of element \f$(i, j, k)\f$ (3D).
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int j, int k, int s, int comp = 0) const
        requires(Dim == 3)
    {
        return data_(layout_.index(i, j, k, slots_.s[s][comp]));
    }

private:
    FormView(Kokkos::View<T*, MemSpace> data, const Form& form)
        : data_(data), layout_(detail::local_layout(form.space().dm())),
          slots_(detail::slots_of(form.space().dm(), Dim))
    {}

    Kokkos::View<T*, MemSpace> data_;
    detail::LocalLayout layout_;
    detail::Slots slots_;
};

} // namespace declib::stag
