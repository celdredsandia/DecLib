/**
 * @file form_view.hpp
 * @brief Kokkos access to forms on structured grids.
 */
#pragma once

#include <array>
#include <type_traits>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"
#include "declib/stag/form_space.hpp"

namespace declib::stag {

/**
 * @brief Device view of a Form indexed by element, stratum and component.
 *
 * Wraps the ghosted local vector (`VecGetKokkosView`) as an unmanaged
 * `Kokkos::View` indexed \f$(k, j, i, \mathrm{slot})\f$ in `LayoutRight`,
 * the order of the `DMStag` local array. Slots are resolved once, when the
 * view is built, from `DMStagGetLocationSlot`. Indices are ghosted element
 * indices, so a kernel can read one stencil width outside its owned box.
 *
 * @tparam T   `Scalar` for a writable view, `const Scalar` for read-only.
 * @tparam Dim Grid dimension (1, 2 or 3).
 *
 * @code{.cpp}
 * declib::stag::FormView<const declib::Scalar, 2> u(form);
 * Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j) {
 *     auto du_x = u(i + 1, j, 0) - u(i, j, 0); // difference of vertex values along x
 * });
 * @endcode
 */
template <class T, int Dim>
class FormView {
    static_assert(std::is_same_v<std::remove_const_t<T>, Scalar>, "FormView holds Scalar values");

public:
    /// @brief Read-only view; refreshes ghosts if stale.
    explicit FormView(const Form& form)
        requires std::is_const_v<T>;

    /// @brief Writable view; marks ghosts stale.
    explicit FormView(Form& form)
        requires(!std::is_const_v<T>);

    /// @brief Value of component @p comp on stratum @p s of element \f$(i)\f$ (1D).
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int s, int comp = 0) const
        requires(Dim == 1);

    /// @brief Value of component @p comp on stratum @p s of element \f$(i, j)\f$ (2D).
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int j, int s, int comp = 0) const
        requires(Dim == 2);

    /// @brief Value of component @p comp on stratum @p s of element \f$(i, j, k)\f$ (3D).
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int j, int k, int s, int comp = 0) const
        requires(Dim == 3);

private:
    Kokkos::View<T****, Kokkos::LayoutRight, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> data_;
    Kokkos::View<const int**, MemSpace> slots_;
    std::array<int, Dim> ghost_start_{};
};

} // namespace declib::stag
