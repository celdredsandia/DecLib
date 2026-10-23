/**
 * @file cochain.hpp
 * @brief Discrete differential forms (cochains).
 */
#pragma once

#include "declib/grid/grid.hpp"

namespace declib {

/// @brief Whether a cochain lives on the primal or the dual complex.
enum class Complex {
    Primal, ///< Values on primal \f$k\f$-cells.
    Dual    ///< Values on dual \f$(n-k)\f$-cells.
};

/**
 * @brief A \f$k\f$-cochain: one scalar per oriented \f$k\f$-cell.
 *
 * A cochain is the discrete counterpart of a differential \f$k\f$-form
 * \f$\omega\f$, obtained through the de Rham map
 * \f$\omega \mapsto \left(\int_{\sigma} \omega\right)_{\sigma}\f$.
 * Values are stored in a PETSc `Vec` whose local array is accessed through
 * Kokkos views, so operators run on any Kokkos backend.
 *
 * @tparam Degree Form degree \f$k\f$.
 * @tparam Kind   Primal or dual complex.
 */
template <int Degree, Complex Kind = Complex::Primal>
class Cochain {
public:
    /// @brief Form degree \f$k\f$.
    static constexpr int degree = Degree;

    /// @brief Allocate a zero-initialised cochain on @p grid.
    explicit Cochain(const Grid& grid);

    /// @brief The grid this cochain is defined on.
    [[nodiscard]] const Grid& grid() const noexcept;

    /// @brief Number of degrees of freedom (locally owned cells).
    [[nodiscard]] std::size_t size() const;
};

} // namespace declib
