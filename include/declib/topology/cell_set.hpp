/**
 * @file cell_set.hpp
 * @brief Identifiers for families of cells and for the dual boundary policy.
 */
#pragma once

#include "declib/core/types.hpp"

namespace declib {

/**
 * @brief All \f$k\f$-cells of one complex.
 *
 * Structured backends refine a CellSet into per-axis strata (for example
 * x-edges and y-edges); see declib::stag::Topology::strata().
 */
struct CellSet {
    Complex complex; ///< Primal or dual complex.
    int k;           ///< Cell dimension, \f$0 \le k \le n\f$.

    /// @brief Equality of complex and dimension.
    friend constexpr bool operator==(const CellSet&, const CellSet&) = default;
};

/**
 * @brief How the dual complex is closed on a non-periodic boundary.
 *
 * On periodic grids the dual complex is closed and both policies coincide.
 */
enum class DualBoundary {
    /**
     * Dual cells are clipped at \f$\partial M\f$ and the dual complex has no
     * cells on the boundary itself. The dual exterior derivative then needs
     * boundary closure terms.
     */
    Truncated,
    /**
     * The dual complex also contains cells lying on \f$\partial M\f$. The
     * discrete Stokes theorem then holds on the dual complex including
     * boundary terms, and natural boundary conditions are imposed through
     * these cells. This is the default.
     */
    Extended
};

} // namespace declib
