/**
 * @file structured_grid.hpp
 * @brief Logically rectangular grids backed by PETSc `DMStag`.
 */
#pragma once

#include <array>

#include "declib/grid/grid.hpp"

namespace declib {

/**
 * @brief Structured (staggered) grid backed by PETSc `DMStag`.
 *
 * `DMStag` stores unknowns on vertices, edges, faces and elements of a
 * Cartesian grid, which maps one-to-one onto primal \f$k\f$-cochains.
 *
 * @tparam Dim Spatial dimension (1, 2 or 3).
 *
 * @code{.cpp}
 * declib::StructuredGrid<2> grid({64, 64}, {1.0, 1.0});
 * auto n_edges = grid.num_cells(1);
 * @endcode
 */
template <int Dim>
class StructuredGrid final : public Grid {
    static_assert(Dim >= 1 && Dim <= 3, "StructuredGrid supports 1D, 2D and 3D");

public:
    /**
     * @brief Create a uniform grid on \f$[0, L_1] \times \dots \times [0, L_d]\f$.
     * @param cells   Number of elements in each direction.
     * @param lengths Physical extent \f$L_i\f$ in each direction.
     */
    StructuredGrid(std::array<int, Dim> cells, std::array<double, Dim> lengths);

    [[nodiscard]] int dimension() const noexcept override { return Dim; }
    [[nodiscard]] std::size_t num_cells(int k) const override;

    /// @brief Uniform spacing \f$h_i = L_i / N_i\f$ along axis @p axis.
    [[nodiscard]] double spacing(int axis) const;
};

} // namespace declib
