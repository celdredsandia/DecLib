/**
 * @file grid.hpp
 * @brief Abstract mesh interface shared by structured and unstructured grids.
 */
#pragma once

#include <cstddef>

namespace declib {

/**
 * @brief Abstract interface of a discrete manifold (a cell complex).
 *
 * A Grid owns a PETSc `DM` and exposes the counts and orientations that the
 * DEC operators need. Concrete implementations are StructuredGrid (backed by
 * `DMStag`) and UnstructuredGrid (backed by `DMPlex`).
 *
 * For a complex of dimension \f$n\f$, a \f$k\f$-cell exists for every
 * \f$0 \le k \le n\f$: vertices (\f$k=0\f$), edges (\f$k=1\f$), faces
 * (\f$k=2\f$) and volumes (\f$k=3\f$).
 */
class Grid {
public:
    virtual ~Grid() = default;

    /// @brief Topological dimension \f$n\f$ of the complex.
    [[nodiscard]] virtual int dimension() const noexcept = 0;

    /**
     * @brief Number of locally owned \f$k\f$-cells.
     * @param k Cell degree, \f$0 \le k \le n\f$.
     * @return Count of cells of degree @p k owned by this MPI rank.
     */
    [[nodiscard]] virtual std::size_t num_cells(int k) const = 0;
};

} // namespace declib
