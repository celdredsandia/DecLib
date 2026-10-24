/**
 * @file connection.hpp
 * @brief Discrete connections for vector-bundle valued forms.
 */
#pragma once

#include <petscvec.h>

#include "declib/core/types.hpp"
#include "declib/forms/bundle.hpp"
#include "declib/topology/cell_set.hpp"

namespace declib {

/**
 * @brief Parallel transport matrices for every incidence pair of a cell set.
 *
 * For each \f$k\f$-cell \f$\sigma\f$ and each \f$(k-1)\f$-cell \f$\tau\f$ in
 * its boundary, stores \f$U(\tau \to \sigma) \in GL(r)\f$, a dense
 * \f$r \times r\f$ matrix, in the order of the incidence matrix entries.
 */
class TransportField {
public:
    virtual ~TransportField() = default;

    /// @brief The \f$k\f$-cells whose boundary incidences are stored.
    [[nodiscard]] virtual CellSet cells() const noexcept = 0;

    /// @brief Fiber rank \f$r\f$.
    [[nodiscard]] virtual int rank() const noexcept = 0;

    /// @brief Matrices stored row-major, \f$r^2\f$ values per incidence pair.
    [[nodiscard]] virtual Vec local_vec() const noexcept = 0;
};

/**
 * @brief A discrete connection on a vector bundle.
 *
 * Needed by the covariant exterior derivative \f$d^\nabla\f$ on bundles
 * whose frame varies from cell to cell. The trivial bundle with a global
 * frame uses the identity connection, for which \f$d^\nabla = d\f$ acting
 * componentwise.
 *
 * @note Planned for phase 3.
 */
class Connection {
public:
    virtual ~Connection() = default;

    /// @brief The bundle this connection acts on.
    [[nodiscard]] virtual const Bundle& bundle() const noexcept = 0;

    /// @brief Transport from \f$(k-1)\f$-cells into incident \f$k\f$-cells.
    [[nodiscard]] virtual const TransportField& transport(CellSet k_cells) const = 0;

    /// @brief True when the curvature vanishes, so \f$d^\nabla \circ d^\nabla = 0\f$.
    [[nodiscard]] virtual bool is_flat() const noexcept = 0;
};

} // namespace declib
