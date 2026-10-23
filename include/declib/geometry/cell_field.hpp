/**
 * @file cell_field.hpp
 * @brief Geometric data stored per cell (measures, coordinates).
 */
#pragma once

#include <petscvec.h>

#include "declib/core/types.hpp"
#include "declib/topology/cell_set.hpp"

namespace declib {

/**
 * @brief A fixed number of real values attached to every cell of a CellSet.
 *
 * Geometry stores its measures and coordinates as CellField objects. They
 * share the layouts, halos and Kokkos access paths of forms, but carry no
 * orientation and are not acted on by DEC operators.
 */
class CellField {
public:
    virtual ~CellField() = default;

    /// @brief The cells this field is attached to.
    [[nodiscard]] virtual CellSet cells() const noexcept = 0;

    /// @brief Values per cell (1 for measures, \f$n\f$ for coordinates).
    [[nodiscard]] virtual int components() const noexcept = 0;

    /// @brief Ghosted local values as a PETSc `Vec` of type `VECKOKKOS`.
    [[nodiscard]] virtual Vec local_vec() const noexcept = 0;
};

} // namespace declib
