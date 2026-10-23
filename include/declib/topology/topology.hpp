/**
 * @file topology.hpp
 * @brief Abstract interface of an oriented cell complex and its dual.
 */
#pragma once

#include <cstddef>
#include <memory>

#include <mpi.h>
#include <petscmat.h>

#include "declib/core/types.hpp"
#include "declib/topology/cell_set.hpp"

namespace declib {

/**
 * @brief An oriented cell complex of dimension \f$n\f$, together with its
 *        dual complex and parallel decomposition.
 *
 * A Topology knows cell counts, incidence (the boundary operators
 * \f$\partial_k\f$) and orientations. It knows nothing about coordinates,
 * lengths, areas or volumes; those belong to Geometry.
 *
 * Dual \f$k\f$-cells are always indexed by the primal \f$(n-k)\f$-cell they
 * cross, on every backend, so that GridMapping needs no backend logic.
 *
 * Topologies are immutable after creation and shared through
 * `std::shared_ptr<const Topology>`.
 *
 * Invariants checked by the test suite:
 * - \f$\partial_k \circ \partial_{k+1} = 0\f$ exactly;
 * - cell counts satisfy the Euler characteristic of the domain.
 */
class Topology : public std::enable_shared_from_this<Topology> {
public:
    virtual ~Topology() = default;

    /// @brief Topological dimension \f$n\f$ of the complex.
    [[nodiscard]] virtual int dimension() const noexcept = 0;

    /// @brief Communicator over which the complex is distributed.
    [[nodiscard]] virtual MPI_Comm comm() const noexcept = 0;

    /// @brief Backend implementing this topology.
    [[nodiscard]] virtual Backend backend() const noexcept = 0;

    /// @brief Number of cells of @p cells owned by this rank.
    [[nodiscard]] virtual std::size_t num_owned(CellSet cells) const = 0;

    /// @brief Number of cells of @p cells on this rank, ghosts included.
    [[nodiscard]] virtual std::size_t num_local(CellSet cells) const = 0;

    /// @brief Number of cells of @p cells over all ranks.
    [[nodiscard]] virtual PetscInt num_global(CellSet cells) const = 0;

    /**
     * @brief The primal partner of a cell set.
     * @return `{opposite(c.complex), dimension() - c.k}`: dual \f$k\f$-cells
     *         correspond one to one with primal \f$(n-k)\f$-cells.
     */
    [[nodiscard]] CellSet dual_of(CellSet c) const noexcept;

    /**
     * @brief Signed incidence matrix \f$\partial_k : C_k \to C_{k-1}\f$.
     *
     * Assembled on demand. Used by tests and by the generic sparse operator
     * path; structured kernels never call it.
     *
     * @param cells The \f$k\f$-cells, \f$1 \le k \le n\f$, on either complex.
     * @return A new PETSc `Mat` owned by the caller.
     */
    [[nodiscard]] virtual Mat incidence(CellSet cells) const = 0;

    /// @brief How the dual complex is closed at non-periodic boundaries.
    [[nodiscard]] virtual DualBoundary dual_boundary() const noexcept = 0;

    /// @brief Halo width in cells, fixed at creation by the widest operator stencil.
    [[nodiscard]] virtual int stencil_width() const noexcept = 0;
};

} // namespace declib
