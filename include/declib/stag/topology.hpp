/**
 * @file topology.hpp
 * @brief Topologically structured cell complexes on PETSc `DMStag`.
 */
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <petscdmstag.h>

#include "declib/core/types.hpp"
#include "declib/stag/index_box.hpp"
#include "declib/topology/topology.hpp"

namespace declib::stag {

/// @brief Boundary type along one axis. Maps to PETSc `DMBoundaryType`.
enum class Boundary {
    Periodic, ///< `DM_BOUNDARY_PERIODIC`: the dual complex is closed.
    Ghosted,  ///< Non-periodic, with a ghost layer (`DM_BOUNDARY_GHOSTED`).
    None      ///< Non-periodic. Stored like Ghosted, since operators read one cell past the boundary.
};

/**
 * @brief One family of cells of a given dimension and axis set.
 *
 * In 3D the primal 1-cells form three strata (x-, y- and z-edges). Each
 * stratum is one `DMStag` location per element. A primal cell of stratum
 * @ref axes at element index \f$(i, j, k)\f$ spans
 * \f$[x_i, x_{i+1}]\f$ along each axis in @ref axes and sits on the plane
 * \f$x_i\f$ along the others.
 */
struct Stratum {
    DMStagStencilLocation location; ///< Where the cell sits relative to its element, e.g. `DMSTAG_LEFT`.
    int k;                          ///< Primal cell dimension, the number of bits set in @ref axes.
    unsigned axes;                  ///< Bit \f$i\f$ set when the cell extends along axis \f$i\f$.
};

/**
 * @brief The logically Cartesian product complex in 1, 2 or 3 dimensions.
 *
 * Cells are identified by an element index \f$(i, j, k)\f$ and a Stratum.
 * The dual of a cubical complex is again cubical, and every dual cell is
 * stored on the stratum of the primal cell it crosses, so one `DMStag`
 * decomposition serves primal and dual forms alike:
 *
 * | Primal cell (3D)    | `DMStag` location                    | Dual cell stored there |
 * |---------------------|--------------------------------------|------------------------|
 * | vertex              | `BACK_DOWN_LEFT`                     | dual 3-cell            |
 * | x-, y-, z-edge      | `BACK_DOWN`, `BACK_LEFT`, `DOWN_LEFT`| dual 2-cell            |
 * | xy-, xz-, yz-face   | `BACK`, `DOWN`, `LEFT`               | dual 1-cell            |
 * | element             | `ELEMENT`                            | dual 0-cell            |
 *
 * On a non-periodic axis with \f$N\f$ elements there are \f$N+1\f$ vertex
 * planes. Dual cells crossing the boundary planes are half cells. With
 * DualBoundary::Extended the dual complex also has cells on the boundary
 * itself; they are not degrees of freedom of any form space, and their
 * values (natural boundary data) are zero unless set.
 *
 * @tparam Dim Grid dimension (1, 2 or 3).
 */
template <int Dim>
class Topology final : public declib::Topology {
    static_assert(Dim >= 1 && Dim <= 3, "stag::Topology supports 1D, 2D and 3D");

public:
    /// @brief Creation options.
    struct Options {
        std::array<PetscInt, Dim> cells{};                   ///< Elements per axis.
        std::array<Boundary, Dim> boundary{};                ///< Boundary type per axis; default Periodic.
        DualBoundary dual_boundary = DualBoundary::Extended; ///< Dual closure on non-periodic axes.
        int stencil_width = 1;                               ///< Halo width in elements, at least 1.
        std::array<PetscInt, Dim> ranks{};                   ///< Ranks per axis; 0 lets PETSc decide.
    };

    /**
     * @brief Create the complex and its base `DMStag`.
     * @throws Error if a cell count is smaller than 1 or a periodic axis has fewer cells than the stencil width.
     */
    [[nodiscard]] static std::shared_ptr<const Topology> create(MPI_Comm comm, const Options& options);

    ~Topology() override;
    Topology(const Topology&) = delete;
    Topology& operator=(const Topology&) = delete;

    /**
     * @brief The base `DMStag`, with one dof on every stratum.
     *
     * Form spaces derive their own layouts from it with
     * `DMStagCreateCompatibleDMStag`, so all spaces share one decomposition.
     */
    [[nodiscard]] DM base_dm() const noexcept;

    /// @brief Elements owned by this rank.
    [[nodiscard]] IndexBox<Dim> owned_box() const;

    /// @brief Elements on this rank, ghosts included.
    [[nodiscard]] IndexBox<Dim> ghosted_box() const;

    /// @brief Creation options, including cell counts and boundary types.
    [[nodiscard]] const Options& options() const noexcept;

    /// @brief True if @p axis is periodic.
    [[nodiscard]] bool periodic(int axis) const;

    /**
     * @brief The strata that make up a cell set.
     *
     * Dual cell sets return the strata of their primal partners, see
     * declib::Topology::dual_of(). Strata are ordered by their axis bit mask.
     */
    [[nodiscard]] std::span<const Stratum> strata(CellSet cells) const;

    /// @brief All \f$2^{\mathrm{Dim}}\f$ strata, ordered by dimension and then by axis bit mask.
    [[nodiscard]] static std::span<const Stratum> all_strata() noexcept;

    /**
     * @brief Create a `DMStag` compatible with base_dm() with @p dof values
     *        on every stratum of primal dimension @p k and none elsewhere.
     * @return A new `DM` owned by the caller.
     */
    [[nodiscard]] DM create_compatible_dm(int k, int dof) const;

    [[nodiscard]] int dimension() const noexcept override { return Dim; }
    [[nodiscard]] MPI_Comm comm() const noexcept override;
    [[nodiscard]] Backend backend() const noexcept override { return Backend::Stag; }
    [[nodiscard]] std::size_t num_owned(CellSet cells) const override;
    [[nodiscard]] std::size_t num_local(CellSet cells) const override;
    [[nodiscard]] PetscInt num_global(CellSet cells) const override;
    [[nodiscard]] Mat incidence(CellSet cells) const override;
    [[nodiscard]] DualBoundary dual_boundary() const noexcept override;
    [[nodiscard]] int stencil_width() const noexcept override;

private:
    Topology(MPI_Comm comm, const Options& options);

    Options options_;
    DM dm_ = nullptr;
};

} // namespace declib::stag
