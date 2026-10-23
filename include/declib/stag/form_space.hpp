/**
 * @file form_space.hpp
 * @brief Form spaces on structured grids.
 */
#pragma once

#include <array>
#include <memory>

#include <petscdmstag.h>

#include "declib/core/types.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/stag/topology.hpp"

namespace declib::stag {

/**
 * @brief A FormSpace laid out on a compatible `DMStag`.
 *
 * Built with `DMStagCreateCompatibleDMStag` from the topology's base DM,
 * with \f$r\f$ dofs (the fiber rank) on exactly the strata named by
 * `mapping.storage_cells(k, orientation, n)` and none elsewhere. Vectors
 * have type `VECKOKKOS`.
 *
 * @tparam Dim Grid dimension (1, 2 or 3).
 */
template <int Dim>
class FormSpace final : public declib::FormSpace {
public:
    /// @brief Created by DiscreteManifold::space(); not called directly.
    FormSpace(std::shared_ptr<const DiscreteManifold> manifold, FormType type);

    /// @brief The structured topology.
    [[nodiscard]] const Topology<Dim>& topology() const noexcept;

    /// @brief Strata carrying values, in slot order.
    [[nodiscard]] std::span<const Stratum> strata() const noexcept;

    /// @brief Slot of fiber component @p comp on stratum number @p s in a local element block.
    [[nodiscard]] PetscInt slot(int s, int comp = 0) const;

    /// @brief Values per element block (strata times fiber rank).
    [[nodiscard]] PetscInt slots_per_element() const noexcept;

    [[nodiscard]] DM dm() const noexcept override;
    [[nodiscard]] std::size_t local_size() const override;
    [[nodiscard]] PetscInt global_size() const override;

private:
    DM dm_ = nullptr;
};

} // namespace declib::stag
