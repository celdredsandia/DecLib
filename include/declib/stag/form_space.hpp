/**
 * @file form_space.hpp
 * @brief Form spaces on structured grids.
 */
#pragma once

#include <memory>
#include <span>
#include <vector>

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
    ~FormSpace() override;
    FormSpace& operator=(const FormSpace&) = delete;

    /// @brief The structured topology.
    [[nodiscard]] const Topology<Dim>& topology() const noexcept;

    /// @brief Primal dimension of the strata carrying values.
    [[nodiscard]] int storage_dimension() const noexcept;

    /// @brief Strata carrying values, ordered by axis bit mask.
    [[nodiscard]] std::span<const Stratum> strata() const;

    [[nodiscard]] DM dm() const noexcept override;
    [[nodiscard]] std::size_t local_size() const override;
    [[nodiscard]] PetscInt global_size() const override;
    [[nodiscard]] IS boundary_indices(const BoundaryRegion& region) const override;

protected:
    [[nodiscard]] SpacePtr clone_with(std::vector<EssentialCondition> conditions) const override;
    void validate_region(const BoundaryRegion& region) const override;

private:
    FormSpace(const FormSpace& other);

    const Topology<Dim>* topology_;
    DM dm_ = nullptr;
};

} // namespace declib::stag
