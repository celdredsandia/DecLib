/**
 * @file form_space.hpp
 * @brief The discrete space of forms of one FormType.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <petscdm.h>
#include <petscis.h>

#include "declib/core/types.hpp"
#include "declib/forms/boundary_condition.hpp"
#include "declib/forms/form_type.hpp"
#include "declib/geometry/geometry.hpp"
#include "declib/topology/cell_set.hpp"

namespace declib {

class DiscreteManifold;
class Form;
class FormSpace;

/// @brief Shared pointer to an immutable FormSpace.
using SpacePtr = std::shared_ptr<const FormSpace>;

/**
 * @brief The space of discrete forms of one FormType on a DiscreteManifold.
 *
 * A FormSpace plays the role of a PETSc `DM`: it fixes the layout, ghost
 * exchange and parallel decomposition, while Form plays the role of a
 * `Vec`. Spaces are created by DiscreteManifold::space() and are immutable.
 *
 * Two spaces are interchangeable for operators when they belong to the same
 * manifold and have the same FormType (see same_type_as()); essential
 * conditions do not change the layout.
 */
class FormSpace : public std::enable_shared_from_this<FormSpace> {
public:
    virtual ~FormSpace() = default;

    /// @brief The manifold this space is defined on.
    [[nodiscard]] const DiscreteManifold& manifold() const noexcept;

    /// @brief Shared handle to the manifold.
    [[nodiscard]] const std::shared_ptr<const DiscreteManifold>& manifold_ptr() const noexcept;

    /// @brief Degree, orientation and fiber of the forms in this space.
    [[nodiscard]] const FormType& type() const noexcept;

    /// @brief Cells the values live on, from the manifold's GridMapping.
    [[nodiscard]] CellSet cells() const noexcept;

    /// @brief True if @p other has the same manifold and FormType.
    [[nodiscard]] bool same_type_as(const FormSpace& other) const noexcept;

    /// @brief Layout `DM`: a `DMStag` on structured grids.
    [[nodiscard]] virtual DM dm() const noexcept = 0;

    /// @brief Number of locally owned values (cells times fiber rank).
    [[nodiscard]] virtual std::size_t local_size() const = 0;

    /// @brief Number of values over all ranks.
    [[nodiscard]] virtual PetscInt global_size() const = 0;

    /// @brief A new form with all values zero.
    [[nodiscard]] Form zeros() const;

    /// @brief A new form holding the de Rham map of @p omega.
    [[nodiscard]] Form from_function(const FormFunction& omega, Quadrature quad = {}) const;

    /**
     * @brief The space this one pairs with.
     *
     * Same manifold, degree \f$n-k\f$, opposite orientation, dual bundle.
     * The duality pairing \f$V \times V^* \to \mathbb{R}\f$ between the two
     * is metric-free, and is where variational derivatives and pairing
     * adjoints take their values. Both spaces store their values on the same
     * primal cells, with the same layout.
     */
    [[nodiscard]] SpacePtr dual_space() const;

    /**
     * @brief A copy of this space with essential boundary conditions.
     *
     * The returned space has the same type and layout. Its conditions are
     * imposed on assembled systems by apply_essential().
     *
     * @throws TypeError if the forms of this space have no cells on the
     *         boundary (for example forms hosted on the dual complex), or if
     *         a region names a periodic axis.
     */
    [[nodiscard]] SpacePtr with_essential(std::vector<EssentialCondition> conditions) const;

    /// @brief Essential conditions of this space.
    [[nodiscard]] std::span<const EssentialCondition> essential_conditions() const noexcept;

    /**
     * @brief Global indices of the locally owned values fixed by @p region.
     * @return A new `IS` owned by the caller.
     */
    [[nodiscard]] virtual IS boundary_indices(const BoundaryRegion& region) const = 0;

protected:
    /// @brief Called by backend implementations.
    FormSpace(std::shared_ptr<const DiscreteManifold> manifold, FormType type);

    /// @brief A copy of the backend space carrying @p conditions.
    [[nodiscard]] virtual SpacePtr clone_with(std::vector<EssentialCondition> conditions) const = 0;

    /// @brief Check that @p region can constrain this space; throws TypeError otherwise.
    virtual void validate_region(const BoundaryRegion& region) const = 0;

private:
    std::shared_ptr<const DiscreteManifold> manifold_;
    FormType type_;

protected:
    /// @brief Essential conditions, set by clone_with() implementations.
    std::vector<EssentialCondition> essential_;
};

} // namespace declib
