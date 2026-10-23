/**
 * @file form_space.hpp
 * @brief The discrete space of forms of one FormType.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include <petscdm.h>

#include "declib/core/types.hpp"
#include "declib/forms/boundary_condition.hpp"
#include "declib/forms/form_type.hpp"
#include "declib/geometry/geometry.hpp"
#include "declib/topology/cell_set.hpp"

namespace declib {

class DiscreteManifold;
class Form;

/**
 * @brief The space of discrete forms of one FormType on a DiscreteManifold.
 *
 * A FormSpace plays the role of a PETSc `DM`: it fixes the layout, ghost
 * exchange and parallel decomposition, while Form plays the role of a
 * `Vec`. It is created by DiscreteManifold::space(), which caches one
 * instance per FormType, so two spaces of the same type compare equal by
 * address.
 *
 * Spaces are immutable once their essential conditions are set and the
 * first Form is created.
 */
class FormSpace : public std::enable_shared_from_this<FormSpace> {
public:
    virtual ~FormSpace() = default;

    /// @brief The manifold this space is defined on.
    [[nodiscard]] const DiscreteManifold& manifold() const noexcept;

    /// @brief Degree, orientation and fiber of the forms in this space.
    [[nodiscard]] const FormType& type() const noexcept;

    /// @brief Cells the values live on, from the manifold's GridMapping.
    [[nodiscard]] CellSet cells() const noexcept;

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
     * adjoints take their values.
     */
    [[nodiscard]] std::shared_ptr<const FormSpace> dual_space() const;

    /**
     * @brief Add an essential boundary condition.
     *
     * Constrained values are excluded from the unknowns seen by solvers.
     * Must be called before the first Form of this space is created.
     */
    void add_essential(EssentialCondition condition);

    /// @brief Essential conditions added so far.
    [[nodiscard]] std::span<const EssentialCondition> essential_conditions() const noexcept;

protected:
    /// @brief Called by backend implementations.
    FormSpace(std::shared_ptr<const DiscreteManifold> manifold, FormType type);

private:
    std::shared_ptr<const DiscreteManifold> manifold_;
    FormType type_;
};

/// @brief Shared pointer to an immutable FormSpace.
using SpacePtr = std::shared_ptr<const FormSpace>;

} // namespace declib
