/**
 * @file discrete_manifold.hpp
 * @brief Topology, grid mapping and geometry combined.
 */
#pragma once

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "declib/core/types.hpp"
#include "declib/forms/bundle.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/forms/form_type.hpp"
#include "declib/geometry/connection.hpp"
#include "declib/geometry/geometry.hpp"
#include "declib/mapping/grid_mapping.hpp"
#include "declib/topology/topology.hpp"

namespace declib {

/**
 * @brief A discrete manifold: Topology + GridMapping + Geometry.
 *
 * The three parts are kept separate so that one topology can carry several
 * geometries and either mapping. Metric-free operators read only the
 * topology and the mapping; metric operators also read the geometry.
 *
 * DiscreteManifold is the factory for form spaces: space() returns the same
 * FormSpace for the same FormType while any reference to it is alive.
 * Create the manifold with create() so that spaces can hold a shared
 * reference back to it.
 *
 * @code{.cpp}
 * auto topo = declib::stag::Topology<2>::create(PETSC_COMM_WORLD, {.cells = {128, 128}});
 * auto geom = declib::stag::UniformGeometry<2>::create(topo, {.lengths = {1.0, 1.0}});
 * auto M    = declib::DiscreteManifold::create(topo, geom);
 * auto V0   = M->space(0, declib::Orientation::Straight);
 * @endcode
 */
class DiscreteManifold : public std::enable_shared_from_this<DiscreteManifold> {
public:
    /**
     * @brief Combine a topology and a geometry under a mapping.
     * @param topology Cell complex.
     * @param geometry Metric data; must be built on @p topology.
     * @param mapping  Where straight and twisted forms live.
     * @throws TypeError if @p geometry belongs to a different topology.
     */
    [[nodiscard]] static std::shared_ptr<const DiscreteManifold>
    create(std::shared_ptr<const Topology> topology,
           std::shared_ptr<const Geometry> geometry,
           GridMapping mapping = GridMapping::straight_on_primal());

    /// @brief The cell complex.
    [[nodiscard]] const Topology& topology() const noexcept;

    /// @brief The metric data.
    [[nodiscard]] const Geometry& geometry() const noexcept;

    /// @brief Where straight and twisted forms live.
    [[nodiscard]] const GridMapping& mapping() const noexcept;

    /// @brief Shorthand for `topology().dimension()`.
    [[nodiscard]] int dimension() const noexcept;

    /// @brief The (cached) space of forms of type @p type.
    [[nodiscard]] SpacePtr space(const FormType& type) const;

    /// @brief The (cached) space of \f$k\f$-forms of orientation @p o with values in @p bundle.
    [[nodiscard]] SpacePtr space(int k, Orientation o, Bundle bundle = Bundle::scalar()) const;

    /// @brief Attach a connection for a non-trivial bundle (phase 3).
    /// Must be called before any space of that bundle is created.
    void set_connection(std::shared_ptr<const Connection> connection);

    /// @brief The connection for @p bundle, or null if none was attached.
    [[nodiscard]] const Connection* connection(const Bundle& bundle) const noexcept;

private:
    DiscreteManifold(std::shared_ptr<const Topology>, std::shared_ptr<const Geometry>, GridMapping);

    std::shared_ptr<const Topology> topology_;
    std::shared_ptr<const Geometry> geometry_;
    GridMapping mapping_;
    // Spaces hold a strong reference to the manifold, so the cache is weak.
    mutable std::mutex mutex_;
    mutable std::vector<std::pair<FormType, std::weak_ptr<const FormSpace>>> spaces_;
    std::vector<std::shared_ptr<const Connection>> connections_;
};

} // namespace declib
