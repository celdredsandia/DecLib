#include "declib/manifold/discrete_manifold.hpp"

#include <algorithm>

#include "declib/core/error.hpp"
#include "declib/stag/form_space.hpp"

namespace declib {

DiscreteManifold::DiscreteManifold(std::shared_ptr<const Topology> topology, std::shared_ptr<const Geometry> geometry,
                                   GridMapping mapping)
    : topology_(std::move(topology)), geometry_(std::move(geometry)), mapping_(mapping)
{}

std::shared_ptr<const DiscreteManifold> DiscreteManifold::create(std::shared_ptr<const Topology> topology,
                                                                 std::shared_ptr<const Geometry> geometry,
                                                                 GridMapping mapping)
{
    if (!topology || !geometry) throw Error("DiscreteManifold::create: null topology or geometry");
    if (&geometry->topology() != topology.get())
        throw Error("DiscreteManifold::create: the geometry belongs to a different topology");
    return std::shared_ptr<const DiscreteManifold>(
        new DiscreteManifold(std::move(topology), std::move(geometry), mapping));
}

const Topology& DiscreteManifold::topology() const noexcept { return *topology_; }

const Geometry& DiscreteManifold::geometry() const noexcept { return *geometry_; }

const GridMapping& DiscreteManifold::mapping() const noexcept { return mapping_; }

int DiscreteManifold::dimension() const noexcept { return topology_->dimension(); }

SpacePtr DiscreteManifold::space(const FormType& type) const
{
    const int n = dimension();
    if (type.degree < 0 || type.degree > n) throw TypeError("DiscreteManifold::space: degree out of range");
    std::lock_guard lock(mutex_);
    std::erase_if(spaces_, [](const auto& entry) { return entry.second.expired(); });
    for (const auto& [t, weak] : spaces_)
        if (t == type)
            if (auto s = weak.lock()) return s;

    SpacePtr space;
    switch (topology_->backend()) {
    case Backend::Stag:
        switch (n) {
        case 1: space = std::make_shared<const stag::FormSpace<1>>(shared_from_this(), type); break;
        case 2: space = std::make_shared<const stag::FormSpace<2>>(shared_from_this(), type); break;
        case 3: space = std::make_shared<const stag::FormSpace<3>>(shared_from_this(), type); break;
        default: throw Error("DiscreteManifold::space: structured grids support dimensions 1 to 3");
        }
        break;
    case Backend::Plex: throw Error("DiscreteManifold::space: unstructured grids are not implemented yet");
    }
    spaces_.emplace_back(type, space);
    return space;
}

SpacePtr DiscreteManifold::space(int k, Orientation o, Bundle bundle) const
{
    return space(FormType{k, o, bundle});
}

void DiscreteManifold::set_connection(std::shared_ptr<const Connection> connection)
{
    if (!connection) throw Error("DiscreteManifold::set_connection: null connection");
    std::erase_if(connections_, [&](const auto& c) { return c->bundle() == connection->bundle(); });
    connections_.push_back(std::move(connection));
}

const Connection* DiscreteManifold::connection(const Bundle& bundle) const noexcept
{
    for (const auto& c : connections_)
        if (c->bundle() == bundle) return c.get();
    return nullptr;
}

} // namespace declib
