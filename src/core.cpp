// Core value types: Bundle, FormType, BoundaryRegion, Topology::dual_of.

#include <sstream>

#include "declib/core/error.hpp"
#include "declib/forms/boundary_condition.hpp"
#include "declib/forms/bundle.hpp"
#include "declib/forms/form_type.hpp"
#include "declib/topology/topology.hpp"

namespace declib {

Bundle::Bundle(BundleKind kind, int rank, Orientation fiber_orientation) noexcept
    : kind_(kind), rank_(rank), fiber_orientation_(fiber_orientation)
{}

Bundle Bundle::scalar() noexcept { return Bundle(BundleKind::Scalar, 1, Orientation::Straight); }

Bundle Bundle::trivial(int rank)
{
    if (rank < 1) throw Error("Bundle::trivial: rank must be at least 1");
    return Bundle(BundleKind::Trivial, rank, Orientation::Straight);
}

Bundle Bundle::tangent(int n)
{
    if (n < 1 || n > 3) throw Error("Bundle::tangent: dimension must be 1, 2 or 3");
    return Bundle(BundleKind::Tangent, n, Orientation::Straight);
}

Bundle Bundle::cotangent(int n, Orientation fiber_orientation)
{
    if (n < 1 || n > 3) throw Error("Bundle::cotangent: dimension must be 1, 2 or 3");
    return Bundle(BundleKind::Cotangent, n, fiber_orientation);
}

BundleKind Bundle::kind() const noexcept { return kind_; }
int Bundle::rank() const noexcept { return rank_; }
Orientation Bundle::fiber_orientation() const noexcept { return fiber_orientation_; }
bool Bundle::is_trivial() const noexcept { return kind_ == BundleKind::Scalar || kind_ == BundleKind::Trivial; }

Bundle Bundle::dual() const
{
    switch (kind_) {
    case BundleKind::Tangent: return Bundle(BundleKind::Cotangent, rank_, fiber_orientation_);
    case BundleKind::Cotangent: return Bundle(BundleKind::Tangent, rank_, fiber_orientation_);
    default: return *this; // R^r is self-dual through its global frame
    }
}

FormType dual_type(const FormType& t, int n)
{
    return FormType{n - t.degree, opposite(t.orientation), t.bundle.dual()};
}

std::string to_string(const FormType& t)
{
    std::ostringstream s;
    s << (t.orientation == Orientation::Straight ? "straight " : "twisted ") << t.degree << "-form";
    if (t.bundle.kind() != BundleKind::Scalar) s << " (fiber rank " << t.bundle.rank() << ")";
    return s.str();
}

BoundaryRegion BoundaryRegion::all() noexcept { return BoundaryRegion(); }

BoundaryRegion BoundaryRegion::face(int axis, Side side) noexcept
{
    BoundaryRegion r;
    r.axis_ = axis;
    r.side_ = side;
    return r;
}

BoundaryRegion BoundaryRegion::label(std::string label, int value)
{
    if (label.empty()) throw Error("BoundaryRegion::label: empty label name");
    BoundaryRegion r;
    r.label_ = std::move(label);
    r.label_value_ = value;
    return r;
}

CellSet Topology::dual_of(CellSet c) const noexcept
{
    return CellSet{opposite(c.complex), dimension() - c.k};
}

} // namespace declib
