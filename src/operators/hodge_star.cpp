#include "declib/operators/hodge_star.hpp"

#include "declib/core/error.hpp"
#include "declib/manifold/discrete_manifold.hpp"
#include "stag/stencil.hpp"

namespace declib {

HodgeStar::HodgeStar(SpacePtr domain, HodgeScheme scheme) : domain_(std::move(domain)), scheme_(scheme)
{
    const FormType& t = domain_->type();
    const DiscreteManifold& M = domain_->manifold();
    const Geometry& G = M.geometry();
    const int n = M.dimension();
    if (scheme_ != HodgeScheme::Diagonal)
        throw TypeError("HodgeStar: only the Diagonal scheme is implemented so far");
    if (!G.is_orthogonal()) throw TypeError("HodgeStar: the Diagonal scheme needs an orthogonal geometry");
    if (M.topology().backend() != Backend::Stag) throw Error("HodgeStar: only structured grids are supported");
    range_ = M.space(n - t.degree, opposite(t.orientation), t.bundle);

    // Both forms sit on the same primal strata; the star is the ratio of the
    // range cell measure to the domain cell measure.
    const CellSet from = domain_->cells();
    const CellSet to = range_->cells();
    auto op = detail::hodge_stencil(n, from.complex, t.degree, t.bundle.rank());
    op.scale_num = G.measure(to).local_vec();
    op.scale_den = G.measure(from).local_vec();
    stencil_ = std::make_shared<const detail::StencilOp>(std::move(op));
}

std::shared_ptr<const HodgeStar> HodgeStar::create(SpacePtr domain, HodgeScheme scheme)
{
    if (!domain) throw Error("HodgeStar::create: null space");
    return std::shared_ptr<const HodgeStar>(new HodgeStar(std::move(domain), scheme));
}

HodgeScheme HodgeStar::scheme() const noexcept { return scheme_; }

const SpacePtr& HodgeStar::domain() const noexcept { return domain_; }

const SpacePtr& HodgeStar::range() const noexcept { return range_; }

void HodgeStar::apply(const Form& x, Form& y) const
{
    check_args(x, y);
    stencil_->apply(x, y);
}

Mat HodgeStar::assemble(MatType type) const { return stencil_->assemble(domain_->dm(), range_->dm(), type); }

bool HodgeStar::is_diagonal() const noexcept { return scheme_ == HodgeScheme::Diagonal; }

std::string HodgeStar::name() const { return "star" + std::to_string(domain_->type().degree); }

} // namespace declib
