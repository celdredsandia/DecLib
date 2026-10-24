#include "declib/operators/exterior_derivative.hpp"

#include "declib/core/error.hpp"
#include "declib/manifold/discrete_manifold.hpp"
#include "stag/stencil.hpp"

namespace declib {

ExteriorDerivative::ExteriorDerivative(SpacePtr domain) : domain_(std::move(domain))
{
    const FormType& t = domain_->type();
    const DiscreteManifold& M = domain_->manifold();
    const int n = M.dimension();
    if (t.degree >= n) throw TypeError("ExteriorDerivative: d of an n-form is zero; no (n+1)-forms exist");
    if (!t.bundle.is_trivial())
        throw TypeError("ExteriorDerivative: the bundle is not trivial; use CovariantDerivative");
    if (M.topology().backend() != Backend::Stag) throw Error("ExteriorDerivative: only structured grids are supported");
    range_ = M.space(t.degree + 1, t.orientation, t.bundle);
    stencil_ = std::make_shared<const detail::StencilOp>(
        detail::exterior_derivative_stencil(n, domain_->cells().complex, t.degree, t.bundle.rank()));
}

std::shared_ptr<const ExteriorDerivative> ExteriorDerivative::create(SpacePtr domain)
{
    if (!domain) throw Error("ExteriorDerivative::create: null space");
    return std::shared_ptr<const ExteriorDerivative>(new ExteriorDerivative(std::move(domain)));
}

const SpacePtr& ExteriorDerivative::domain() const noexcept { return domain_; }

const SpacePtr& ExteriorDerivative::range() const noexcept { return range_; }

void ExteriorDerivative::apply(const Form& x, Form& y) const
{
    check_args(x, y);
    stencil_->apply(x, y);
}

OpPtr ExteriorDerivative::pairing_adjoint() const
{
    // Summation by parts: <d a, b> = (-1)^(k+1) <a, d b> for a k-form a,
    // exactly, since the extended dual cells carry zero boundary data.
    const int k = domain_->type().degree;
    const Scalar sign = (k % 2 == 0) ? Scalar(-1) : Scalar(1);
    return sign * OpPtr(create(range_->dual_space()));
}

Mat ExteriorDerivative::assemble(MatType type) const
{
    return stencil_->assemble(domain_->dm(), range_->dm(), type);
}

std::string ExteriorDerivative::name() const { return "d" + std::to_string(domain_->type().degree); }

// Phase 3.
CovariantDerivative::CovariantDerivative(SpacePtr domain) : domain_(std::move(domain)) {}

std::shared_ptr<const CovariantDerivative> CovariantDerivative::create(SpacePtr)
{
    throw Error("CovariantDerivative is not implemented yet (planned for phase 3)");
}

const SpacePtr& CovariantDerivative::domain() const noexcept { return domain_; }

const SpacePtr& CovariantDerivative::range() const noexcept { return range_; }

void CovariantDerivative::apply(const Form&, Form&) const
{
    throw Error("CovariantDerivative is not implemented yet (planned for phase 3)");
}

} // namespace declib
