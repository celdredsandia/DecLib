#include "declib/operators/derived.hpp"

#include "declib/core/error.hpp"
#include "declib/manifold/discrete_manifold.hpp"
#include "declib/operators/exterior_derivative.hpp"

namespace declib {

OpPtr codifferential(SpacePtr space, HodgeScheme scheme)
{
    if (!space) throw Error("codifferential: null space");
    const int n = space->manifold().dimension();
    const int k = space->type().degree;
    if (k < 1) throw TypeError("codifferential: needs a k-form with k >= 1");
    // delta = (-1)^(n(k+1)+1) star d star
    const OpPtr s1 = HodgeStar::create(space, scheme);
    const OpPtr d = ExteriorDerivative::create(s1->range());
    const OpPtr s2 = HodgeStar::create(d->range(), scheme);
    const Scalar sign = ((n * (k + 1) + 1) % 2 == 0) ? Scalar(1) : Scalar(-1);
    return sign * (s2 * (d * s1));
}

OpPtr hodge_laplacian(SpacePtr space, HodgeScheme scheme)
{
    if (!space) throw Error("hodge_laplacian: null space");
    const int n = space->manifold().dimension();
    const int k = space->type().degree;
    OpPtr dd, dd2;
    if (k < n) {
        const OpPtr d = ExteriorDerivative::create(space);
        dd = codifferential(d->range(), scheme) * d;
    }
    if (k > 0) {
        const OpPtr delta = codifferential(space, scheme);
        dd2 = OpPtr(ExteriorDerivative::create(delta->range())) * delta;
    }
    if (dd && dd2) return dd2 + dd;
    return dd ? dd : dd2;
}

OpPtr mass(SpacePtr space, HodgeScheme scheme) { return HodgeStar::create(std::move(space), scheme); }

OpPtr lie_derivative(const Form&, SpacePtr, std::shared_ptr<const InteriorScheme>)
{
    throw Error("lie_derivative is not implemented yet (planned for phase 2)");
}

} // namespace declib
