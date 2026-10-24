#include "declib/forms/form_space.hpp"

#include "declib/core/error.hpp"
#include "declib/forms/form.hpp"
#include "declib/manifold/discrete_manifold.hpp"

namespace declib {

FormSpace::FormSpace(std::shared_ptr<const DiscreteManifold> manifold, FormType type)
    : manifold_(std::move(manifold)), type_(std::move(type))
{
    if (!manifold_) throw Error("FormSpace: null manifold");
}

const DiscreteManifold& FormSpace::manifold() const noexcept { return *manifold_; }

const std::shared_ptr<const DiscreteManifold>& FormSpace::manifold_ptr() const noexcept { return manifold_; }

const FormType& FormSpace::type() const noexcept { return type_; }

CellSet FormSpace::cells() const noexcept { return manifold_->mapping().cells(type_.degree, type_.orientation); }

bool FormSpace::same_type_as(const FormSpace& other) const noexcept
{
    return manifold_.get() == other.manifold_.get() && type_ == other.type_;
}

Form FormSpace::zeros() const { return Form(shared_from_this()); }

Form FormSpace::from_function(const FormFunction& omega, Quadrature quad) const
{
    Form f(shared_from_this());
    manifold_->geometry().de_rham(omega, f, quad);
    return f;
}

SpacePtr FormSpace::dual_space() const { return manifold_->space(dual_type(type_, manifold_->dimension())); }

SpacePtr FormSpace::with_essential(std::vector<EssentialCondition> conditions) const
{
    for (const auto& c : conditions) validate_region(c.region);
    auto all = essential_;
    all.insert(all.end(), conditions.begin(), conditions.end());
    return clone_with(std::move(all));
}

std::span<const EssentialCondition> FormSpace::essential_conditions() const noexcept { return essential_; }

} // namespace declib
