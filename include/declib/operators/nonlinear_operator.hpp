/**
 * @file nonlinear_operator.hpp
 * @brief Nonlinear operators on forms, and scalar functionals.
 */
#pragma once

#include <memory>
#include <string>

#include <petscsnes.h>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/operators/linear_operator.hpp"

namespace declib {

/**
 * @brief A differentiable nonlinear map \f$F : V \to W\f$.
 *
 * Examples: potential vorticity \f$q = \star(du + f)/\star h\f$, or a Hodge
 * star whose coefficients depend on the solution.
 */
class NonlinearOperator : public std::enable_shared_from_this<NonlinearOperator> {
public:
    virtual ~NonlinearOperator() = default;

    /// @brief Space of inputs.
    [[nodiscard]] virtual const SpacePtr& domain() const noexcept = 0;

    /// @brief Space of outputs.
    [[nodiscard]] virtual const SpacePtr& range() const noexcept = 0;

    /// @brief \f$y = F(x)\f$.
    virtual void apply(const Form& x, Form& y) const = 0;

    /**
     * @brief The linearization \f$DF(x)\f$.
     *
     * Holds @p x by reference, so it follows later updates of @p x.
     */
    [[nodiscard]] virtual OpPtr jacobian(const Form& x) const = 0;

    /**
     * @brief Install residual \f$F(x) - b\f$ and Jacobian callbacks on @p snes.
     *
     * Requires domain() and range() to have the same layout.
     */
    void attach(SNES snes, const Form& b) const;

    /// @brief Name used in error messages.
    [[nodiscard]] virtual std::string name() const;
};

/**
 * @brief A differentiable scalar functional \f$\mathcal{F} : V \to \mathbb{R}\f$.
 *
 * Energies and Hamiltonians. The gradient is the variational derivative,
 * an element of the dual space \f$V^*\f$ defined through the metric-free
 * pairing:
 * \f$d\mathcal{F}(x)[\delta x] = \langle \delta x, \delta\mathcal{F}/\delta x \rangle\f$.
 * Keeping gradients in \f$V^*\f$ makes brackets built from \f$d\f$,
 * \f$\star\f$, \f$\wedge\f$, \f$i_X\f$ and \f$i_X^\dagger\f$ antisymmetric
 * by construction.
 */
class Functional : public std::enable_shared_from_this<Functional> {
public:
    virtual ~Functional() = default;

    /// @brief Space of arguments.
    [[nodiscard]] virtual const SpacePtr& domain() const noexcept = 0;

    /// @brief \f$\mathcal{F}(x)\f$, reduced over all ranks.
    [[nodiscard]] virtual Scalar value(const Form& x) const = 0;

    /// @brief \f$g = \delta\mathcal{F}/\delta x\f$, with @p g in `domain()->dual_space()`.
    virtual void gradient(const Form& x, Form& g) const = 0;

    /// @brief Name used in error messages.
    [[nodiscard]] virtual std::string name() const;
};

/// @brief Shared pointer to an immutable NonlinearOperator.
using NonlinearPtr = std::shared_ptr<const NonlinearOperator>;

/// @brief Shared pointer to an immutable Functional.
using FunctionalPtr = std::shared_ptr<const Functional>;

} // namespace declib
