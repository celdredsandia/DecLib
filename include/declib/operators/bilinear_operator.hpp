/**
 * @file bilinear_operator.hpp
 * @brief Bilinear operators \f$B : U \times V \to W\f$ on forms.
 */
#pragma once

#include <memory>
#include <string>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/operators/linear_operator.hpp"

namespace declib {

/**
 * @brief A bilinear map \f$B : U \times V \to W\f$ between form spaces.
 *
 * Examples are the wedge product and the interior product. Freezing one
 * argument gives a LinearOperator, which is how bilinear operators enter
 * solvers and how their Jacobians are formed:
 * \f$\partial B(a,b)[\delta a, \delta b] = B(\delta a, b) + B(a, \delta b)\f$,
 * that is `fix_right(b)` applied to \f$\delta a\f$ plus `fix_left(a)`
 * applied to \f$\delta b\f$.
 */
class BilinearOperator : public std::enable_shared_from_this<BilinearOperator> {
public:
    virtual ~BilinearOperator() = default;

    /// @brief Space \f$U\f$ of the first argument.
    [[nodiscard]] virtual const SpacePtr& left() const noexcept = 0;

    /// @brief Space \f$V\f$ of the second argument.
    [[nodiscard]] virtual const SpacePtr& right() const noexcept = 0;

    /// @brief Space \f$W\f$ of the result.
    [[nodiscard]] virtual const SpacePtr& range() const noexcept = 0;

    /// @brief \f$c = B(a, b)\f$.
    virtual void apply(const Form& a, const Form& b, Form& c) const = 0;

    /// @brief \f$c \mathrel{+}= s B(a, b)\f$. The default calls apply() into a temporary.
    virtual void apply_add(const Form& a, const Form& b, Form& c, Scalar s = 1) const;

    /// @brief Convenience: a new form \f$B(a, b)\f$ in range().
    [[nodiscard]] Form operator()(const Form& a, const Form& b) const;

    /**
     * @brief The linear operator \f$b \mapsto B(a, b)\f$.
     *
     * Holds @p a by reference: updating @p a updates the operator, which is
     * what a Newton iteration needs. @p a must outlive the returned operator.
     */
    [[nodiscard]] virtual OpPtr fix_left(const Form& a) const;

    /// @brief The linear operator \f$a \mapsto B(a, b)\f$. Holds @p b by reference.
    [[nodiscard]] virtual OpPtr fix_right(const Form& b) const;

    /// @brief Name used in error messages.
    [[nodiscard]] virtual std::string name() const;
};

/// @brief Shared pointer to an immutable BilinearOperator.
using BilinearPtr = std::shared_ptr<const BilinearOperator>;

} // namespace declib
