/**
 * @file exterior_derivative.hpp
 * @brief The discrete exterior derivative and its covariant version.
 */
#pragma once

#include <memory>

#include "declib/core/types.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/operators/linear_operator.hpp"

namespace declib {

/**
 * @brief The exterior derivative \f$d_k : \Lambda^k \to \Lambda^{k+1}\f$.
 *
 * Metric-free: it reads only the topology and the mapping. On the primal
 * complex \f$d_k = \partial_{k+1}^T\f$. On the dual complex
 * \f$\tilde d_k = \pm \partial_{n-k}\f$, with the sign fixed by the
 * orientation convention and, for DualBoundary::Extended, boundary terms
 * from the extended dual cells.
 *
 * Bundle-valued forms are differentiated componentwise, which requires a
 * trivial bundle; use CovariantDerivative otherwise.
 *
 * Invariants checked by the test suite:
 * - \f$d_{k+1} \circ d_k = 0\f$ to round-off;
 * - Stokes: \f$\langle d\alpha, c \rangle = \langle \alpha, \partial c \rangle\f$
 *   on random chains;
 * - \f$d R = R d\f$ for the de Rham map \f$R\f$, up to quadrature error;
 * - pairing_adjoint() equals the dual-complex derivative up to sign.
 */
class ExteriorDerivative final : public LinearOperator {
public:
    /**
     * @brief Build \f$d\f$ on @p domain.
     * @param domain Space of \f$k\f$-forms, \f$k < n\f$, either orientation.
     * @return Operator with range `manifold.space(k+1, same orientation, same bundle)`.
     * @throws TypeError if \f$k = n\f$ or the bundle is not trivial.
     */
    [[nodiscard]] static std::shared_ptr<const ExteriorDerivative> create(SpacePtr domain);

    [[nodiscard]] const SpacePtr& domain() const noexcept override;
    [[nodiscard]] const SpacePtr& range() const noexcept override;
    void apply(const Form& x, Form& y) const override;
    void apply_transpose(const Form& y, Form& x) const override;
    [[nodiscard]] OpPtr pairing_adjoint() const override;

private:
    explicit ExteriorDerivative(SpacePtr domain);

    SpacePtr domain_;
    SpacePtr range_;
};

/**
 * @brief The covariant exterior derivative \f$d^\nabla\f$ on bundle-valued forms.
 *
 * Uses the Connection attached to the manifold for the form's bundle:
 * \f$(d^\nabla \alpha)_\sigma = \sum_{\tau \in \partial\sigma} \pm U(\tau \to \sigma)\, \alpha_\tau\f$.
 * Satisfies \f$d^\nabla \circ d^\nabla = 0\f$ when the connection is flat.
 *
 * @note Planned for phase 3.
 */
class CovariantDerivative final : public LinearOperator {
public:
    /// @brief Build \f$d^\nabla\f$ on @p domain. @throws TypeError if no connection is attached.
    [[nodiscard]] static std::shared_ptr<const CovariantDerivative> create(SpacePtr domain);

    [[nodiscard]] const SpacePtr& domain() const noexcept override;
    [[nodiscard]] const SpacePtr& range() const noexcept override;
    void apply(const Form& x, Form& y) const override;

private:
    explicit CovariantDerivative(SpacePtr domain);

    SpacePtr domain_;
    SpacePtr range_;
};

} // namespace declib
