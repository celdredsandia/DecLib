/**
 * @file pairing.hpp
 * @brief The metric-free duality pairing between a form space and its dual.
 */
#pragma once

#include <memory>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"
#include "declib/forms/form_space.hpp"

namespace declib {

/**
 * @brief The duality pairing \f$\langle \cdot, \cdot \rangle : V \times V^* \to \mathbb{R}\f$.
 *
 * Pairs a \f$k\f$-form \f$\alpha\f$ with a form \f$\beta\f$ of the
 * complementary type (degree \f$n-k\f$, opposite orientation, dual
 * bundle): the discrete counterpart of
 * \f$\int_M \alpha \wedge \beta\f$. Both live on the same primal cells,
 * so the pairing is a signed sum of products with no metric factors:
 * \f[
 *   \langle \alpha, \beta \rangle = \sum_\sigma s_\sigma\, \alpha_\sigma \beta_{\star\sigma},
 * \f]
 * with signs \f$s_\sigma = \pm 1\f$ fixed by the orientation convention.
 * Fiber values are contracted with the canonical pairing of
 * \f$E \times E^*\f$.
 */
class Pairing {
public:
    /// @brief The pairing between @p space and `space->dual_space()`.
    [[nodiscard]] static std::shared_ptr<const Pairing> create(SpacePtr space);

    /// @brief Space of the first argument.
    [[nodiscard]] const SpacePtr& left() const noexcept;

    /// @brief Space of the second argument, `left()->dual_space()`.
    [[nodiscard]] const SpacePtr& right() const noexcept;

    /// @brief \f$\langle \alpha, \beta \rangle\f$, reduced over all ranks.
    [[nodiscard]] Scalar operator()(const Form& alpha, const Form& beta) const;

private:
    explicit Pairing(SpacePtr space);

    SpacePtr left_;
    SpacePtr right_;
};

} // namespace declib
