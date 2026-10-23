/**
 * @file derived.hpp
 * @brief Operators composed from \f$d\f$, \f$\star\f$, \f$\wedge\f$ and \f$i_X\f$.
 */
#pragma once

#include <memory>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/operators/hodge_star.hpp"
#include "declib/operators/interior_product.hpp"
#include "declib/operators/linear_operator.hpp"

namespace declib {

/**
 * @brief Codifferential \f$\delta = (-1)^{n(k+1)+1} \star^{-1} d \star\f$ on @p space.
 *
 * Maps \f$k\f$-forms to \f$(k-1)\f$-forms of the same orientation.
 */
[[nodiscard]] OpPtr codifferential(SpacePtr space, HodgeScheme scheme = HodgeScheme::Diagonal);

/// @brief Hodge Laplacian \f$\Delta = d\delta + \delta d\f$ on @p space.
[[nodiscard]] OpPtr hodge_laplacian(SpacePtr space, HodgeScheme scheme = HodgeScheme::Diagonal);

/**
 * @brief Mass matrix \f$M : V \to V^*\f$ of the \f$L^2\f$ inner product.
 *
 * \f$(\alpha, \beta) = \langle \alpha, M \beta \rangle\f$; \f$M\f$ is
 * \f$\star\f$ up to sign.
 */
[[nodiscard]] OpPtr mass(SpacePtr space, HodgeScheme scheme = HodgeScheme::Diagonal);

/**
 * @brief Lie derivative \f$L_X = d\, i_X + i_X\, d\f$ on @p space, by Cartan's formula.
 *
 * Holds @p X by reference.
 */
[[nodiscard]] OpPtr lie_derivative(const Form& X, SpacePtr space,
                                   std::shared_ptr<const InteriorScheme> scheme);

} // namespace declib
