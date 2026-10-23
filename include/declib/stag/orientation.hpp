/**
 * @file orientation.hpp
 * @brief Orientation conventions of structured complexes.
 *
 * The single source of the sign conventions used by the structured
 * backend. The sign tables of the dual exterior derivative, the Hodge star
 * and the duality pairing are all generated from these functions, so they
 * cannot drift apart.
 *
 * - Primal edges point along the positive axis.
 * - Primal faces are oriented by their normal axis through the right-hand
 *   rule (yz, zx, xy).
 * - Elements are positively oriented.
 * - A dual \f$k\f$-cell carries the outer orientation induced by the primal
 *   \f$(n-k)\f$-cell it crosses.
 */
#pragma once

#include "declib/core/types.hpp"
#include "declib/stag/topology.hpp"

namespace declib::stag {

/**
 * @brief Sign of a primal \f$(k-1)\f$-cell in the boundary of a primal \f$k\f$-cell.
 *
 * @param cell     The \f$k\f$-cell stratum.
 * @param face     A \f$(k-1)\f$-cell stratum in its boundary.
 * @param high     True for the face on the high side of the removed axis.
 * @return \f$+1\f$ or \f$-1\f$, the entry of \f$\partial_k\f$.
 */
[[nodiscard]] int incidence_sign(const Stratum& cell, const Stratum& face, bool high) noexcept;

/**
 * @brief Sign \f$s_\sigma\f$ of the duality pairing on a stratum.
 *
 * \f$\langle \alpha, \beta \rangle = \sum_\sigma s_\sigma \alpha_\sigma \beta_{\star\sigma}\f$
 * where \f$\sigma\f$ ranges over the primal cells of @p stratum.
 */
[[nodiscard]] int pairing_sign(const Stratum& stratum, int n) noexcept;

} // namespace declib::stag
