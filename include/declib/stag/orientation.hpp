/**
 * @file orientation.hpp
 * @brief Orientation conventions of structured complexes.
 *
 * The single source of the sign conventions used by the structured
 * backend. The exterior derivatives, the Hodge star and the duality
 * pairing all take their signs from these functions.
 *
 * - A primal \f$k\f$-cell spanning the axes \f$a_1 < \dots < a_k\f$ carries
 *   the orientation of \f$dx^{a_1} \wedge \dots \wedge dx^{a_k}\f$: edges
 *   point along the positive axis, faces are oriented as xy, xz, yz, and
 *   elements are positively oriented.
 * - A dual \f$k\f$-cell crossing the primal cell with axes \f$A\f$ spans the
 *   complementary axes \f$B = A^c\f$ and is oriented the same way, by
 *   \f$dx^{b_1} \wedge \dots \wedge dx^{b_k}\f$ with \f$b_1 < \dots < b_k\f$.
 * - On \f$\mathbb{R}^n\f$ with its standard orientation, twisted forms are
 *   identified with straight forms, so the outer orientation of a dual cell
 *   is the one induced by this inner orientation and the ambient one.
 */
#pragma once

#include <bit>

#include "declib/core/types.hpp"

namespace declib::stag {

/**
 * @brief Sign of the permutation that sorts the axes of @p a followed by the
 *        complementary axes in dimension @p n.
 *
 * This is \f$s\f$ in \f$dx^{A} \wedge dx^{A^c} = s\, dx^1 \wedge \dots \wedge dx^n\f$.
 * It is the sign of the duality pairing on a stratum and of the Euclidean
 * Hodge star \f$\star dx^A = s\, dx^{A^c}\f$.
 */
KOKKOS_INLINE_FUNCTION constexpr int complement_sign(unsigned a, int n) noexcept
{
    int inversions = 0;
    for (int i = 0; i < n; ++i) {
        if ((a >> i) & 1u) {
            // Axes of the complement that come before axis i.
            const unsigned below = (1u << i) - 1u;
            inversions += std::popcount(~a & below);
        }
    }
    return (inversions % 2 == 0) ? 1 : -1;
}

/**
 * @brief Position of axis @p b among the sorted axes of @p a.
 *
 * The cubical boundary is
 * \f$\partial \sigma = \sum_m (-1)^m (\sigma^{+}_{b_m} - \sigma^{-}_{b_m})\f$,
 * where \f$\sigma^{\pm}_{b}\f$ are the faces of \f$\sigma\f$ at the high and
 * low end of axis \f$b\f$ and \f$m\f$ is the position returned here.
 */
KOKKOS_INLINE_FUNCTION constexpr int axis_position(unsigned a, int b) noexcept
{
    return std::popcount(a & ((1u << b) - 1u));
}

/// @brief \f$(-1)^m\f$ for the position \f$m\f$ of axis @p b in @p a.
KOKKOS_INLINE_FUNCTION constexpr int boundary_sign(unsigned a, int b) noexcept
{
    return (axis_position(a, b) % 2 == 0) ? 1 : -1;
}

/// @brief The axes of dimension @p n not in @p a.
KOKKOS_INLINE_FUNCTION constexpr unsigned complement(unsigned a, int n) noexcept
{
    return ~a & ((1u << n) - 1u);
}

} // namespace declib::stag
