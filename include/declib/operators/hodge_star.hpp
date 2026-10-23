/**
 * @file hodge_star.hpp
 * @brief The discrete Hodge star \f$\star\f$.
 */
#pragma once

#include "declib/forms/cochain.hpp"

namespace declib {

/// @brief Choice of discrete Hodge star.
enum class HodgeKind {
    Diagonal, ///< Ratio of dual to primal cell volumes (circumcentric dual).
    Galerkin  ///< Mass matrix of Whitney forms; symmetric positive definite.
};

/**
 * @brief Apply the Hodge star \f$\star_k\f$, mapping primal \f$k\f$-cochains
 *        to dual \f$(n-k)\f$-cochains.
 *
 * For the diagonal star,
 * \f$(\star_k)_{\sigma\sigma} = |\star\sigma| \, / \, |\sigma|\f$.
 *
 * @param omega Primal \f$k\f$-cochain.
 * @param kind  Which discretisation to use.
 * @return The dual cochain \f$\star\omega\f$.
 * @warning The diagonal star requires a well-centred (Delaunay) mesh to be
 *          positive definite.
 */
template <int K, int N>
Cochain<N - K, Complex::Dual> hodge_star(const Cochain<K, Complex::Primal>& omega,
                                         HodgeKind kind = HodgeKind::Diagonal);

} // namespace declib
