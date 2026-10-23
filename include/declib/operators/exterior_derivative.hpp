/**
 * @file exterior_derivative.hpp
 * @brief The discrete exterior derivative \f$d\f$.
 */
#pragma once

#include "declib/forms/cochain.hpp"

namespace declib {

/**
 * @brief Apply the discrete exterior derivative \f$d_k\f$.
 *
 * \f$d_k\f$ is the transpose of the boundary operator \f$\partial_{k+1}\f$,
 * so Stokes' theorem holds exactly:
 * \f[
 *   \langle d\omega, \sigma \rangle = \langle \omega, \partial\sigma \rangle .
 * \f]
 * It satisfies \f$d_{k+1} \circ d_k = 0\f$ to machine precision.
 *
 * @tparam K    Degree of the input cochain.
 * @tparam Kind Primal or dual complex.
 * @param omega Input \f$k\f$-cochain.
 * @return The \f$(k+1)\f$-cochain \f$d\omega\f$.
 */
template <int K, Complex Kind>
Cochain<K + 1, Kind> d(const Cochain<K, Kind>& omega);

} // namespace declib
