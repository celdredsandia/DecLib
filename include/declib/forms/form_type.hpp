/**
 * @file form_type.hpp
 * @brief The type of a discrete differential form.
 */
#pragma once

#include <string>

#include "declib/core/types.hpp"
#include "declib/forms/bundle.hpp"

namespace declib {

/**
 * @brief Degree, orientation and fiber of a form.
 *
 * Form types are runtime values. Operators check them once, when they are
 * constructed, and throw TypeError on a mismatch; nothing is checked per
 * application. Where a form is stored is not part of its type: it follows
 * from the type through the GridMapping of the DiscreteManifold.
 */
struct FormType {
    int degree = 0;                               ///< Degree \f$k\f$, \f$0 \le k \le n\f$.
    Orientation orientation = Orientation::Straight; ///< Straight or twisted.
    Bundle bundle = Bundle::scalar();             ///< Fiber; scalar unless stated.

    /// @brief Equality of all three fields.
    friend bool operator==(const FormType&, const FormType&) = default;
};

/**
 * @brief The type paired with @p t by the metric-free duality pairing.
 * @return Degree \f$n-k\f$, opposite orientation, dual bundle.
 */
[[nodiscard]] FormType dual_type(const FormType& t, int n);

/// @brief Human-readable name, for example `"twisted 2-form"`.
[[nodiscard]] std::string to_string(const FormType& t);

} // namespace declib
