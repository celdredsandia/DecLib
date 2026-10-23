/**
 * @file error.hpp
 * @brief Exceptions thrown by DecLib++.
 */
#pragma once

#include <stdexcept>

namespace declib {

/// @brief Base class of every exception thrown by DecLib++.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Thrown when operator construction or composition receives form
 *        spaces of the wrong type.
 *
 * Examples: applying a Hodge star scheme to a geometry it does not support,
 * composing \f$A \circ B\f$ when the range of \f$B\f$ is not the domain of
 * \f$A\f$, or building an exterior derivative on \f$n\f$-forms.
 */
class TypeError : public Error {
public:
    using Error::Error;
};

/// @brief Thrown by a PETSc call that returned a nonzero error code.
class PetscError : public Error {
public:
    using Error::Error;
};

} // namespace declib
