/**
 * @file essential.hpp
 * @brief Imposing essential boundary conditions on assembled systems.
 */
#pragma once

#include <petscmat.h>

#include "declib/forms/form.hpp"
#include "declib/forms/form_space.hpp"

namespace declib {

/**
 * @brief Impose the essential conditions of @p space on \f$A x = b\f$.
 *
 * Writes the prescribed values into @p x, then eliminates the constrained
 * rows and columns of @p A (`MatZeroRowsColumns` with unit diagonal) and
 * adjusts @p b so that the solution of the modified system satisfies the
 * conditions and the original equations at all other dofs. Symmetry of
 * @p A is preserved.
 *
 * @param space Space carrying the conditions (see FormSpace::with_essential()).
 * @param A     Square assembled operator on @p space.
 * @param x     Initial guess; receives the boundary values.
 * @param b     Right-hand side; adjusted in place.
 */
void apply_essential(const FormSpace& space, Mat A, Form& x, Form& b);

} // namespace declib
