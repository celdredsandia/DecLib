/**
 * @file linear_operator.hpp
 * @brief Linear operators between form spaces, and their algebra.
 */
#pragma once

#include <memory>
#include <string>

#include <petscksp.h>
#include <petscmat.h>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"
#include "declib/forms/form_space.hpp"

namespace declib {

class LinearOperator;

/// @brief Shared pointer to an immutable LinearOperator.
using OpPtr = std::shared_ptr<const LinearOperator>;

/**
 * @brief A linear map \f$A : V \to W\f$ between form spaces.
 *
 * Operators are built once (type checks, stencil and geometry factors
 * precomputed, optional assembly) and applied many times. Construction
 * throws TypeError on incompatible spaces; apply() only checks that its
 * arguments belong to domain() and range().
 *
 * Every linear operator can present itself to PETSc either matrix-free
 * (shell()) or assembled (assemble()), so `KSP`, `SNES` and `TS` work on
 * it unchanged.
 *
 * Invariants checked by the test suite for every concrete operator:
 * - shell(), assemble() and apply() agree;
 * - \f$\langle A x, y \rangle = \langle x, A^T y \rangle\f$ in the
 *   Euclidean inner product of the coefficient vectors;
 * - \f$\langle A x, \beta \rangle = \langle x, A^\dagger \beta \rangle\f$
 *   in the duality pairing, for pairing_adjoint().
 */
class LinearOperator : public std::enable_shared_from_this<LinearOperator> {
public:
    LinearOperator() = default;
    LinearOperator(const LinearOperator&) = delete;
    LinearOperator& operator=(const LinearOperator&) = delete;
    virtual ~LinearOperator();

    /// @brief The space \f$V\f$ of inputs.
    [[nodiscard]] virtual const SpacePtr& domain() const noexcept = 0;

    /// @brief The space \f$W\f$ of outputs.
    [[nodiscard]] virtual const SpacePtr& range() const noexcept = 0;

    /// @brief \f$y = A x\f$.
    virtual void apply(const Form& x, Form& y) const = 0;

    /// @brief \f$y \mathrel{+}= a A x\f$. The default calls apply() into a temporary.
    virtual void apply_add(const Form& x, Form& y, Scalar a = 1) const;

    /// @brief Algebraic transpose \f$x = A^T y\f$ of the coefficient matrix.
    /// The default multiplies by the transpose of the cached assembled matrix.
    virtual void apply_transpose(const Form& y, Form& x) const;

    /// @brief Convenience: a new form \f$A x\f$ in range().
    [[nodiscard]] Form operator()(const Form& x) const;

    /**
     * @brief Adjoint under the duality pairing.
     *
     * Returns \f$A^\dagger : W^* \to V^*\f$, where \f$V^*\f$ is
     * `domain()->dual_space()`, such that
     * \f$\langle A x, \beta \rangle = \langle x, A^\dagger \beta \rangle\f$
     * exactly. The default is built from apply_transpose() and the sign
     * pattern of the pairing; operators with a cheaper kernel override it.
     * For the exterior derivative it is, up to sign, the dual-complex
     * exterior derivative.
     */
    [[nodiscard]] virtual OpPtr pairing_adjoint() const;

    /// @brief A `MATSHELL` that calls apply() and apply_transpose(). Owned by the caller.
    [[nodiscard]] Mat shell() const;

    /// @brief An assembled sparse matrix of type @p type. Owned by the caller.
    [[nodiscard]] virtual Mat assemble(MatType type = MATAIJKOKKOS) const;

    /// @brief True when the coefficient matrix is diagonal; inverse() is then exact.
    [[nodiscard]] virtual bool is_diagonal() const noexcept { return false; }

    /// @brief Name used in error messages and `-mat_view` output.
    [[nodiscard]] virtual std::string name() const;

protected:
    /// @brief assemble() computed once and kept for the lifetime of the operator.
    [[nodiscard]] Mat cached_matrix() const;

    /// @brief Throw TypeError unless @p x and @p y belong to domain() and range().
    void check_args(const Form& x, const Form& y) const;

private:
    mutable Mat cached_ = nullptr;
};

/// @brief Options for operators inverted with a Krylov solver.
struct KSPOptions {
    std::string prefix;             ///< PETSc options prefix for the internal `KSP`.
    Real rtol = 1e-10;              ///< Relative tolerance.
    OpPtr preconditioner = nullptr; ///< Operator assembled for the preconditioner; null uses the operator itself.
};

// Operator algebra. Composition is lazy; each function checks space
// compatibility and throws TypeError on a mismatch. assemble() on a composite
// uses MatMatMult, MatAXPY and diagonal inversion where possible.

/// @brief \f$A \circ B\f$. Requires `B->range() == A->domain()`.
[[nodiscard]] OpPtr operator*(OpPtr A, OpPtr B);

/// @brief \f$A + B\f$. Requires equal domains and equal ranges.
[[nodiscard]] OpPtr operator+(OpPtr A, OpPtr B);

/// @brief \f$A - B\f$. Requires equal domains and equal ranges.
[[nodiscard]] OpPtr operator-(OpPtr A, OpPtr B);

/// @brief \f$a A\f$.
[[nodiscard]] OpPtr operator*(Scalar a, OpPtr A);

/**
 * @brief \f$A^{-1}\f$.
 *
 * Exact and pointwise when `A->is_diagonal()`; otherwise each application
 * runs a `KSP` solve configured by @p options.
 */
[[nodiscard]] OpPtr inverse(OpPtr A, KSPOptions options = {});

/// @brief The identity on @p space.
[[nodiscard]] OpPtr identity(SpacePtr space);

/// @brief The algebraic transpose \f$A^T : W \to V\f$ as an operator.
[[nodiscard]] OpPtr transpose(OpPtr A);


} // namespace declib
