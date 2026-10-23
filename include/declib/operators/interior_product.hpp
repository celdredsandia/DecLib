/**
 * @file interior_product.hpp
 * @brief The discrete interior product and its pairing adjoint.
 */
#pragma once

#include <memory>
#include <string>

#include "declib/core/types.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/mapping/grid_mapping.hpp"
#include "declib/operators/bilinear_operator.hpp"

namespace declib {

/// @brief How a vector field \f$X\f$ is represented as a discrete form.
enum class VectorRepresentation {
    Flux,       ///< The twisted \f$(n-1)\f$-form \f$i_X \mathrm{vol}\f$.
    Circulation ///< The straight 1-form \f$X^\flat\f$.
};

/**
 * @brief The representation implied by a vector-field space.
 * @throws TypeError if @p type is neither a twisted \f$(n-1)\f$-form nor a straight 1-form.
 */
[[nodiscard]] VectorRepresentation representation_of(const FormType& type, int n);

/**
 * @brief A discretization of the interior product.
 *
 * Concrete schemes are deliberately not chosen yet. A scheme states which
 * vector representations and form types it accepts.
 */
class InteriorScheme {
public:
    virtual ~InteriorScheme() = default;

    /// @brief Scheme name, also its PETSc options value (`-declib_interior_type <name>`).
    [[nodiscard]] virtual std::string name() const = 0;

    /// @brief True if the scheme accepts these argument types on @p mapping.
    [[nodiscard]] virtual bool supports(VectorRepresentation x, const FormType& form,
                                        const GridMapping& mapping) const = 0;
};

class InteriorProductAdjoint;

/**
 * @brief The interior product \f$(X, \alpha) \mapsto i_X \alpha\f$, \f$\Lambda^k \to \Lambda^{k-1}\f$.
 *
 * Bilinear in the vector field \f$X\f$ (left argument) and the form
 * \f$\alpha\f$ (right argument). The representation of \f$X\f$ is read from
 * the type of its space, see representation_of().
 *
 * The scheme is a required argument with no default.
 */
class InteriorProduct final : public BilinearOperator {
public:
    /**
     * @brief Build \f$i_X\f$.
     * @param vector_space Space holding \f$X\f$.
     * @param form_space   Space of \f$k\f$-forms, \f$k \ge 1\f$.
     * @param scheme       Discretization.
     * @throws TypeError if the scheme does not support the types.
     */
    [[nodiscard]] static std::shared_ptr<const InteriorProduct>
    create(SpacePtr vector_space, SpacePtr form_space, std::shared_ptr<const InteriorScheme> scheme);

    /**
     * @brief The adjoint \f$(X, \beta) \mapsto i_X^\dagger \beta\f$.
     *
     * \f$\beta\f$ lives in `range()->dual_space()` and the result in
     * `right()->dual_space()`, with
     * \f$\langle i_X \alpha, \beta \rangle = \langle \alpha, i_X^\dagger \beta \rangle\f$
     * for all \f$X, \alpha, \beta\f$, to round-off.
     */
    [[nodiscard]] std::shared_ptr<const InteriorProductAdjoint> adjoint() const;

    /// @brief The discretization in use.
    [[nodiscard]] const InteriorScheme& scheme() const noexcept;

    [[nodiscard]] const SpacePtr& left() const noexcept override;
    [[nodiscard]] const SpacePtr& right() const noexcept override;
    [[nodiscard]] const SpacePtr& range() const noexcept override;
    void apply(const Form& X, const Form& alpha, Form& out) const override;

private:
    InteriorProduct(SpacePtr vector_space, SpacePtr form_space, std::shared_ptr<const InteriorScheme> scheme);

    SpacePtr vector_space_;
    SpacePtr form_space_;
    SpacePtr range_;
    std::shared_ptr<const InteriorScheme> scheme_;
};

/**
 * @brief The pairing adjoint \f$i_X^\dagger\f$ of an InteriorProduct.
 *
 * In the continuum the adjoint of \f$i_X\f$ under
 * \f$\int \alpha \wedge \beta\f$ is \f$(-1)^{k+1} i_X\f$ acting on the
 * complementary twisted forms. Discretely the two cannot be defined
 * independently and remain exactly adjoint, so this operator is defined as
 * the pairing adjoint of `InteriorProduct::fix_left(X)`. A dedicated kernel
 * computes it and is tested against that definition. Exact adjointness is
 * what makes discrete Lie derivatives and Hamiltonian brackets conserve
 * energy.
 *
 * Only constructed through InteriorProduct::adjoint().
 */
class InteriorProductAdjoint final : public BilinearOperator {
public:
    /// @brief The interior product this is the adjoint of.
    [[nodiscard]] const InteriorProduct& primal() const noexcept;

    [[nodiscard]] const SpacePtr& left() const noexcept override;
    [[nodiscard]] const SpacePtr& right() const noexcept override;
    [[nodiscard]] const SpacePtr& range() const noexcept override;
    void apply(const Form& X, const Form& beta, Form& out) const override;

private:
    friend class InteriorProduct;
    explicit InteriorProductAdjoint(std::shared_ptr<const InteriorProduct> primal);

    std::shared_ptr<const InteriorProduct> primal_;
};

} // namespace declib
