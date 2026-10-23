/**
 * @file wedge.hpp
 * @brief The discrete wedge product.
 */
#pragma once

#include <memory>
#include <string>

#include "declib/core/types.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/mapping/grid_mapping.hpp"
#include "declib/operators/bilinear_operator.hpp"

namespace declib {

/**
 * @brief A discretization of the wedge product.
 *
 * Concrete schemes are deliberately not chosen yet; candidates include the
 * cubical cup product, its averaged (symmetrised) version, and
 * interpolated products across the primal and dual complexes. A scheme
 * declares its properties so that Wedge can reject unsupported argument
 * types and the test suite can check exactly the identities it claims.
 */
class WedgeScheme {
public:
    virtual ~WedgeScheme() = default;

    /// @brief Scheme name, also its PETSc options value (`-declib_wedge_type <name>`).
    [[nodiscard]] virtual std::string name() const = 0;

    /// @brief True if the scheme reads no geometry (possible only when all forms share a complex).
    [[nodiscard]] virtual bool is_metric_free() const noexcept = 0;

    /// @brief True if \f$d(a \wedge b) = da \wedge b + (-1)^k a \wedge db\f$ holds to round-off.
    [[nodiscard]] virtual bool exact_leibniz() const noexcept = 0;

    /// @brief True if \f$a \wedge b = (-1)^{kl}\, b \wedge a\f$ holds to round-off.
    [[nodiscard]] virtual bool exact_graded_commutativity() const noexcept = 0;

    /// @brief True if the scheme accepts arguments of these types on @p mapping.
    [[nodiscard]] virtual bool supports(const FormType& left, const FormType& right,
                                        const GridMapping& mapping) const = 0;
};

/**
 * @brief Pointwise bilinear map of fibers \f$E_1 \times E_2 \to E_3\f$.
 *
 * Used by Wedge for bundle-valued forms: scalar multiplication, dot
 * product, cross product or matrix product.
 */
class FiberProduct {
public:
    /// @brief \f$\mathbb{R} \times \mathbb{R} \to \mathbb{R}\f$, ordinary multiplication.
    [[nodiscard]] static FiberProduct scalar() noexcept;
    /// @brief \f$\mathbb{R}^r \times \mathbb{R}^r \to \mathbb{R}\f$.
    [[nodiscard]] static FiberProduct dot() noexcept;
    /// @brief \f$\mathbb{R}^3 \times \mathbb{R}^3 \to \mathbb{R}^3\f$.
    [[nodiscard]] static FiberProduct cross() noexcept;
    /// @brief \f$\mathbb{R} \times \mathbb{R}^r \to \mathbb{R}^r\f$.
    [[nodiscard]] static FiberProduct scale() noexcept;

    /// @brief Bundle of the result for fibers @p left and @p right. @throws TypeError if incompatible.
    [[nodiscard]] Bundle result(const Bundle& left, const Bundle& right) const;

private:
    enum class Kind { Scalar, Dot, Cross, Scale };
    explicit FiberProduct(Kind kind) noexcept;
    Kind kind_;
};

/**
 * @brief The wedge product \f$\wedge : \Lambda^k \times \Lambda^l \to \Lambda^{k+l}\f$.
 *
 * The result orientation follows product_orientation(): straight
 * \f$\wedge\f$ straight and twisted \f$\wedge\f$ twisted are straight, mixed
 * products are twisted. The result degree is \f$k + l \le n\f$.
 *
 * The scheme is a required argument with no default.
 */
class Wedge final : public BilinearOperator {
public:
    /**
     * @brief Build \f$\wedge\f$ on @p left \f$\times\f$ @p right.
     * @throws TypeError if \f$k + l > n\f$, the scheme does not support the
     *         types, or the fibers are incompatible with @p fiber.
     */
    [[nodiscard]] static std::shared_ptr<const Wedge>
    create(SpacePtr left, SpacePtr right, std::shared_ptr<const WedgeScheme> scheme,
           FiberProduct fiber = FiberProduct::scalar());

    /// @brief The discretization in use.
    [[nodiscard]] const WedgeScheme& scheme() const noexcept;

    [[nodiscard]] const SpacePtr& left() const noexcept override;
    [[nodiscard]] const SpacePtr& right() const noexcept override;
    [[nodiscard]] const SpacePtr& range() const noexcept override;
    void apply(const Form& a, const Form& b, Form& c) const override;

private:
    Wedge(SpacePtr left, SpacePtr right, std::shared_ptr<const WedgeScheme> scheme, FiberProduct fiber);

    SpacePtr left_;
    SpacePtr right_;
    SpacePtr range_;
    std::shared_ptr<const WedgeScheme> scheme_;
    FiberProduct fiber_;
};

} // namespace declib
