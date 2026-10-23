/**
 * @file hodge_star.hpp
 * @brief The discrete Hodge star.
 */
#pragma once

#include <memory>

#include "declib/core/types.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/operators/linear_operator.hpp"

namespace declib {

/// @brief Discretization of the Hodge star.
enum class HodgeScheme {
    /**
     * Ratio of measures, \f$(\star_k)_{\sigma\sigma} = |\star\sigma| / |\sigma|\f$
     * for straight-to-twisted. Requires an orthogonal geometry
     * (Geometry::is_orthogonal()).
     */
    Diagonal,
    /**
     * Mass matrix of tensor-product Whitney forms on each cell. Symmetric
     * positive definite on any mapped geometry, but not diagonal, so its
     * inverse is a solve. Planned for phase 3.
     */
    Galerkin,
    /// Wider-stencil, higher-order stars. Reserved.
    HighOrder
};

/**
 * @brief The Hodge star \f$\star_k : \Lambda^k \to \tilde\Lambda^{n-k}\f$.
 *
 * Maps a \f$k\f$-form to an \f$(n-k)\f$-form of the opposite orientation,
 * which under either GridMapping lives on the opposite complex but on the
 * same primal cells. It carries the metric: it is the only first-order
 * operator that reads the Geometry.
 *
 * In the continuum \f$\star\star = (-1)^{k(n-k)}\f$; the Diagonal scheme
 * builds \f$\star_{n-k}\f$ on the opposite complex so that this holds
 * exactly.
 */
class HodgeStar final : public LinearOperator {
public:
    /**
     * @brief Build \f$\star\f$ on @p domain.
     * @param domain Space of \f$k\f$-forms of either orientation.
     * @param scheme Discretization.
     * @return Operator with range `manifold.space(n-k, opposite orientation, same bundle)`.
     * @throws TypeError if @p scheme does not support the geometry.
     */
    [[nodiscard]] static std::shared_ptr<const HodgeStar> create(SpacePtr domain,
                                                                 HodgeScheme scheme = HodgeScheme::Diagonal);

    /// @brief The discretization in use.
    [[nodiscard]] HodgeScheme scheme() const noexcept;

    [[nodiscard]] const SpacePtr& domain() const noexcept override;
    [[nodiscard]] const SpacePtr& range() const noexcept override;
    void apply(const Form& x, Form& y) const override;
    void apply_transpose(const Form& y, Form& x) const override;
    [[nodiscard]] bool is_diagonal() const noexcept override;

private:
    HodgeStar(SpacePtr domain, HodgeScheme scheme);

    SpacePtr domain_;
    SpacePtr range_;
    HodgeScheme scheme_;
};

} // namespace declib
