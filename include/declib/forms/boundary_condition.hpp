/**
 * @file boundary_condition.hpp
 * @brief Boundary regions and essential boundary conditions.
 */
#pragma once

#include <optional>
#include <string>

#include "declib/core/types.hpp"
#include "declib/geometry/geometry.hpp"

namespace declib {

/// @brief Low or high end of a structured-grid axis.
enum class Side {
    Low, ///< Index 0 along the axis.
    High ///< Last index along the axis.
};

/**
 * @brief A part of \f$\partial M\f$.
 *
 * Structured grids name boundary faces by axis and side; unstructured
 * meshes will name them by a `DMLabel` value.
 */
class BoundaryRegion {
public:
    /// @brief The whole boundary.
    [[nodiscard]] static BoundaryRegion all() noexcept;

    /// @brief One face of a structured box: the @p side end of axis @p axis.
    [[nodiscard]] static BoundaryRegion face(int axis, Side side) noexcept;

    /// @brief Cells marked with @p value in the `DMLabel` named @p label (unstructured grids).
    [[nodiscard]] static BoundaryRegion label(std::string label, int value);

    /// @brief True for all().
    [[nodiscard]] bool is_all() const noexcept { return !axis_ && label_.empty(); }

    /// @brief The axis of a face() region.
    [[nodiscard]] std::optional<int> axis() const noexcept { return axis_; }

    /// @brief The side of a face() region.
    [[nodiscard]] Side side() const noexcept { return side_; }

    /// @brief The label name of a label() region, empty otherwise.
    [[nodiscard]] const std::string& label_name() const noexcept { return label_; }

private:
    BoundaryRegion() = default;

    std::optional<int> axis_;
    Side side_ = Side::Low;
    std::string label_;
    int label_value_ = 0;
};

/**
 * @brief An essential (strong) boundary condition.
 *
 * Fixes the values of a form on the cells of its host complex that lie in
 * @ref region. Attach conditions with FormSpace::with_essential() and
 * impose them on an assembled system with apply_essential(), which
 * eliminates the constrained rows and columns, so the remaining degrees of
 * freedom form the relative cochain space \f$C^k(M, \partial M)\f$ (or
 * relative to a part of \f$\partial M\f$).
 *
 * Natural (weak) conditions are not expressed with this class: they are
 * values on the extended dual boundary cells, see DualBoundary::Extended
 * and Form::set_boundary_values().
 */
struct EssentialCondition {
    BoundaryRegion region = BoundaryRegion::all(); ///< Where the condition holds.
    /// @brief Prescribed form; empty means homogeneous (zero) values.
    std::optional<FormFunction> value;
};

} // namespace declib
