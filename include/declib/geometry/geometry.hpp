/**
 * @file geometry.hpp
 * @brief Abstract interface of the metric data on a Topology.
 */
#pragma once

#include <functional>
#include <span>

#include "declib/core/types.hpp"
#include "declib/geometry/cell_field.hpp"
#include "declib/topology/cell_set.hpp"
#include "declib/topology/topology.hpp"

namespace declib {

class Form;

/// @brief Where dual vertices are placed inside primal top-dimensional cells.
enum class DualPlacement {
    Circumcentric, ///< At circumcentres; makes the diagonal Hodge star consistent on orthogonal grids.
    Barycentric    ///< At barycentres; always inside the cell.
};

/**
 * @brief A continuous form evaluated at a point.
 *
 * Called with the physical coordinates @p x (length \f$n\f$) and writes the
 * form's components into @p values, in the basis
 * \f$dx^{i_1} \wedge \dots \wedge dx^{i_k}\f$ with
 * \f$i_1 < \dots < i_k\f$ in lexicographic order, times the fiber rank for
 * bundle-valued forms. Evaluated on the host.
 */
using FormFunction = std::function<void(std::span<const Real> x, std::span<Scalar> values)>;

/// @brief Quadrature used by the de Rham map.
struct Quadrature {
    int order = 4; ///< Polynomial degree integrated exactly on each cell.
};

/**
 * @brief Everything metric about a discrete manifold.
 *
 * Geometry supplies vertex coordinates, the placement of dual vertices,
 * primal and dual cell measures, and anything else a metric operator needs.
 * It holds a reference to its Topology and never modifies it. Metric-free
 * operators (exterior derivative, duality pairing) never read it.
 *
 * Geometries are immutable after creation and shared through
 * `std::shared_ptr<const Geometry>`.
 */
class Geometry : public std::enable_shared_from_this<Geometry> {
public:
    virtual ~Geometry() = default;

    /// @brief The topology this geometry is attached to.
    [[nodiscard]] virtual const Topology& topology() const noexcept = 0;

    /// @brief Placement of dual vertices.
    [[nodiscard]] virtual DualPlacement dual_placement() const noexcept = 0;

    /**
     * @brief Measures \f$|\sigma|\f$ of every cell of a family.
     *
     * Lengths for \f$k=1\f$, areas for \f$k=2\f$, volumes for \f$k=3\f$ and
     * 1 for vertices, on either complex. Dual cells on an extended boundary
     * have the measure of their part on \f$\partial M\f$.
     */
    [[nodiscard]] virtual const CellField& measure(CellSet cells) const = 0;

    /// @brief Coordinates of primal vertices or dual vertices (\f$n\f$ components).
    [[nodiscard]] virtual const CellField& coordinates(Complex complex) const = 0;

    /**
     * @brief True when the metric is diagonal in the grid frame.
     *
     * Uniform and rectilinear grids are orthogonal; mapped grids are not.
     * Operators use this to select cheaper kernels and to reject schemes
     * (such as the diagonal Hodge star) that need orthogonality.
     */
    [[nodiscard]] virtual bool is_orthogonal() const noexcept = 0;

    /**
     * @brief De Rham map: integrate a continuous form over every cell.
     *
     * Writes \f$R(\omega)_\sigma = \int_\sigma \omega\f$ into @p out for every
     * cell \f$\sigma\f$ that @p out lives on, using the orientation of
     * \f$\sigma\f$ (inner for straight forms, outer for twisted forms).
     *
     * @param omega Continuous form, see FormFunction.
     * @param out   Form whose space selects degree, orientation and cells.
     * @param quad  Quadrature rule.
     */
    virtual void de_rham(const FormFunction& omega, Form& out, Quadrature quad = {}) const = 0;
};

} // namespace declib
