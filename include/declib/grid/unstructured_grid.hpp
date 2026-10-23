/**
 * @file unstructured_grid.hpp
 * @brief Simplicial and polyhedral meshes backed by PETSc `DMPlex`.
 */
#pragma once

#include <string>

#include "declib/grid/grid.hpp"

namespace declib {

/**
 * @brief Unstructured cell complex backed by PETSc `DMPlex`.
 *
 * The `DMPlex` Hasse diagram provides the incidence relations from which
 * the boundary operators (and hence the exterior derivative) are built.
 */
class UnstructuredGrid final : public Grid {
public:
    /**
     * @brief Load a mesh from a file readable by `DMPlexCreateFromFile`
     *        (Gmsh, Exodus II, HDF5, ...).
     * @param path Path to the mesh file.
     * @param interpolate Create intermediate cells (edges, faces) when true.
     */
    explicit UnstructuredGrid(const std::string& path, bool interpolate = true);

    [[nodiscard]] int dimension() const noexcept override;
    [[nodiscard]] std::size_t num_cells(int k) const override;
};

} // namespace declib
