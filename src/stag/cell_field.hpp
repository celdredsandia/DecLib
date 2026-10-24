// Internal: CellField on a DMStag.
#pragma once

#include <petscdmstag.h>

#include "declib/geometry/cell_field.hpp"
#include "detail/petsc.hpp"

namespace declib::stag::detail {

class StagCellField final : public CellField {
public:
    // Takes ownership of `dm`; creates a zeroed local vector.
    StagCellField(DM dm, CellSet cells, int components) : dm_(dm), cells_(cells), components_(components)
    {
        DECLIB_PETSC(DMCreateLocalVector(dm_, &local_));
        DECLIB_PETSC(VecSet(local_, 0));
    }
    ~StagCellField() override
    {
        VecDestroy(&local_);
        DMDestroy(&dm_);
    }
    StagCellField(const StagCellField&) = delete;
    StagCellField& operator=(const StagCellField&) = delete;

    CellSet cells() const noexcept override { return cells_; }
    int components() const noexcept override { return components_; }
    Vec local_vec() const noexcept override { return local_; }
    DM dm() const noexcept { return dm_; }

private:
    DM dm_;
    Vec local_ = nullptr;
    CellSet cells_;
    int components_;
};

} // namespace declib::stag::detail
