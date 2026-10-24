// Test driver: PETSc (and with it Kokkos and MPI) wraps the whole run, and
// only rank 0 prints GoogleTest output.
#include <gtest/gtest.h>
#include <petscsys.h>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    if (PetscInitialize(&argc, &argv, nullptr, nullptr)) return 1;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }
    int result = RUN_ALL_TESTS();
    // A failure on any rank fails the run.
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
    PetscFinalize();
    return result;
}
