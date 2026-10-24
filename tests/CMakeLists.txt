# Use an installed GoogleTest when there is one, otherwise fetch it.
find_package(GTest 1.11 QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(declib_tests
  main.cpp
  test_forms.cpp
  test_topology.cpp
  test_exterior_derivative.cpp
  test_hodge_star.cpp
  test_operators.cpp
  test_convergence.cpp
)
target_link_libraries(declib_tests PRIVATE DecLib::declib GTest::gtest)

# Extra flags for mpiexec, e.g. --oversubscribe on small machines.
set(DECLIB_MPIEXEC_PREFLAGS "" CACHE STRING "Flags passed to mpiexec before the test executable")
separate_arguments(_declib_preflags UNIX_COMMAND "${DECLIB_MPIEXEC_PREFLAGS}")
foreach(np 1 2 4)
  add_test(NAME declib_tests_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_declib_preflags}
                   $<TARGET_FILE:declib_tests>)
  set_tests_properties(declib_tests_np${np} PROPERTIES PROCESSORS ${np}
                       ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
endforeach()
