cmake_minimum_required(VERSION 3.24)
project(DecLibPP VERSION 0.1.0 LANGUAGES C CXX)

option(DECLIB_BUILD_LIBRARY "Build the C++ library (needs PETSc configured with Kokkos)" ON)
option(DECLIB_BUILD_TESTS "Build the C++ test suite" ON)
option(DECLIB_BUILD_EXAMPLES "Build the examples" ON)
option(DECLIB_BUILD_PYTHON "Build the pybind11 extension declib._core" OFF)
option(DECLIB_BUILD_DOCS "Build the Sphinx/Doxygen documentation" OFF)

if(DECLIB_BUILD_LIBRARY)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # PETSc is found through pkg-config: point PKG_CONFIG_PATH at
  # $PETSC_DIR/lib/pkgconfig. Kokkos comes from the same installation
  # (Kokkos_DIR=$PETSC_DIR/lib/cmake/Kokkos) when PETSc downloaded it.
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)
  find_package(Kokkos REQUIRED)
  find_package(MPI REQUIRED COMPONENTS C CXX)

  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${PETSc_INCLUDE_DIRS})
  check_cxx_source_compiles([=[
    #include <petscsys.h>
    #if defined(PETSC_USE_COMPLEX) || !defined(PETSC_HAVE_KOKKOS)
    #error unsupported
    #endif
    int main() { return 0; }
  ]=] DECLIB_PETSC_OK)
  unset(CMAKE_REQUIRED_INCLUDES)
  if(NOT DECLIB_PETSC_OK)
    message(FATAL_ERROR "DecLib++ needs PETSc configured with real scalars and Kokkos (--with-kokkos)")
  endif()

  add_library(declib
    src/core.cpp
    src/detail/petsc.cpp
    src/forms/form.cpp
    src/forms/form_space.cpp
    src/manifold/discrete_manifold.cpp
    src/operators/derived.cpp
    src/operators/essential.cpp
    src/operators/exterior_derivative.cpp
    src/operators/hodge_star.cpp
    src/operators/linear_operator.cpp
    src/operators/pairing.cpp
    src/stag/form_space.cpp
    src/stag/layout.cpp
    src/stag/stencil.cpp
    src/stag/topology.cpp
    src/stag/uniform_geometry.cpp
  )
  add_library(DecLib::declib ALIAS declib)
  target_include_directories(declib
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(declib PUBLIC PkgConfig::PETSc Kokkos::kokkos MPI::MPI_CXX)
  target_compile_options(declib PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>)

  include(GNUInstallDirs)
  install(TARGETS declib EXPORT DecLibTargets)
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  if(DECLIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()

  if(DECLIB_BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif()

  if(DECLIB_BUILD_PYTHON)
    add_subdirectory(python)
  endif()
endif()

if(DECLIB_BUILD_DOCS)
  add_subdirectory(docs)
endif()
