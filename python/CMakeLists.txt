# The extension declib._core. It is written next to the Python sources so
# that the package can be used in place with PYTHONPATH=<source>/python.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
                OUTPUT_VARIABLE _petsc4py_include OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED HINTS ${_pybind11_dir})

pybind11_add_module(_core MODULE src/core.cpp)
target_include_directories(_core PRIVATE ${_petsc4py_include})
target_link_libraries(_core PRIVATE DecLib::declib)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/declib)
install(TARGETS _core LIBRARY DESTINATION declib)

if(DECLIB_BUILD_TESTS)
  add_test(NAME python_tests
           COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME example_quickstart_py
           COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/examples/quickstart.py)
  set_tests_properties(python_tests example_quickstart_py PROPERTIES
                       ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH};OMP_NUM_THREADS=1")
endif()
