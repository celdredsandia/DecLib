# Documentation target: `cmake --build build --target docs`.
# Enabled with -DDECLIB_BUILD_DOCS=ON from the top-level CMakeLists.txt.

find_package(Doxygen REQUIRED)
find_program(SPHINX_BUILD sphinx-build REQUIRED)

set(DECLIB_DOCS_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(DECLIB_DOCS_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html)

option(DECLIB_DOCS_WERROR "Treat Sphinx warnings as errors" ON)
set(_sphinx_flags -j auto)
if(DECLIB_DOCS_WERROR)
  list(APPEND _sphinx_flags -W --keep-going)
endif()

# conf.py runs Doxygen with docs/Doxyfile, so one command builds everything.
add_custom_target(docs
  COMMAND ${CMAKE_COMMAND} -E env "PATH=$ENV{PATH}"
          ${SPHINX_BUILD} -b html ${_sphinx_flags}
          ${DECLIB_DOCS_SOURCE} ${DECLIB_DOCS_OUTPUT}
  WORKING_DIRECTORY ${DECLIB_DOCS_SOURCE}
  COMMENT "Building HTML documentation in ${DECLIB_DOCS_OUTPUT}"
  VERBATIM)
