cmake_minimum_required(VERSION 3.24)
project(DecLibPP VERSION 0.1.0 LANGUAGES CXX)

option(DECLIB_BUILD_DOCS "Build the Sphinx/Doxygen documentation" OFF)

# ... library, Python bindings and tests go here ...

if(DECLIB_BUILD_DOCS)
  add_subdirectory(docs)
endif()
