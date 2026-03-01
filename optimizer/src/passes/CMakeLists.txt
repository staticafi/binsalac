
cmake_minimum_required(VERSION 3.10)
project(optimizer_passes)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/optimizer/passes
)

set(HEADER_FILES
  ${INCLUDE_DIR}/traits.hpp
)

set(SOURCE_FILES
)

add_subdirectory(analysis)
add_subdirectory(transformation)
add_subdirectory(debug)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})

