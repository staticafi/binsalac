cmake_minimum_required(VERSION 3.10)
project(optimizer_passes_transformation)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/passes/transformation
)

set(HEADER_FILES
  ${INCLUDE_DIR}/merge_constants.hpp
  ${INCLUDE_DIR}/remove_indirections.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/merge_constants.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/remove_indirections.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


