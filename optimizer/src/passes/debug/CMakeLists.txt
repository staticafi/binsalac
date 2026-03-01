cmake_minimum_required(VERSION 3.10)
project(optimizer_passes_debug)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/passes/debug
)

set(HEADER_FILES
  ${INCLUDE_DIR}/serialize_points_to.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_points_to.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


