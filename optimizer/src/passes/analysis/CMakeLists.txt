cmake_minimum_required(VERSION 3.10)
project(optimizer_lib)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/passes/analysis
)

set(HEADER_FILES
  ${INCLUDE_DIR}/local_points_to_analysis.hpp
  ${INCLUDE_DIR}/global_points_to_analysis.hpp
  ${INCLUDE_DIR}/available_copy_analysis.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/local_points_to_analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/global_points_to_analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/available_copy_analysis.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


