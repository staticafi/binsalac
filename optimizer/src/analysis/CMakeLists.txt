cmake_minimum_required(VERSION 3.10)
project(optimizer_lib)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/optimizer/analysis
)

set(HEADER_FILES
  ${INCLUDE_DIR}/available_copy_query.hpp
  ${INCLUDE_DIR}/liveness_query.hpp
  ${INCLUDE_DIR}/points_to_query.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/available_copy_query.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/liveness_query.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/points_to_query.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


