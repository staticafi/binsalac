cmake_minimum_required(VERSION 3.10)
project(optimizer_program)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/optimizer/analysis
)

set(HEADER_FILES
  ${INCLUDE_DIR}/points_to_query.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/points_to_query.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


