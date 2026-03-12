cmake_minimum_required(VERSION 3.10)
project(optimizer_lib)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/optimizer/utils
)

set(HEADER_FILES
)

set(SOURCE_FILES
)

add_subdirectory(points_to)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})

