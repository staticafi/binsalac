cmake_minimum_required(VERSION 3.10)
project(optimizer_lib)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/optimizer/translation
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/sala_to_ir.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ir_to_sala.cpp
)

set(HEADER_FILES
  ${INCLUDE_DIR}/sala_to_ir.hpp
  ${INCLUDE_DIR}/ir_to_sala.hpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


