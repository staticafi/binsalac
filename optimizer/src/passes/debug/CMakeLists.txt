set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/passes/debug
)

set(HEADER_FILES
  ${INCLUDE_DIR}/dump_available_copy.hpp
  ${INCLUDE_DIR}/dump_liveness.hpp
  ${INCLUDE_DIR}/dump_points_to.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/dump_available_copy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dump_liveness.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dump_points_to.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


