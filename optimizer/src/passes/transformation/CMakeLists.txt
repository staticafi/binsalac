set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/passes/transformation
)

set(HEADER_FILES
  ${INCLUDE_DIR}/dead_instruction_elimination.hpp
  ${INCLUDE_DIR}/dead_variables_elimination.hpp
  ${INCLUDE_DIR}/merge_constants.hpp
  ${INCLUDE_DIR}/propagate_copy.hpp
  ${INCLUDE_DIR}/remove_indirections.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/dead_instruction_elimination.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dead_variables_elimination.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/merge_constants.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/propagate_copy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/remove_indirections.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


