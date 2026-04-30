set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/optimizer/programIR
)

set(HEADER_FILES
  ${INCLUDE_DIR}/ir_types.hpp
  ${INCLUDE_DIR}/program_repr.hpp

  ${INCLUDE_DIR}/program_ir.hpp
  ${INCLUDE_DIR}/function_ir.hpp
  ${INCLUDE_DIR}/basic_block_ir.hpp
  ${INCLUDE_DIR}/instruction_ir.hpp
  ${INCLUDE_DIR}/variable_ir.hpp
  ${INCLUDE_DIR}/constant_ir.hpp
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/program_ir.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/function_ir.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basic_block_ir.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/instruction_ir.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/variable_ir.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/constant_ir.cpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


