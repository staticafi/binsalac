set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/utils/points_to
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/basic_block_may_state_store.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/defines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/may_value.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transfer_may.cpp
)

set(HEADER_FILES
  ${INCLUDE_DIR}/defines.hpp
  ${INCLUDE_DIR}/import.hpp
  ${INCLUDE_DIR}/nodes.hpp

  ${INCLUDE_DIR}/basic_block_may_state_store.hpp
  ${INCLUDE_DIR}/may_value.hpp
  ${INCLUDE_DIR}/transfer_may.hpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


