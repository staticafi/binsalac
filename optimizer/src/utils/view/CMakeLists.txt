set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/utils/view
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/flattened_cfg_view.cpp
)

set(HEADER_FILES
  ${INCLUDE_DIR}/flattened_cfg_view.hpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})
