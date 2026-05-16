set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/optimizer/utils/liveness
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/defines.cpp
)

set(HEADER_FILES
  ${INCLUDE_DIR}/defines.hpp
)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})


