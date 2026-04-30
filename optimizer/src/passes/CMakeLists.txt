set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/optimizer/passes
)

set(HEADER_FILES
)

set(SOURCE_FILES
)

add_subdirectory(analysis)
add_subdirectory(transformation)
add_subdirectory(debug)

target_sources(optimizer PRIVATE ${SOURCE_FILES} ${HEADER_FILES})

