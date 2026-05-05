set(THIS_TARGET_NAME optimizer)

set(INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/optimizer
)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.cpp
)

set(HEADER_FILES
    ${INCLUDE_DIR}/optimizer.hpp
)

add_library(${THIS_TARGET_NAME} STATIC ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(${THIS_TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_subdirectory(query)
add_subdirectory(passes)
add_subdirectory(programIR)
add_subdirectory(translation)
add_subdirectory(utils)

target_link_libraries(${THIS_TARGET_NAME}
    sala
    utility
    ${BOOST_LIST_OF_LIBRARIES_TO_LINK_WITH}
)
