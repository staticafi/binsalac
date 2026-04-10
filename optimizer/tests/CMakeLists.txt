cmake_minimum_required(VERSION 3.10)
project(optimizer_tests)
add_library(optimizer_tests_common INTERFACE)

message("Searching for Catch2 library ...")
find_package(Catch2 3 CONFIG REQUIRED)
message(STATUS "Catch2 version: ${Catch2_VERSION}")

target_include_directories(optimizer_tests_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(optimizer_tests_common INTERFACE
    optimizer
    sala
    utility
    ${BOOST_LIST_OF_LIBRARIES_TO_LINK_WITH}
)

include(Catch)
include(CTest)

add_subdirectory(translation)
add_subdirectory(utils)
add_subdirectory(programIR)
add_subdirectory(analysis)

