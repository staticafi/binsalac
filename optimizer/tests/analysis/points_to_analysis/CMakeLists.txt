cmake_minimum_required(VERSION 3.10)
project(points_to_analysis_chain_tests)

set(SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/points_to_analysis_chain_test.cpp
)

set(POINTS_TO_PROGRAMS_SRC_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/programs
)

set(POINTS_TO_PROGRAMS_DST_DIR
  ${CMAKE_CURRENT_BINARY_DIR}/programs
)

add_custom_target(points_to_analysis_chain_test_data ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${POINTS_TO_PROGRAMS_DST_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${POINTS_TO_PROGRAMS_SRC_DIR}
          ${POINTS_TO_PROGRAMS_DST_DIR}
)

add_executable(points_to_analysis_chain_tests ${SOURCE_FILES})
add_dependencies(points_to_analysis_chain_tests points_to_analysis_chain_test_data)

target_compile_features(points_to_analysis_chain_tests PRIVATE cxx_std_20)

target_compile_definitions(points_to_analysis_chain_tests
  PRIVATE
    POINTS_TO_ANALYSIS_PROGRAMS_DIR="${POINTS_TO_PROGRAMS_DST_DIR}"
)

target_link_libraries(points_to_analysis_chain_tests
  PRIVATE
    optimizer_tests_common
    Catch2::Catch2WithMain
)

catch_discover_tests(points_to_analysis_chain_tests
  ADD_TAGS_AS_LABELS
)
