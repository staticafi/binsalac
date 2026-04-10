cmake_minimum_required(VERSION 3.10)
project(liveness_analysis_chain_tests)

set(SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/liveness_analysis_chain_test.cpp
)

set(LIVENESS_PROGRAMS_SRC_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/programs
)

set(LIVENESS_PROGRAMS_DST_DIR
  ${CMAKE_CURRENT_BINARY_DIR}/programs
)

add_custom_target(liveness_analysis_chain_test_data ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${LIVENESS_PROGRAMS_DST_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${LIVENESS_PROGRAMS_SRC_DIR}
          ${LIVENESS_PROGRAMS_DST_DIR}
)

add_executable(liveness_analysis_chain_tests ${SOURCE_FILES})
add_dependencies(liveness_analysis_chain_tests liveness_analysis_chain_test_data)

target_compile_features(liveness_analysis_chain_tests PRIVATE cxx_std_20)

target_compile_definitions(liveness_analysis_chain_tests
  PRIVATE
    LIVENESS_ANALYSIS_PROGRAMS_DIR="${LIVENESS_PROGRAMS_DST_DIR}"
)

target_link_libraries(liveness_analysis_chain_tests
  PRIVATE
    optimizer_tests_common
    Catch2::Catch2WithMain
)

catch_discover_tests(liveness_analysis_chain_tests
  ADD_TAGS_AS_LABELS
)
