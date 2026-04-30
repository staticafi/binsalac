set(SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/available_copy_analysis_chain_test.cpp
)

set(AVAILABLE_COPY_PROGRAMS_SRC_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/programs
)

set(AVAILABLE_COPY_PROGRAMS_DST_DIR
  ${CMAKE_CURRENT_BINARY_DIR}/programs
)

add_custom_target(available_copy_analysis_chain_test_data ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${AVAILABLE_COPY_PROGRAMS_DST_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${AVAILABLE_COPY_PROGRAMS_SRC_DIR}
          ${AVAILABLE_COPY_PROGRAMS_DST_DIR}
)

add_executable(available_copy_analysis_chain_tests ${SOURCE_FILES})
add_dependencies(available_copy_analysis_chain_tests available_copy_analysis_chain_test_data)

target_compile_features(available_copy_analysis_chain_tests PRIVATE cxx_std_20)

target_compile_definitions(available_copy_analysis_chain_tests
  PRIVATE
  AVAILABLE_COPY_ANALYSIS_PROGRAMS_DIR="${AVAILABLE_COPY_PROGRAMS_DST_DIR}"
)

target_link_libraries(available_copy_analysis_chain_tests
  PRIVATE
    optimizer_tests_common
    Catch2::Catch2WithMain
)

catch_discover_tests(available_copy_analysis_chain_tests
  ADD_TAGS_AS_LABELS
)
