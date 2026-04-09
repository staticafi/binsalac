cmake_minimum_required(VERSION 3.10)
project(utils_tests)

set(SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/dynamic_bitset_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/sparse_map_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/sparse_set_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/type_list_compilation_test.cpp
)

add_executable(utils_tests ${SOURCE_FILES})
target_compile_features(utils_tests PRIVATE cxx_std_20)

target_link_libraries(utils_tests
  PRIVATE
    optimizer_tests_common
    Catch2::Catch2WithMain
)

catch_discover_tests(utils_tests
  ADD_TAGS_AS_LABELS
)
