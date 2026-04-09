cmake_minimum_required(VERSION 3.10)
project(program_ir_tests)

set(SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/program_ir_test.cpp
)

add_executable(program_ir_tests ${SOURCE_FILES})
target_compile_features(program_ir_tests PRIVATE cxx_std_20)

target_link_libraries(program_ir_tests
  PRIVATE
    optimizer_tests_common
    Catch2::Catch2WithMain
)

catch_discover_tests(program_ir_tests
  ADD_TAGS_AS_LABELS
)
