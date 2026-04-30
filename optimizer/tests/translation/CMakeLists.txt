set(SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/translation_test.cpp
)

add_executable(translation_tests
  ${SOURCE_FILES}
)
target_compile_features(translation_tests PRIVATE cxx_std_20)


target_link_libraries(translation_tests
  PRIVATE
    optimizer_tests_common
    Catch2::Catch2WithMain
)

catch_discover_tests(translation_tests
  ADD_TAGS_AS_LABELS
)
