set(THIS_TARGET_NAME llvm2sala)

file(GLOB "${THIS_TARGET_NAME}_HPP" "./*.hpp")
file(GLOB "${THIS_TARGET_NAME}_CPP" "./*.cpp")

add_executable(${THIS_TARGET_NAME}
    "${${THIS_TARGET_NAME}_HPP}"
    "${${THIS_TARGET_NAME}_CPP}"
    )

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
else()
    include_directories(${LLVM_INCLUDE_DIRS})
endif()

target_link_libraries(${THIS_TARGET_NAME}
    sala
    utility
    llvmutl
    ${BOOST_LIST_OF_LIBRARIES_TO_LINK_WITH}
    ${LLVM_LIST_OF_LIBRARIES_TO_LINK_WITH}
    )

# set_target_properties(${THIS_TARGET_NAME} PROPERTIES
#     DEBUG_OUTPUT_NAME "${THIS_TARGET_NAME}_${CMAKE_SYSTEM_NAME}_Debug"
#     RELEASE_OUTPUT_NAME "${THIS_TARGET_NAME}_${CMAKE_SYSTEM_NAME}_Release"
#     RELWITHDEBINFO_OUTPUT_NAME "${THIS_TARGET_NAME}_${CMAKE_SYSTEM_NAME}_RelWithDebInfo"
#     )

install(TARGETS ${THIS_TARGET_NAME} DESTINATION ".")
if(WIN32)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/zlibd1.dll OPTIONAL DESTINATION .)
endif()
