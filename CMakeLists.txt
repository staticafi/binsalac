message("Including the following salac's components to the build:")
add_subdirectory(./llvm2llvm)
    message("  llvm2llvm")
add_subdirectory(./llvm2sala)
    message("  llvm2sala")
add_subdirectory(./sala2sala)
    message("  sala2sala")

if(DEFINED BINSALAC_INSTALL_DIR)
    install(PROGRAMS
        "./salac.py"
        DESTINATION "${BINSALAC_INSTALL_DIR}")
    install(FILES
        "./README.md"
        "./LICENSE.txt"
        DESTINATION "${BINSALAC_INSTALL_DIR}")
endif(DEFINED BINSALAC_INSTALL_DIR)
