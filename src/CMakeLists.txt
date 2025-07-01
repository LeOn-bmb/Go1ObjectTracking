# Hauptanwendung kompilieren
add_executable(send_perception ../src/NanoHead/send_perception.cc)
target_link_libraries(send_perception ${SDKLIBS})
target_include_directories(send_perception PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Example um Kalibrierungsdaten der Kameras zu erhalten
add_executable(example_getCalibParamsFile ../src/NanoHead/example_getCalibParamsFile.cc)
target_link_libraries(example_getCalibParamsFile ${SDKLIBS})