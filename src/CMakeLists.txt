# Hauptanwendung kompilieren
add_executable(send_perception ../src/NanoHead/perception.cc)
target_link_libraries(send_perception ${SDKLIBS})
target_include_directories(send_perception PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(zeroqt ../src/NanoHead/send_jpeg_framesZeroMQ.cc)
target_link_libraries(zeroqt ${SDKLIBS})
target_include_directories(zeroqt PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(rawzeroqt ../src/NanoHead/send_raw_framesZeroMQ.cc)
target_link_libraries(rawzeroqt ${SDKLIBS})
target_include_directories(rawzeroqt PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Example um Kalibrierungsdaten der Kameras zu erhalten
add_executable(example_getCalibParamsFile ../src/NanoHead/example_getCalibParamsFile.cc)
target_link_libraries(example_getCalibParamsFile ${SDKLIBS})