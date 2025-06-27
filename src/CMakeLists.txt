# Testanwendungen kompilieren
add_executable(gstream ../src/NanoHead/sendGStreamer.cc)
target_link_libraries(gstream ${SDKLIBS})
target_include_directories(gstream PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(zeroqt ../src/NanoHead/send_jpeg_framesZeroMQ.cc)
target_link_libraries(zeroqt ${SDKLIBS})
target_include_directories(zeroqt PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Hauptanwendung kompilieren
add_executable(rawzeroqt ../src/NanoHead/send_raw_framesZeroMQ.cc)
target_link_libraries(rawzeroqt ${SDKLIBS})
target_include_directories(rawzeroqt PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Example um Kalibrierungsdaten der Kameras zu erhalten
add_executable(example_getCalibParamsFile ../src/NanoHead/example_getCalibParamsFile.cc)
target_link_libraries(example_getCalibParamsFile ${SDKLIBS})