# Hauptanwendung kompilieren
add_executable(send_perception ../src/NanoHead/send_perception.cc)
target_link_libraries(send_perception ${SDKLIBS})
target_include_directories(send_perception PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Anwendung um Kalibrierungsdaten der Kameras zu erhalten
add_executable(getCalibParamsFile ../src/NanoHead/getCalibParamsFile.cc)
target_link_libraries(getCalibParamsFile ${SDKLIBS})