cmake_minimum_required( VERSION 2.8 )
project(Go1ObjectTracking)

# Output-Verzeichnis der ausführbaren Datei
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/)

find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIR})
    message(STATUS "OpenCV ${OpenCV_VERSION} FOUND")
    message(STATUS ${OpenCV_LIBS})
    if(${OpenCV_VERSION_MAJOR} EQUAL 4)
        add_definitions(-DOPENCV_VERSION_4)
    elseif(${OpenCV_VERSION_MAJOR} LESS 4)
        add_definitions(-DOPENCV_VERSION_3)
    endif()
else()
    message(FATAL_ERROR "OpenCV Library Not Found")
endif()

# ZeroMQ / cppzmq
find_package(cppzmq REQUIRED)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib/amd64/)
link_directories(${PROJECT_SOURCE_DIR}/lib/amd64/)
else()
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib/arm64/)
link_directories(${PROJECT_SOURCE_DIR}/lib/arm64/)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
include_directories(${PROJECT_SOURCE_DIR}/include)

# Alle Bibliotheken zusammenführen
set(SDKLIBS
    unitree_camera
    tstc_V4L2_xu_camera
    udev
    systemlog
    ${OpenCV_LIBS}
    cppzmq           # C++ Header-only
    zmq              # ZeroMQ Bibliothek
)


# Anwendung aus SDK einbinden
add_subdirectory(${PROJECT_SOURCE_DIR}/src)