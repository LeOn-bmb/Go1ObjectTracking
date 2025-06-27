/**
  * @file perception.cc
  * @brief Ergänzte Version von example_getRectFrame.cc um FPS-Messung, ...
  * @details Dieses Beispiel optimiert die Perfomance und sendet die Frames an den Jetson Xavier NX.
  * 
  * Basierend auf:
  * - example_getRectFrame.cc
  *   @author ZhangChunyang
  *   @date 2021.07.31
  *   @version 1.0.1
  * 
  * @copyright Copyright (c) 2020-2021, Hangzhou Yushu Technology Stock CO.LTD. All Rights Reserved.
  * 
  * @copyright Zusätzliche Modifikationen (c) 2025, [LeOn-bmb].
  * Modifikationen:
  * - Hinzufügen einer FPS-Optimierung durch:
  *     - Stereo Bild trennen (nur li. senden)
  *     - ...
  * - FPS-Messung
  * - ZeroMQ Versand
  * - Bild drehen
  * - Depth & Rect -Frame kombinieren
  * Datum: 26.07.2025
  */

#include <UnitreeCameraSDK.hpp>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <cstdint>

struct Header {
    uint32_t left_size;
    uint32_t left_width;
    uint32_t left_height;
    uint32_t left_type;
//    uint32_t depth_width;
//    uint32_t depth_height;
//    uint32_t depth_size;
//    uint32_t depth_type;
};

int main(int argc, char *argv[]) {

    UnitreeCamera cam("stereo_camera_config.yaml"); ///< init camera by device node number
    if(!cam.isOpened())   ///< get camera open state
        exit(EXIT_FAILURE);

    cam.startCapture(); ///< disable image h264 encoding and share memory sharing
    cam.startStereoCompute();
    usleep(500000);     // 0.5 Sekunden warten für Stabilisierung

    bool size_printed = false;  //Frame-Size Debug einmalig

    // FPS-Messung: Zeit- und Zähler-Setup
        auto last_fps_time = std::chrono::steady_clock::now();
        int frame_count = 0;
        int seconds_elapsed = 0;
        int fps_outputs = 0; // Anzahl der ausgegebenen FPS-Werte

    // ZeroMQ: Client Init 
        zmq::context_t context(1);
        zmq::socket_t publisher(context, ZMQ_PUSH);
        publisher.connect("tcp://192.168.123.15:5555");

    while (cam.isOpened()) {
//        cv::Mat left, right, depth;
//        std::chrono::microseconds t;
        cv::Mat left, right;

        if (!cam.getRectStereoFrame(left, right)) {     // get rectify left,right frame  
            usleep(1000);
            continue;
        }
/*        if (!cam.getDepthFrame(depth, true, t)) {       // get stereo camera depth image
            usleep(1000);
            continue;
        }
*/
        // Frame um 180° drehen
        cv::flip(left, left, -1);        

        // Datenmenge berechnen (li. Bild)
        size_t left_data_size = left.total() * left.elemSize();
        // Depthdaten
//        size_t depth_data_size = depth.total() * depth.elemSize();

        // Debug-Anzeige
/*        cv::imshow("Left Frame Debug", left);
        if (cv::waitKey(1) == 27) break;  // ESC zum Beenden
*/
        // Frame-Size Debug
        if (!size_printed) {
        std::cout << "Left Frame Size: " << left.cols << "x" << left.rows << std::endl;
        size_printed = true;
        }

        // Header füllen
        Header header = {
            static_cast<uint32_t>(left_data_size),
            static_cast<uint32_t>(left.cols),
            static_cast<uint32_t>(left.rows),
            static_cast<uint32_t>(left.type())
//            static_cast<uint32_t>(depth_data_size),
//            static_cast<uint32_t>(depth.cols),
//            static_cast<uint32_t>(depth.rows),
//            static_cast<uint32_t>(depth.type())
        };

        // Nachricht bauen: HEADER | LEFT-DATEN | DEPTH-DATEN
//        size_t total_size = sizeof(header) + left_data_size + depth_data_size;
        size_t total_size = sizeof(header) + left_data_size;
        zmq::message_t message(total_size);
        uint8_t *ptr = reinterpret_cast<uint8_t*>(message.data());

        // Header kopieren
        memcpy(ptr, &header, sizeof(header));
        ptr += sizeof(header);
        // Left kopieren
        memcpy(ptr, left.data, left_data_size);
/*        ptr += stereo_data_size;
        // Depth kopieren
        memcpy(ptr, depth.data, depth_data_size);
*/
        // Nachricht senden
        publisher.send(message, zmq::send_flags::none);

        // FPS-Zähler
                frame_count++;
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_time).count();
                if (elapsed >= 1) {
                    seconds_elapsed++;
                    last_fps_time = now;

                    if (seconds_elapsed >= 2 && fps_outputs < 15) {
                    std::cout << "Sekunde " << seconds_elapsed -1 << ": FPS = " << frame_count << std::endl;
                    fps_outputs++;
                    }
                    frame_count = 0;
                }
    }

    cam.stopCapture();
    cam.stopStereoCompute();
    return 0;
}