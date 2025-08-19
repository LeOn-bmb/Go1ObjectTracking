/**
  * @file perception.cc
  * @brief Ergänzte Version von example_getRectFrame.cc zum Senden via ZeroMQ
  * @details Dieses Beispiel optimiert die Perfomance und sendet die Frames via ZeroMQ an den Jetson Xavier NX.
  * @author LeOn-bmb
  * 
  * Basierend auf:
  *   example_getRectFrame.cc
  *   @author ZhangChunyang
  *   @date 2021.07.31
  *   @version 1.0.1
  * 
  * @copyright Copyright (c) 2020-2021, Hangzhou Yushu Technology Stock CO.LTD. All Rights Reserved.
  * 
  * @copyright Zusätzliche Modifikationen (c) 2025, [LeOn-bmb].
  * Modifikationen:
  * - Bildakquise und -vorbereitung (drehen)
  * - Rektifizierte Stereobilder getrennent via ZeroMQ senden
  * - FPS-Messung
  * - Debugginganzeigen (bei Bedarf)
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

    uint32_t right_size;
    uint32_t right_width;
    uint32_t right_height;
    uint32_t right_type;
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

    // ZeroMQ: PUSH to Xavier NX 
        zmq::context_t context(1);
        zmq::socket_t publisher(context, ZMQ_PUSH);
        publisher.set(zmq::sockopt::sndhwm, 4);
        publisher.connect("tcp://192.168.123.15:5555");

    while (cam.isOpened()) {
        cv::Mat left, right;

        if (!cam.getRectStereoFrame(left, right)) {     // get rectify left,right frame  
            usleep(1000);
            continue;
        }

        // Bilder um 180° drehen
        cv::flip(left, left, -1);        
        cv::flip(right, right, -1);
        
        // Datenmenge berechnen (li. Bild)
        size_t left_data_size = left.total() * left.elemSize();
        size_t right_data_size = right.total() * right.elemSize();


        // Debug-Anzeige
/*        cv::imshow("Left Frame Debug", left);
        cv::imshow("Right Frame Debug", right);

        // Frame-Size Debug
        if (!size_printed) {
        std::cout << "Left Frame Size: " << left.cols << "x" << left.rows << std::endl;
        std::cout << "Right Frame Size: " << right.cols << "x" << right.rows << std::endl;
        size_printed = true;
        }
*/
        // Header füllen
        Header header = {
            static_cast<uint32_t>(left_data_size),
            static_cast<uint32_t>(left.cols),
            static_cast<uint32_t>(left.rows),
            static_cast<uint32_t>(left.type()),

            static_cast<uint32_t>(right_data_size),
            static_cast<uint32_t>(right.cols),
            static_cast<uint32_t>(right.rows),
            static_cast<uint32_t>(right.type()),
        };

        // Nachricht bauen: HEADER | LEFT-DATEN | RIGHT-DATEN
        // Allokiere Nachricht
        size_t total_size = sizeof(header) + left_data_size + right_data_size;
        zmq::message_t message(total_size);
        uint8_t *ptr = reinterpret_cast<uint8_t*>(message.data());


        // Payload füllen
        memcpy(ptr, &header, sizeof(header));
        ptr += sizeof(header);
        memcpy(ptr, left.data, left_data_size);
        ptr += left_data_size;
        memcpy(ptr, right.data, right_data_size);

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
                
                if (cv::waitKey(1) == 27) break;  // ESC zum Beenden
    }

    cam.stopCapture();
    cam.stopStereoCompute();
    return 0;
}