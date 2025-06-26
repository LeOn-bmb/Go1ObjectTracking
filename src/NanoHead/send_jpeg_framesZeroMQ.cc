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
  * - Hinzufügen einer FPS-Messung- und Optimierung
  * - ZeroMQ Versand
  * - Bilder kombinieren
  * - Bilder drehen
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
    uint32_t stereo_size;
//    uint32_t depth_size;
};

int main(int argc, char *argv[]){
    
    UnitreeCamera cam("stereo_camera_config.yaml"); ///< init camera by device node number
    if(!cam.isOpened())   ///< get camera open state
        exit(EXIT_FAILURE);

    cam.startCapture(); ///< disable image h264 encoding and share memory sharing
    cam.startStereoCompute();
    usleep(500000);

    // FPS-Messung: Zeit- und Zähler-Setup
        auto last_fps_time = std::chrono::steady_clock::now();
        int frame_count = 0;
        int seconds_elapsed = 0;
        int fps_outputs = 0; // Anzahl der ausgegebenen FPS-Werte

    // ZeroMQ Client Inititialisierung 
        zmq::context_t context(1);
        zmq::socket_t publisher(context, ZMQ_PUSH);
        publisher.connect("tcp://192.168.123.15:5555"); // Verbindung zur Empfänger-IP

    while(cam.isOpened()){
//        cv::Mat left, right, depth;
        cv::Mat left, right;
//        std::chrono::microseconds t;

        if(!cam.getRectStereoFrame(left,right)){ // get rectify left,right frame  
            usleep(1000);
            continue;
        }

/*        if(!cam.getDepthFrame(depth, true, t)){  // get stereo camera depth image 
            usleep(1000);
            continue;
        }
*/        
        // Stereo-Bilder kombinieren
        cv::Mat stereo;
        cv::hconcat(left, right, stereo); 
        // Dieses um 180° drehen
        cv::flip(stereo,stereo, -1);

        // Anzeigen des Frames
        //cv::imshow("Rectified Stereo", stereo);
        //cv::imshow("UnitreeCamera-Depth", depth);

        // JPEG-Komprimierung für geringere Bandbreite
        std::vector<uchar> stereo_buffer;
        cv::imencode(".jpg", stereo, stereo_buffer);

        // Depthdaten
//        size_t depth_data_size = depth.total() * depth.elemSize();

        // Nachricht bauen
        Header header = { 
            static_cast<uint32_t>(stereo_buffer.size()),
//            static_cast<uint32_t>(depth_data_size)
        };

        // Nachricht besteht aus: HEADER | JPEG-DATEN   //| DEPTH-DATEN
 //       size_t total_size = sizeof(header) + stereo_buffer.size() + depth_data_size;
        size_t total_size = sizeof(header) + stereo_buffer.size();
        zmq::message_t message(total_size);
        auto *ptr = reinterpret_cast<uint8_t*>(message.data());

        // Header kopieren
        memcpy(ptr, &header, sizeof(header));
        ptr += sizeof(header);
        // Stereo kopieren
        memcpy(ptr, stereo_buffer.data(), stereo_buffer.size());
//        ptr += stereo_buffer.size();
        // Depth kopieren
//        memcpy(ptr, depth.data, depth_data_size);

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
    
    cam.stopCapture(); ///< stop camera capturing
    cam.stopStereoCompute();

    return 0;
}

//Depth auskommentiert !!