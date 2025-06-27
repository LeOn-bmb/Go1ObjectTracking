/**
  * @file perception.cc
  * @brief Ergänzte Version von example_putImagetrans.cc um FPS-Messung
  * @details Dieses Beispiel optimiert die Perfomance und sendet die Frames an den Jetson Xavier NX.
  * 
  * Basierend auf:
  * - example_putImagetrans.cc
  *   @author SunMingzhe
  *   @date 2021.12.07
  *   @version 1.0.1
  * 
  * @copyright Copyright (c) 2020-2021, Hangzhou Yushu Technology Stock CO.LTD. All Rights Reserved.
  * 
  * @copyright Zusätzliche Modifikationen (c) 2025, [LeOn-bmb].
  * Modifikationen:
  * - Hinzufügen einer FPS-Messung- und Optimierung
  * Datum: 11.07.2025
  */

#include <UnitreeCameraSDK.hpp>
#include <unistd.h>

int main(int argc, char *argv[])
{
   
    UnitreeCamera cam("trans_rect_config.yaml"); ///< init camera by device node number
    if(!cam.isOpened())   ///< get camera open state
        exit(EXIT_FAILURE);   
    cam.startCapture(true,false); ///< disable share memory sharing and able image h264 encoding

    usleep(500000);

    // FPS-Messung: Zeit- und Zähler-Setup
    auto last_fps_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    int seconds_elapsed = 0;
    int fps_outputs = 0; // Anzahl der ausgegebenen FPS-Werte


    while(cam.isOpened())
    {
        cv::Mat left, right;
        if(!cam.getRectStereoFrame(left, right))
        {
            usleep(1000);
            continue;
        }

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

        char key = cv::waitKey(10);
        if(key == 27) // press ESC key
           break;
    }
    
    cam.stopCapture(); ///< stop camera capturing
    
    return 0;
}