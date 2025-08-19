/**
  * @file getCalibParamsFile.cc
  * @details This Skript get camera internal parameters and rectified intrinsics
  * @author LeOn-bmb
  * @date  2025.08.18
  */

#include <UnitreeCameraSDK.hpp>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    // Kamera initialisieren
    UnitreeCamera cam("stereo_camera_config.yaml");
    if(!cam.isOpened()){
        std::cerr << "Failed to open camera" << std::endl;
        return EXIT_FAILURE;
    }

    cam.startCapture();  
    usleep(100000); // etwas warten, bis Parameter geladen sind

    // Parameter-Container
    std::vector<cv::Mat> leftParams;
    std::vector<cv::Mat> rightParams;

    if(!cam.getCalibParams(leftParams, false)){
        std::cerr << "Failed to get left calibration parameters" << std::endl;
        return EXIT_FAILURE;
    }
    if(!cam.getCalibParams(rightParams, true)){
        std::cerr << "Failed to get right calibration parameters" << std::endl;
        return EXIT_FAILURE;
    }
    
    // YAML-Datei schreiben
    cv::FileStorage fs("camCalibParams.yaml", cv::FileStorage::WRITE);

    fs << "LeftIntrinsic"   << leftParams[0];
    fs << "LeftDistortion"  << leftParams[1];
    fs << "LeftXi"          << leftParams[2];
    fs << "LeftRotation"    << leftParams[3];
    fs << "LeftTranslation" << leftParams[4];
    fs << "LeftKFE"         << leftParams[5];

    fs << "RightIntrinsic"   << rightParams[0];
    fs << "RightDistortion"  << rightParams[1];
    fs << "RightXi"          << rightParams[2];
    fs << "RightRotation"    << rightParams[3];
    fs << "RightTranslation" << rightParams[4];
    fs << "RightKFE"         << rightParams[5];

    fs.release();

    std::cout << "Calibration parameters saved to camCalibParams.yaml" << std::endl;
    usleep(100000);
    cam.stopCapture();
    return 0;
}