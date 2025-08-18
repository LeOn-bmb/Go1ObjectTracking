Introduction
---

Dieses Projekt baut auf dem Unitree Camera SDK auf, das unter der Mozilla Public License 2.0 veröffentlicht wurde. Der vorhandene Quellcode wurde erweitert und angepasst, um neue Funktionalitäten zu realisieren.

Ziel dieses Projekts ist es, den Go1-Roboter von Unitree mit einem Echtzeit-Objekterkennungs- und Verfolgungssystem auszustatten. Dabei erkennt der Roboter bestimmte Objekte (z.B. Flaschen), bestimmt ihre Entfernung mittels aktiver Stereovision und steuert anschließend auf das nächste Ziel zu, bis ein definierter Sicherheitsabstand erreicht ist. Dieser Abstand wird dann dynamisch gehalten.

🧭 1. Overview
---

Dieses Verzeichnis stellt den vollständigen Software-Stack zur Verfügung, um den Go1-Roboter für folgende Aufgaben vorzubereiten:

- Objekterkennung mittels YOLO11n-Engine-Model (Inference optimiert mit TensorRT für maximale Performance auf Jetson-Plattformen)
- Tiefenmessung via Depth-Frames aus der Unitree Kamera (ermöglicht präzise Distanzberechnungen zu erkannten Objekten)
- Zielverfolgung durch Navigation bis zum Objekt mit aktivem Abstandsregler
- Modulare Architektur mit ZeroMQ für die Bildübertragung zwischen Kamera-Head (Jetson Nano) und Verarbeitungseinheit (Xavier NX)

Das System wurde für ressourcenbeschränkte Edge-Hardware wie den Jetson Xavier NX optimiert und unterstützt optimierte Objekterkennungs-Modelle über TensorRT. Shape (width, height) nah an RectFrame-Size der Go1 Kamera angepasst und YOLO-Kompatibel (teilbar durch 32).

🔧 2. Build-Time Dependencies (für Modellkonvertierung auf Host/XavierNX, Python 3.8)
---

- [Python3.8 oder höher](https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/) - erforderlich für ultralytics
- [Ultralytics](https://docs.ultralytics.com/de/quickstart/) - zum Laden und Exportieren von YOLO .pt-Modellen in .onnx
- [PyTorch](https://docs.ultralytics.com/de/guides/nvidia-jetson/#install-pytorch-and-torchvision) – automatisch mit ultralytics installiert (nur kompatibel mit x86 / nicht direkt auf Jetson)
- ONNX, ONNX-Simplifier – für ONNX-Export, falls simplify=True
- trtexec - CLI-Tool aus dem NVIDIA TensorRT Toolkit (wandelt .onnx → .engine um; befindet sich i. d. R. unter /usr/src/tensorrt/bin/trtexec)

🚀 2.1 Runtime Dependencies (auf Jetson Go1, Python 3.6)
---

- OpenCV (Version 4 oder höher) - für Bildverarbeitung
- CMake (Version 3.11 oder höher) - zum bauen von C-Anwendungen
- Python3 (Version 3.6 oder höher)
- [ZeroMQ](https://zeromq.org/get-started/) - leichtgewichtige Messaging-Library (für Bildstreaming vom Jetson Nano zum Xavier NX)
  
Nur auf Xavier NX erforderlich für Objekterkennung
---

- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA-Inferenz-Bibliothek, die speziell für NVIDIA GPUs die maximale Inferenz-Performance aus ONNX-Modellen herausholt (z. B. Version 7.1.3.0 unter JetPack 4.5)
- [PyCUDA](https://wiki.tiker.net/PyCuda/Installation/Linux/) – nützlich für Memory Binding, CUDA Streams mit TensorRT & Speicherverwaltung in GPU

📁 3. Build 
---

🔨 Bauen der C++-Anwendung auf dem Jetson Nano
```
cd Go1ObjectTracking;
mkdir build && cd build;
cmake ..;
make
```

🚀 4. Run 
---

🎥 Stereo-Kameras auf dem nano head auf blockierende Prozesse überprüfen:
```
v4l2-ctl --list-devices;
lsof /dev/video1;
kill -9 <PID>   # falls erforderlich
```

📤 Senden der Frames an den Jetson Xavier NX:
```
cd Go1ObjectTracking; 
./bin/send_perception
```

🏁 Hauptprogramm auf dem Xavier NX starten:
```
cd Go1ObjectTracking/src/XavierNX; 
python3 main.py
```