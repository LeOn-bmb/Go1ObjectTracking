Introduction
---
Dieses Projekt verwendet Code aus der Unitree Camera SDK, das unter der Mozilla Public License 2.0 lizenziert ist. Änderungen an diesem Verzeichnis wurden vorgenommen.

Dieses Verzeichnis verfolgt das Ziel den Go1 von Unitree dazu zu nutzen um Objekte zu erkennen und zu dem nahgelegensten dieser bis auf einen festgelegten Abstand zu navigieren und den Abstand zu halten.

1.Overview
---

The SDK allows depth and color streaming, and provides intrinsic calibration information. The library also offers pointcloud, depth image aligned to color image.

2.Dependencies 📦
---

- [OpenCV] (Version 4 oder höher, GStreamer-Unterstützung benötigt)
- [CMake] (Version 3.11 oder höher)
- [Python3.8](https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/)
- [ZeroMQ](https://zeromq.org/get-started/) - Live-Stream vom Nano Head zum Xavier NX als alternative zu GStreamer
- [Ultralytics](https://docs.ultralytics.com/de/quickstart/) - YOLO für die Echtzeit-Objekterkennung
- [ONNX](https://onnxruntime.ai/docs/install/) - CPU/GPU-Management für YOLO

3.Build 📁
---

```
cd Go1ObjectTracking;
mkdir build && cd build;
cmake ..;
make
```

4.Run 🚀
---

🎥 Stereo-Kameras auf dem nano head auf blockierende Prozesse überprüfen:
```
v4l2-ctl --list-devices;
lsof /dev/video1;
kill -9 <PID>   # falls erforderlich
```

Senden der Frames an den Jetson Xavier NX:
```
cd Go1ObjectTracking; 
./bin/send_perception
```

🏁 Hauptprogramm auf dem Xavier NX starten:
```
cd Go1ObjectTracking/src/XavierNX; 
python3 main.py
```