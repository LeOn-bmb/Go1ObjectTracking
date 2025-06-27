import cv2
import time

# GStreamer-Pipeline für den Empfang des H.264-Streams per UDP (Port 9201 von .13)
gst_pipeline = (
    "udpsrc port=9201 caps=\"application/x-rtp, encoding-name=H264, payload=96\" "
    "! rtph264depay "
    "! h264parse "
    "! omxh264dec "
    "! videoconvert "
    "! appsink "
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Konnte GStreamer-Stream nicht öffnen!")
    exit()

print("✅ Stream wird empfangen...")

# FPS-Messung Setup
last_fps_time = time.time()
frame_count = 0
seconds_elapsed = 0
fps_outputs = 0     # Anzahl der ausgegebenen FPS-Werte

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Kein Frame empfangen")
        break

    # Das Stereo-Bild um 180 Grad drehen
    flipped_stereo_frame = cv2.flip(frame, -1)

    # Stereo-Bild aufteilen (horizontale Anordnung)
    height, width, channels = flipped_stereo_frame.shape
    width_half = width // 2

     # Sicherstellen, dass die Teilbilder eigene Speicherbereiche haben
    left_frame = flipped_stereo_frame[:, :width_half].copy()
    right_frame = flipped_stereo_frame[:, width_half:].copy()

    # Einzelbilder anzeigen (zum Debuggen)
    #cv2.imshow("Rectified Stereo", flipped_stereo_frame) 
    #cv2.imshow("Left Frame", left_frame)
    #cv2.imshow("Right Frame", right_frame)    

    frame_count += 1
    now = time.time()
    elapsed = now - last_fps_time
    if elapsed >= 1.0:
        seconds_elapsed += 1
        last_fps_time = now

        if seconds_elapsed >= 2 and fps_outputs < 15:
            print(f"Sekunde {seconds_elapsed - 1}: FPS = {frame_count}")
            fps_outputs += 1

        frame_count = 0

    # Beenden mit Taste 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()