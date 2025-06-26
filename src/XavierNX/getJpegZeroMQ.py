import zmq
import cv2
import numpy as np
import struct
import time


# FPS-Messung Setup
last_fps_time = time.time()
frame_count = 0
seconds_elapsed = 0
fps_outputs = 0     # Anzahl der ausgegebenen FPS-Werte

# ZeroMQ Server Init
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")  # auf Verbindung warten

print("Empfänger bereit...")

while True:
    # Nachricht erhalten
    message = socket.recv()
    # Header extrahieren
    header_size = struct.calcsize("II")  # 8 Byte
    stereo_size, depth_size = struct.unpack("II", message[:header_size])

    # Daten extrahieren
    stereo_data = message[header_size:header_size + stereo_size]
#    depth_data = message[header_size + stereo_size:]

    # Stereo JPEG decodieren
    stereo_array = np.frombuffer(stereo_data, dtype=np.uint8)
    stereo_img = cv2.imdecode(stereo_array, cv2.IMREAD_COLOR)

    # Depth in Numpy Array wandeln
#    depth_array = np.frombuffer(depth_data, dtype=np.uint16)

#    depth_height = stereo_img.shape[0]
#    depth_width = depth_array.size // depth_height
#    depth_img = depth_array.reshape((depth_height, depth_width))


    # Anzeige
    #cv2.imshow("Stereo Image", stereo_img)

    # Optional: Depth normalisieren zum Anzeigen
#    depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #cv2.imshow("Depth Image", depth_display)

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


    # Abbruch
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
socket.close()
context.term()
