import zmq
import cv2
import numpy as np
import struct
import time

# Header-Format:
# uint32_t stereo_size;
# uint32_t stereo_width;
# uint32_t stereo_height;
# uint32_t stereo_type;

HEADER_FORMAT = "IIII"  #16 Byte 
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# ZeroMQ Server Init
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")  # auf Verbindung warten
print("Empfänger bereit...")

# FPS-Messung Setup
last_fps_time = time.time()
frame_count = 0
seconds_elapsed = 0
fps_outputs = 0     # Anzahl der ausgegebenen FPS-Werte

while True:
    # Nachricht erhalten
    message = socket.recv()

    # Header extrahieren
    header_data = message[:HEADER_SIZE]
        (
            stereo_size,
#            depth_size,
            stereo_width,
            stereo_height,
#            depth_width,
#            depth_height,
            stereo_type,
#            depth_type,
        ) = struct.unpack(HEADER_FORMAT, header_data)

    # Daten extrahieren
    stereo_data = message[HEADER_SIZE : HEADER_SIZE + stereo_size]
#    depth_data = message[HEADER_SIZE + stereo_size:]

    # StereoFrame als Numpy-Array
    stereo_array = np.frombuffer(stereo_data, dtype=np.uint8).reshape((stereo_height, stereo_width, 3))

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

    # FPS Zähler
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