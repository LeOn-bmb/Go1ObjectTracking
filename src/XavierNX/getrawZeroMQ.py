import zmq
import cv2
import numpy as np
import struct
import time

# Header: uint32 left_size, left_width, left_height, left_type
HEADER_FORMAT = "IIII"
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

# Nur den neuesten Frame verarbeiten
def get_latest_message(sock):
    message = None
    while True:
        try:
            message = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break  # Keine neueren mehr im Puffer
    return message

while True:
    # Nachricht erhalten
    message = get_latest_message(socket)
    if message is None:
        time.sleep(0.001)
        continue

    # Header extrahieren
    header_data = message[:HEADER_SIZE]
    (
            left_size,
            left_width,
            left_height,
            left_type
#            depth_size,
#            depth_width,
#            depth_height,
#            depth_type,
    ) = struct.unpack(HEADER_FORMAT, header_data)

    # Bilddaten extrahieren
    left_data = message[HEADER_SIZE:HEADER_SIZE + left_size]
#    depth_data = message[HEADER_SIZE + stereo_size:]

    # Bild-Typ bestimmen (hier: CV_8UC3)
    if left_type == cv2.CV_8UC3:
        dtype = np.uint8
        img = np.frombuffer(left_data, dtype=dtype).reshape((left_height, left_width, 3))
    else:
        print(f"Unbekannter left_type: {left_type}")
        continue

    # Depth als Numpy-Array
#    depth_array = np.frombuffer(depth_data, dtype=np.uint16)

#    depth_height = stereo_img.shape[0]
#    depth_width = depth_array.size // depth_height
#    depth_img = depth_array.reshape((depth_height, depth_width))

    # Debug-Anzeige
#    cv2.imshow("Left Frame", img)

    # Optional: Depth normalisieren zum Anzeigen
#    depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #cv2.imshow("Depth Image", depth_display)

    # FPS-Zähler
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