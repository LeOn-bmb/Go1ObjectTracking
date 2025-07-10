import zmq
import cv2
import numpy as np
import struct
import time

# Header: uint32 left_size, left_width, left_height, left_type, depth_size, depth_width, depth_height, depth_type
HEADER_FORMAT = "IIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# --- ZeroMQ Server Init ---
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")  # auf Verbindung warten
print("Empfänger bereit...")

# FPS-Messung Setup
last_fps_time = time.time()
frame_count = 0
fps_outputs = 0# Anzahl der ausgegebenen FPS-Werte
seconds_elapsed = 0
# Array-Speicher für FPS-Werte
fps_list = []

size_printed = 0  #Frame-Size Debug einmalig

# Nur den neuesten Frame verarbeiten
def get_latest_message(sock):
    message = None
    while True:
        try:
            message = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
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
            left_type,
            depth_size,
            depth_width,
            depth_height,
            depth_type,
    ) = struct.unpack(HEADER_FORMAT, header_data)

    # Bilddaten extrahieren
    left_data = message[HEADER_SIZE:HEADER_SIZE + left_size]

    # Bild-Typ bestimmen (CV_8UC3)
    if left_type == cv2.CV_8UC3:
        dtype = np.uint8
        img = np.frombuffer(left_data, dtype=dtype).reshape((left_height, left_width, 3))
    else:
        print(f"Unbekannter left_type: {left_type}")
        continue
    # Debug-Anzeige
#    cv2.imshow("Left Frame", img)

#    if depth_size > 0:
#        # Bilddaten extrahieren
#        depth_data = message[HEADER_SIZE + left_size:]
#        depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_height, depth_width))
#
#        # Depth normalisieren zum Anzeigen und Debug-Anzeige
#        depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
#        cv2.imshow("Depth Frame", depth_colored)

    # Frame-Size Debug
#    if size_printed == 0:
#        print(f"Left Frame Size: {left_width} x {left_height}")
#        size_printed = 1

    # --- FPS-Zähler ---
    frame_count += 1
    now = time.time()
    elapsed = now - last_fps_time
    if elapsed >= 1.0:
        seconds_elapsed += 1
        last_fps_time = now

        if seconds_elapsed >= 5 and fps_outputs < 15:
            print(f"Sekunde {seconds_elapsed - 4}: FPS = {frame_count}")
            fps_outputs += 1
        frame_count = 0

        # Durchschnitt ausgeben
        if fps_outputs == 15:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"\n✅ Durchschnittliche FPS über 15 Sekunden: {avg_fps:.2f}")
            break

    # ESC zum Abbrechen
    if cv2.waitKey(1) & 0xFF == 27:
        break

socket.close()
context.term()
cv2.destroyAllWindows()