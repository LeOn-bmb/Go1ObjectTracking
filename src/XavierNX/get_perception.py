from models.yolov8trt_wrapper import YOLOv8TensorRT
import zmq
import cv2
import numpy as np
import struct
import time

# Header: uint32 left_size, left_width, left_height, left_type, right_size, right_width, right_height, right_type
HEADER_FORMAT = "IIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# --- ZeroMQ Server Init ---
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")  # auf Verbindung warten
print("Empfänger bereit...")

# --- Init YOLO-Model ---
model = YOLOv8TensorRT(
    engine_path="./models/trained_yolov8n.engine",
    input_width=480,
    input_height=416,
    conf_thresh=0.2,
    iou_thresh=0.4,
)
CLASS_NAMES = ["bottle", "can"]

# Funktion um die Bounding Boxes zu zeichnen
def draw_detections(left_img, detections, class_names):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        label = f"{class_names[int(cls_id)]}: {conf:.2f}"
        cv2.rectangle(left_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(left_img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return left_img

# --- FPS-Messung Setup ---
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
            right_size,
            right_width,
            right_height,
            right_type,
    ) = struct.unpack(HEADER_FORMAT, header_data)

    # Bilddaten extrahieren
    left_data = message[HEADER_SIZE:HEADER_SIZE + left_size]
    right_data = message[HEADER_SIZE + left_size : HEADER_SIZE + left_size + right_size]

    # Mapping OpenCV-Typ → (NumPy-Datentyp, Shape-Dimensionen)
    opencv_type_map = {
        cv2.CV_8UC1: (np.uint8, 1),
        cv2.CV_8UC3: (np.uint8, 3),
        cv2.CV_16UC1: (np.uint16, 1),
        cv2.CV_32FC1: (np.float32, 1),
    }

    if left_type in opencv_type_map and right_type in opencv_type_map:
        left_dtype, left_channels = opencv_type_map[left_type]
        right_dtype, right_channels = opencv_type_map[right_type]

        try:
            if left_channels == 1:
                left_img = np.frombuffer(left_data, dtype=left_dtype).reshape((left_height, left_width))
            else:
                left_img = np.frombuffer(left_data, dtype=left_dtype).reshape((left_height, left_width, left_channels))

            if right_channels == 1:
                right_img = np.frombuffer(right_data, dtype=right_dtype).reshape((right_height, right_width))
            else:
                right_img = np.frombuffer(right_data, dtype=right_dtype).reshape((right_height, right_width, right_channels))

        except ValueError as e:
            print(f"Fehler beim Umformen der Bilder: {e}")
            continue

    # --- Inferenz mit YOLOv8 TensorRT ---
    detections = model.infer(left_img)

    # Bounding Boxes zeichnen
    left_img = draw_detections(left_img, detections, CLASS_NAMES)

    # ✅ Debug-Anzeige (Left)
    cv2.imshow("Left Frame", left_img)
#    cv2.imshow("Right Frame", right_img)

    # ✅ Frame-Size Debug
#    if size_printed == 0:
#        print(f"Left Frame: {left_width}x{left_height} | Right Frame: {right_width}x{right_height}")
#        print(f"[DEBUG] Empfangener left_type: {left_type}, expected: {cv2.CV_8UC3}")
#        print(f"[DEBUG] Empfangener right_type: {right_type}, expected: {cv2.CV_8UC3}")
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
            fps_list.append(frame_count)
            fps_outputs += 1
        frame_count = 0

        # Durchschnitt ausgeben
        if fps_outputs == 15:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"\n✅ Durchschnittliche FPS über 15 Sekunden: {avg_fps:.2f}")

    # ESC zum Abbrechen
    if cv2.waitKey(1) & 0xFF == 27:
        break

socket.close()
context.term()
cv2.destroyAllWindows()