import zmq
import cv2
import numpy as np
import struct
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# ---------------------- Konfiguration ----------------------
ENGINE_PATH = "./models/yolov8n.engine"  # <-- Modell hier wählen
INPUT_WIDTH, INPUT_HEIGHT = 480, 416
CONF_THRESH = 0.3
IOU_THRESH = 0.4
# -----------------------------------------------------------

# --- TensorRT Setup ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Bindings und Speicherpuffer
input_idx = engine.get_binding_index("images") if "images" in engine else 0
output_idx = 1 - input_idx
input_shape = (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
input_size = trt.volume(input_shape) * np.float32().itemsize
d_input = cuda.mem_alloc(input_size)
host_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
# Set input shape in context
context.set_binding_shape(input_idx, input_shape)

# Output-Shape dynamisch auslesen
output_shape = tuple(context.get_binding_shape(output_idx))  # an Modell angepasst
output_size = trt.volume(output_shape) * np.float32().itemsize
host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
d_output = cuda.mem_alloc(output_size)
bindings = [int(d_input), int(d_output)]

# Preprocessing
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))    # von 400 x 464 (INPUT von der Go1 Kamera) zu 416 x 480 (YOLO-Optimiert)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img.ravel()

# FPS-Setup
last_fps_time = time.time()
frame_count = 0
fps_outputs = 0
seconds_elapsed = 0
# Array-Speicher für FPS-Werte
fps_list = []

size_printed = 0  #Frame-Size Debug einmalig

# --- ZeroMQ-Setup ---
HEADER_FORMAT = "IIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

context_zmq = zmq.Context()
socket = context_zmq.socket(zmq.PULL)
socket.bind("tcp://*:5555")
print("🚀 Empfänger bereit...")

def get_latest_message(sock):
    message = None
    while True:
        try:
            message = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return message

while True:
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
            print(f"❌ Fehler beim Umformen der Bilder: {e}")
            continue
    else:
        print(f"❌ Unbekannter OpenCV-Typ – Left: {left_type}, Right: {right_type}")
        continue

    # --- Inferenz starten ---
    host_input[:] = preprocess(left_img)
    cuda.memcpy_htod(d_input, host_input)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(host_output, d_output)


    # ✅ Frame-Size Debug
#    if size_printed == 0:
#        print(f"Left Frame: {left_width}x{left_height} | Right Frame: {right_width}x{right_height}")
#        print(f"[DEBUG] Empfangener left_type: {left_type}, expected: {cv2.CV_8UC3} ({cv2.CV_8UC3})")
#        print(f"[DEBUG] Empfangener right_type: {right_type}, expected: {cv2.CV_8UC3}")
#        size_printed = 1

    # --- Debugging Anzeige ---
    # Das an TensorRT übergebene Bild anzeigen
#    inference_frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
#    cv2.imshow("🟣 Inferenz-Input", inference_frame)

    # --- FPS-Zähler ---
    frame_count += 1
    now = time.time()
    elapsed = now - last_fps_time
    if elapsed >= 1.0:
        seconds_elapsed += 1
        last_fps_time = now

        if seconds_elapsed >= 5 and fps_outputs < 15:
            print(f"📸 Sekunde {seconds_elapsed - 4}: FPS (mit {os.path.basename(ENGINE_PATH)}) = {frame_count}")
            fps_list.append(frame_count)
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
context_zmq.term()
cv2.destroyAllWindows()