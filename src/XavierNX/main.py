from models.yolov8trt_wrapper import YOLOv8TensorRT
import zmq
import cv2
import numpy as np
import struct
import time
import argparse
import os

# --- FPS-Messung Setup ---
last_fps_time = time.time()
frame_count = 0
fps_outputs = 0       # Anzahl der ausgegebenen FPS-Werte
seconds_elapsed = 0
fps_list = []         # Array-Speicher für FPS-Werte

size_printed = 0      #Frame-Size Debug einmalig

# Header: uint32 left_size, left_width, left_height, left_type, right_size, right_width, right_height, right_type
HEADER_FORMAT = "IIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# --- ZeroMQ Server Init ---
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")  # auf Verbindung warten
print("Empfänger bereit...")

# Parser-Argumente definieren
parser = argparse.ArgumentParser(description="Empfängt Bilder, führt Objekterkennung durch und führt optional Debugfunktionen per Tastendruck durch.")
parser.add_argument('--debug-img', metavar='imgname', 
                    help='Dateiname für das zu speichernde Bild, dann wählbar l(eft), r(ight), d(isparity), c(ombined)')
parser.add_argument('--debug-view', choices=['left', 'right', 'both'],
                    help='Debug-Anzeige: linkes/rechtes oder beide Bilder anzeigen')
parser.add_argument('--debug-disp', action='store_true',
                    help='Disparitätskarte anzeigen')
parser.add_argument('--debug-size', action='store_true',
                    help='Frame-Größen und Typen debuggen')
parser.add_argument('--debug-fps', action='store_true',
                    help='FPS-Messung ausgeben')
args = parser.parse_args()

# --- Init YOLOv8 TensorRT-Modell ---
model = YOLOv8TensorRT(
    engine_path="./models/trained_yolov8n.engine",
    input_width=480,
    input_height=416,
    conf_thresh=0.2,
    iou_thresh=0.4,
)
CLASS_NAMES = ["bottle", "can"]

# Kamera-Parameter (aus Kalibrierung!)
fx = 193.85525427753637            # Fokalweite in Pixeln
baseline_mm = 24.799092979022241   # Basislinie in mm

# --- Globale Konfigurationsparameter für StereoSGBM ---
MIN_DISPARITY = 0
NUM_DISPARITIES = 64  # Muss durch 16 teilbar sein
WINDOW_SIZE = 7       # 5–15 empfohlen, ungerade
NUM_CHANNELS = 1      # Graustufenbild
P1 = 8 * NUM_CHANNELS * WINDOW_SIZE ** 2
P2 = 32 * NUM_CHANNELS * WINDOW_SIZE ** 2

# --- Hilfsfunktion: Tiefenschätzung aus Bounding Box ---
def get_depth_from_bbox(bbox, disparity_map, fx, baseline_mm):
    # --- Disparitätsbereich für 10–100 cm berechnen ---
    z_min_mm = 100.0   # 10 cm
    z_max_mm = 1000.0  # 100 cm
    d_max = (fx * baseline_mm) / z_min_mm   # Disparität für 10 cm
    d_min = (fx * baseline_mm) / z_max_mm   # Disparität für 100 cm
    margin = 0.05  # 5 % Puffer
    d_max *= (1 + margin)
    d_min *= (1 - margin)
#    print(f"Valid disparity range: {d_min:.2f} - {d_max:.2f} px")

    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = disparity_map.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w-1, x2); y2 = min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    # Fokus auf mittleren Vertikalbereich
    y_start_middle = int(y1 + (y2 - y1) * 0.35)
    y_end_middle = int(y1 + (y2 - y1) * 0.70)
    if y_end_middle <= y_start_middle:
        return None

    disparity_roi = disparity_map[y_start_middle:y_end_middle, x1:x2]
    if disparity_roi.size == 0:
        return None

    # Disparitäten im erlaubten Bereich filtern
    valid_disparities = disparity_roi[
        (disparity_roi >= d_min) & (disparity_roi <= d_max)
    ]
    if valid_disparities.size == 0:
        return None
    
    median_disp = np.percentile(valid_disparities, 28)
    
    # Debug: Disparität zu BB
#    print(f"BBox: {bbox} | Median Disparity: {median_disp:.2f}")
    depth = (fx * baseline_mm) / median_disp    # Tiefenformel
    return depth

# --- Hilfsfunktion: Bounding Boxes zeichnen ---
def draw_detections(left_img, detections, class_names, disparity_map):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        depth = get_depth_from_bbox([x1, y1, x2, y2], disparity_map, fx, baseline_mm)

        label = f"{class_names[int(cls_id)]}: {conf:.2f}"
        if depth:
            label += f" | {depth:.1f} mm"

        cv2.rectangle(left_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(left_img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return left_img

# --- Debugfunktion: PNG-Aufnahme ---
def capture_frame(img, filename):
    # Sicherstellen, dass der Dateiname auf .png endet
    if not filename.lower().endswith('.png'):
        filename = os.path.splitext(filename)[0] + '.png'

    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        filename_with_timestamp = f"{base}_{timestamp}{ext}"
        cv2.imwrite(filename_with_timestamp, img)
        print(f"Bild gespeichert als: {filename_with_timestamp}")
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

# --- Debugfunktion: Anzeige (Left/Right) ---
def show_debug_views(left_img, right_img, view):
    if view == 'left':
        cv2.imshow("Left Frame", left_img)
    elif view == 'right':
        cv2.imshow("Right Frame", right_img)
    elif view == 'both':
        cv2.imshow("Left Frame", left_img)
        cv2.imshow("Right Frame", right_img)

# --- Debugfunktion: Frame-Größe & Typ ---
def print_debug_frame_info(left_width, left_height, right_width, right_height, left_type, right_type):
    print(f"Left Frame: {left_width}x{left_height} | Right Frame: {right_width}x{right_height}")
    print(f"Empfangener left_type: {left_type}, expected: {cv2.CV_8UC3}")
    print(f"Empfangener right_type: {right_type}, expected: {cv2.CV_8UC3}")

# --- Debugfunktion: Disparität anzeigen ---
def show_disparity_map(disparity):
    valid_disparities = disparity[disparity > 0]
    if valid_disparities.size == 0:
        cv2.imshow("Disparity", np.zeros_like(disparity))
        return

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
#    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imshow("Disparity", disp_vis)

# --- Debugfunktion: FPS-Anzeige ---
def update_fps_counter(frame_count, last_fps_time, seconds_elapsed, fps_outputs, fps_list):
    now = time.time()
    elapsed = now - last_fps_time

    if elapsed >= 1.0:
        seconds_elapsed += 1
        last_fps_time = now

        if args.debug_fps and seconds_elapsed >= 5 and fps_outputs < 15:
            print(f"Sekunde {seconds_elapsed - 4}: FPS = {frame_count}")
            fps_list.append(frame_count)
            fps_outputs += 1

        frame_count = 0

        if args.debug_fps and fps_outputs == 15:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"\n✅ Durchschnittliche FPS über 15 Sekunden: {avg_fps:.2f}")

    return frame_count, last_fps_time, seconds_elapsed, fps_outputs

# Nur den neuesten Frame verarbeiten
def get_latest_message(sock):
    message = None
    while True:
        try:
            message = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return message

# --- Hauptprogramm! ---
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

    # --- Disparität berechnen ---
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Vorverarbeitung
    # Sanftes Rauschen entfernen
    gray_left = cv2.bilateralFilter(gray_left, 7, 75, 75)
    gray_right = cv2.bilateralFilter(gray_right, 7, 75, 75)
    gray_left = cv2.GaussianBlur(gray_left, (5,5), 0)
    gray_right = cv2.GaussianBlur(gray_right, (5,5), 0)

    # Lokale Kontrastanpassung
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_left = clahe.apply(gray_left)
    gray_right = clahe.apply(gray_right)   

    # --- StereoSGBM-Kofig ---
    stereo = cv2.StereoSGBM_create(
        minDisparity = MIN_DISPARITY,
        numDisparities = NUM_DISPARITIES,
        blockSize = WINDOW_SIZE,
        P1 = P1,
        P2 = P2,
        disp12MaxDiff = 1,
        uniquenessRatio = 5,
        speckleWindowSize = 100,
        speckleRange = 16,
        preFilterCap = 31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    # Alle negativen Werte (ungültige Bereiche) auf 0 setzen
    disparity[disparity < 0] = 0

    # --- Inferenz mit YOLOv8 TensorRT ---
    detections = model.infer(left_img)

    # --- Ergebnisse zeichnen ---
    left_img = draw_detections(left_img, detections, CLASS_NAMES, disparity)

    # ✅ Debug-Anzeige
    if args.debug_view == 'left':
        cv2.imshow("Left Frame", left_img)
    elif args.debug_view == 'right':
        cv2.imshow("Right Frame", right_img)
    elif args.debug_view == 'both':
        cv2.imshow("Left Frame", left_img)
        cv2.imshow("Right Frame", right_img)
    # ✅ Frame-Size Debug
    if args.debug_size and size_printed == 0:
        print_debug_frame_info(left_width, left_height, right_width, right_height, left_type, right_type)
        size_printed = 1
    # ✅ Disparität Debug
    if args.debug_disp:
        show_disparity_map(disparity)
    # ✅ FPS-Zähler
    if args.debug_fps:
        frame_count += 1
        frame_count, last_fps_time, seconds_elapsed, fps_outputs = update_fps_counter(
            frame_count, last_fps_time, seconds_elapsed, fps_outputs, fps_list
        )

    key = cv2.waitKey(1) & 0xFF
    # ESC zum Abbrechen
    if key == 27:
        break

    # 'l', 'r' oder 'd' im debug-img Mode zum Bild aufnehmen
    elif key == ord('l') and args.debug_img:
        capture_frame(left_img, args.debug_img)
    elif key == ord('r') and args.debug_img:
        capture_frame(right_img, args.debug_img)
    elif key == ord('d') and args.debug_img:
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
#        disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        capture_frame(disp_vis, args.debug_img)
    elif key == ord('c') and args.debug_img:
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        # Konvertieren die Graustufen-Disparitätskarte in ein Farbbild
        disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        combined = np.hstack([left_img, disp_colored])
        capture_frame(combined, args.debug_img)

# --- Aufräumen ---
socket.close()
context.term()
cv2.destroyAllWindows()