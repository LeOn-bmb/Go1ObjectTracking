"""
Hauptskript:
- Empfängt Bilder der Stereokamera (links & rechts) vom Jetson Nano
- Erkennt Objekte und zeichnet Bounding Boxen
- Bestimmt die Distanz zu den Objekten mittels Triangulation und Stereodisparität
- Bietet optional Debug-Ausgaben (Anzeige der Bilder oder Disparitätskarte, Aufzeichnung von Frames)
- Misst die FPS
- Sendet relevante, geglättete Bewegungsdaten an den Raspberry Pi
"""

from models.yolov8trt_wrapper import YOLOv8TensorRT
import zmq
import cv2
import math
import numpy as np
import struct
import time
import argparse
import os
import json

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

# --- ZeroMQ Context + Sockets ---
context = zmq.Context()

# --- ZeroMQ (PULL vom Nano) ---
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.RCVHWM, 4)
socket.setsockopt(zmq.LINGER, 0)
socket.bind("tcp://*:5555")  # auf Verbindung warten

# Poller erzeugen
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

print("Empfänger bereit...")

# --- ZeroMQ (PUSH zum Raspberry) ---
robot_socket = context.socket(zmq.PUSH)
robot_socket.setsockopt(zmq.SNDHWM, 4)   # max queued messages before dropping
robot_socket.setsockopt(zmq.LINGER, 0)   # kein blockierendes Schließen
robot_socket.connect(f"tcp://192.168.123.161:5560")

# Sende- / Glättungs-Konfiguration
SMOOTH_ALPHA = 0.5      # EWMA Faktor 0..1 / höher = reaktiver, niedriger = ruhiger

# global smoothing state (replace per-class smoothing if desired)
last_committed = None           # {'u_px': float, 'z_mm': float, 'class_id': int, 'ts': float}
RESET_Z_THRESHOLD_MM = 300.0    # wenn neuer z mehr ist, dann Reset

# --- Parser-Argumente definieren ---
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
    conf_thresh=0.25,
    iou_thresh=0.4,
)
CLASS_NAMES = ["bottle", "can"]

# Update-Intervall der Objekterkennung
DETECTION_INTERVAL = 4         # Inferenz nur jede N Frames
MAX_STALE_INFERENCES = 15        # wie viele Inferenz-Zyklen stale reuse erlaubt
last_valid_detections = None    #  zuletzt erfolgreiche (nicht-leere) detections (list)
last_sent_payload = None
missed_inference_count = 0      # wie viele Inferenz-Zyklen in Folge leer waren

# --- Kamera-Parameter (aus Kalibrierung!) ---
# Kalibrierungsdatei laden
fs = cv2.FileStorage("camCalibParams.yaml", cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise SystemExit("[FATAL] Konnte camCalibParams.yaml nicht öffnen. Pfad prüfen.")

# linke Intrinsic-Matrix und Translationsvektor auslesen
left_intrinsic_node = fs.getNode("LeftIntrinsicMatrix")
translation_node = fs.getNode("Translation")

if left_intrinsic_node.empty() or translation_node.empty():
    fs.release()
    raise SystemExit("[FATAL] Fehlende Knoten in camCalibParams.yaml: 'LeftIntrinsicMatrix' oder 'Translation'.")

left_intrinsic = left_intrinsic_node.mat()
translation = translation_node.mat()
fs.release()

# Einzelwerte extrahieren
fx = float(left_intrinsic[0, 0])      # Fokalweite in Pixeln
cx = float(left_intrinsic[0, 2])
if args.debug_size:
    print("Left camera intrinsics:")
    print(f"fx = {fx}, cx = {cx}")

baseline_mm = abs(translation[0, 0])    # Basislinie in mm
if args.debug_size:
    print(f"Baseline: {baseline_mm:.3f} mm")

# Update-Intervall der Tiefenmessung
DISPARITY_UPDATE_RATE = 6
disparity = None
frame_idx = -1      # Frame-Index (für Disparität + Inferenz scheduling)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# --- StereoSGBM-Kofiguration und Berechnung ---
stereo = cv2.StereoSGBM_create(
    minDisparity = 0,
    numDisparities = 64,        # Muss durch 16 teilbar sein
    blockSize = 9,              # 5–15 empfohlen, ungerade
    P1 = 8 * 1 * 7 ** 2,        # Graustufenbild
    P2 = 32 * 1 * 7 ** 2,       # Graustufenbild
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=16,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# --- Hilfsfunktion: Nur den neuesten Frame verarbeiten ---
def get_latest_message(sock, poller, timeout_ms=5):
    """
    Pollt die Socket für up to timeout_ms Millisekunden.
    Falls Daten vorhanden sind: leert die Queue und gibt die letzte Nachricht zurück.
    Sonst: None.
    """
    socks = dict(poller.poll(timeout_ms))
    if socks.get(sock) != zmq.POLLIN:
        return None

    # Queue leeren, letzte Nachricht zurückgeben
    last_msg = None
    while True:
        try:
            last_msg = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return last_msg

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
    
    # Bounding Box auf Bildgrenzen begrenzen
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w-1, x2); y2 = min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    # Nur mittleren Bereich der BB verwenden
    y_start_middle = int(y1 + (y2 - y1) * 0.35)
    y_end_middle = int(y1 + (y2 - y1) * 0.70)
    if y_end_middle <= y_start_middle:
        return None

    disparity_roi = disparity_map[y_start_middle:y_end_middle, x1:x2]
    disparity_roi = disparity_roi[::2, ::2]  # nur jedes 2. Pixel
    if disparity_roi.size == 0:
        return None

    # Disparitäten im erlaubten Bereich filtern
    valid_disparities = disparity_roi[
        (disparity_roi >= d_min) & (disparity_roi <= d_max)
    ]
    if valid_disparities.size == 0:
        return None
    
    median_disp    = np.median(valid_disparities)
    disp_clipped = np.clip(valid_disparities, median_disp*0.5, median_disp*1.5)
    chosen_disp = np.median(disp_clipped)

    # Debug: Disparität zu BB
#    print(f"BBox: {bbox} | Median Disparity: {chosen_disp:.2f}")

    # Tiefe in mm
    z_mm = (fx * baseline_mm) / chosen_disp    # Tiefenformel
    return z_mm

# --- Hilfsfunktion: Bounding Boxes inkl. Abstand (z_mm falls vorhanden) zeichnen ---
def draw_processed_detections(left_img, detections):
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        cls_name = det['class_name']
        conf = det['conf']
        z_mm = det.get('z_mm', None)

        label = f"{cls_name} {conf:.2f}"
        if z_mm is not None:
            label += f" | {z_mm:.0f} mm"

        cv2.rectangle(left_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(left_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return left_img

# --- Hilfsfunktion: Bounding Boxes aufbauen ---
def build_processed_detections(raw_dets, disparity_map, fx, baseline_mm, class_names):
    processed = []
    for det in raw_dets:
        x1, y1, x2, y2, conf, cls_id = det
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        # z_mm kann None sein
        z_mm = get_depth_from_bbox([x1, y1, x2, y2], disparity_map, fx, baseline_mm)

        processed.append({
            'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'conf': float(conf),
            'class_id': int(cls_id),
            'class_name': class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else f"class_{int(cls_id)}",
            'u_px': float(u),
            'v_px': float(v),
            'z_mm': float(z_mm) if z_mm is not None else None
        })
    return processed

# --- Hilfsfunktion: Koordinaten vorbereiten und senden ---
def prepare_and_send_from_processed(processed, robot_socket, img_w, cx_val, alpha=SMOOTH_ALPHA):
    """
    Sendet kompaktes JSON mit:
      - u_px: horizontale Mitte der BBox (geglättet) in Pixel
      - u_offset_px: u_px - cx (Pixel, positiv -> rechts)
      - u_norm: normalisierte Abweichung in [-1..1] bezogen auf halbe Bildbreite
      - angle_rad: Winkel von Bildmitte zur BB-mitte in radiant
      - angle_deg: Der Winkel in degrees
      - z_mm: Entfernung in mm (geglättet)
      - class_id, class_name, conf
    Gibt das gesendete payload (dict) zurück oder None.
    """
    global last_committed

    if not processed:
        return None

    # Zuerst nach gültiger Tiefe sortieren, sonst fallback auf Conf
    with_depth = [d for d in processed if d['z_mm'] is not None]
    if with_depth:
        nearest = min(with_depth, key=lambda x: x['z_mm'])
    else:
        nearest = max(processed, key=lambda x: x['conf'])  # fallback

    # rohe Messwerte
    u_raw = nearest['u_px']
    z_raw = nearest['z_mm']
    cls = nearest['class_id']
    conf = nearest['conf']

    # Globale EWMA Glättung
    if last_committed is None:
        u_s = u_raw
        z_s = z_raw
    else:
        # immer u_px glätten
        u_s = alpha * u_raw + (1.0 - alpha) * last_committed['u_px']

        # nur z_mm glätten, wenn valide
        if z_raw is None:
            z_s = last_committed['z_mm']
        elif last_committed['z_mm'] is None:
            z_s = z_raw
        elif abs(z_raw - last_committed['z_mm']) > RESET_Z_THRESHOLD_MM:
            z_s = z_raw  # großer Sprung -> sofort übernehmen
        else:
            z_s = alpha * z_raw + (1.0 - alpha) * last_committed['z_mm']

    # Update global state
    last_committed = {
        'u_px': float(u_s),
        'z_mm': float(z_s) if z_s is not None else None,
        'class_id': int(cls),
        'conf': float(conf),
        'ts': time.time()
    }

    # Ableitungen für Steuerung
    u_offset_px = float(u_s - cx_val)

    # Winkel (radiant und degree)
    angle_rad = math.atan2(u_offset_px, fx)
    angle_deg = math.degrees(angle_rad)

    # Normieren auf halbe Bildbreite
    u_norm = float(u_offset_px / (img_w / 2.0))

    # --- payload ---
    payload = {
        'detection': {
            'class_id': int(cls),
            'class_name': nearest.get('class_name', ''),
            'conf': float(conf),
            'u_px': float(u_s),
            'u_offset_px': u_offset_px,
            'angle_rad': angle_rad,
            'angle_deg': angle_deg,
        }
    }
    if z_s is not None:
        payload['detection']['z_mm'] = float(z_s)

    # Senden (non-blocking)
    try:
        robot_socket.send(json.dumps(payload).encode('utf-8'), flags=zmq.NOBLOCK)
        return payload
    except zmq.Again:
        # Leitung voll, Nachricht verworfen
        return None
    except Exception as e:
        print(f"Fehler beim Senden an Raspberry: {e}")
        return None

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
    # Disparität normalisieren für bessere Sichtbarkeit
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_vis_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imshow("Disparity", disp_vis_color)

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

# --- Hauptprogramm! ---
while True:
    # Nachricht erhalten
    message = get_latest_message(socket, poller, timeout_ms=15)
    if message is None:
        continue
    
    frame_idx += 1

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

    # --- Disparität nur alle X Frames berechnen ---
    if frame_idx % DISPARITY_UPDATE_RATE == 0:
        # --- Vorverarbeitung ---
        # Konvertieren zu Graustufenbilder
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) 
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Lokale Kontrastanpassung (CLAHE)
        gray_left = clahe.apply(gray_left)
        gray_right = clahe.apply(gray_right)

        # Kantenerhaltendes Glätten (Bilateral)
        gray_left = cv2.bilateralFilter(gray_left, 5, 50, 50)
        gray_right = cv2.bilateralFilter(gray_right, 5, 50, 50)

        # --- Disparität berechnen ---
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Nur positive Werte behalten
        disparity[disparity < 0] = 0.0

        # --- Nachbearbeitung ---
        # Leichter Bilateral-Filter
        disparity = cv2.bilateralFilter(disparity, d=7, sigmaColor=2.0, sigmaSpace=7)

        # Median-Filter (gegen Ausreißer) auf 16U anwenden
        disp16 = (disparity * 16.0).astype(np.uint16)
        disp16 = cv2.medianBlur(disp16, 5)  #3 oder 5
        disparity  = disp16.astype(np.float32) / 16.0

    # Sicherstellen, dass disparity nicht None ist, bevor es verwendet wird
    if disparity is None:
        continue

    # --- YOLO-Inferenz mit YOLOv8 TensorRT nur jede DETECTION_INTERVAL Frames + stale reuse ---
    detections_for_processing = []  # Liste zum Senden und Zeichnen

    if frame_idx % DETECTION_INTERVAL == 0:
        try:
            raw_dets = model.infer(left_img)  # tatsächliche Inferenz
        except Exception as e:
            raw_dets = []
            print(f"[WARN] model.infer() failed: {e}")

        if raw_dets:
            # processed Detections einmalig bauen
            processed = build_processed_detections(raw_dets, disparity, fx, baseline_mm, CLASS_NAMES)

            if processed:
                # neue Detections vorhanden -> akzeptieren
                last_valid_detections = processed
                missed_inference_count = 0
                detections_for_processing = processed
            else:
                # Inferenz lieferte nichts -> erhöhten miss-counter
                missed_inference_count += 1
                # fallback: stale reuse (nur solange miss count im Limit)
                if last_valid_detections is not None and missed_inference_count <= MAX_STALE_INFERENCES:
                    detections_for_processing = last_valid_detections
                else:
                    detections_for_processing = []
        else:
            missed_inference_count += 1
            # Zwischenframe: verwenden der zuletzt gültigen Detections (falls im stale-Limit)
            if last_valid_detections is not None and missed_inference_count <= MAX_STALE_INFERENCES:
                detections_for_processing = last_valid_detections
            else:
                detections_for_processing = []
    else:
        if last_valid_detections is not None and missed_inference_count <= MAX_STALE_INFERENCES:
            detections_for_processing = last_valid_detections
        else:
            detections_for_processing = []

    # --- Wenn detections_for_processing vorhanden, berechne & sende Payload einmal ---
    if detections_for_processing:
        payload = prepare_and_send_from_processed(detections_for_processing, robot_socket, img_w=left_width, cx_val=cx, alpha=SMOOTH_ALPHA)
        if payload is not None:
            last_sent_payload = payload
            # wenn wir erfolgreich gesendet haben, kann missed_inference_count zurückgesetzt werden
            missed_inference_count = 0
    else:
        # keine aktuellen Detections: stale resend (payload-wiederholung) innerhalb Limit
        if last_sent_payload is not None and missed_inference_count <= MAX_STALE_INFERENCES:
            try:
                robot_socket.send(json.dumps(last_sent_payload).encode('utf-8'), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                print(f"Fehler beim Senden (stale): {e}")
        else:
            # keine Detections und stale-Limit überschritten -> "kein Ziel" senden
            empty_payload = {"detection": None}
            try:
                robot_socket.send(json.dumps(empty_payload).encode('utf-8'), flags=zmq.NOBLOCK)
                last_sent_payload = empty_payload
            except zmq.Again:
                pass
            except Exception as e:
                print(f"Fehler beim Senden (kein Ziel): {e}")

    # --- Ergebnisse zeichnen (immer mit detections_for_processing, damit Anzeige konsistent ist) ---
    left_img = draw_processed_detections(left_img, detections_for_processing)

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

    # --- Tastendruck für Debug-Aufnahmen ---
    key = cv2.waitKey(1) & 0xFF
    if args.debug_img and key in [ord('l'), ord('r'), ord('d'), ord('c')]:
        if key == ord('l'):
            capture_frame(left_img, args.debug_img)
        elif key == ord('r'):
            capture_frame(right_img, args.debug_img)
        elif key == ord('d') and disparity is not None:
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)
            disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            capture_frame(disp_vis, args.debug_img)
        elif key == ord('c'):
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)
            disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            combined = np.hstack([left_img, disp_colored])
            capture_frame(combined, args.debug_img)
    
    # --- Escape-Taste zum Beenden ---
    if key == 27:  # ESC
        break

# --- Aufräumen ---
socket.close()
robot_socket.close()
context.term()
cv2.destroyAllWindows()