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
from models.sort import Sort
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

# Sende- / Glättungs-Konfiguration / z_mm, u_px abfedern
# EWMA Faktor 0..1 / höher = reaktiver, niedriger = ruhiger
ALPHA_U_TRACK = 0.6   # Glättung für u_px (horizontale Position)
ALPHA_Z_TRACK = 0.4   # Glättung für z_mm (Distanz/Disparität)

last_committed = None           # Globaler Glättungsstatus anhand letzter Werte
RESET_Z_THRESHOLD_MM = 250.0    # wenn neuer z mehr ist, dann Reset

# --- Parser-Argumente definieren ---
parser = argparse.ArgumentParser(description="Empfängt Bilder, führt Objekterkennung durch und führt optional Debugfunktionen per Tastendruck durch.")
parser.add_argument('--debug-img', metavar='imgname', 
                    help='Dateiname für das zu speichernde Bild, dann wählbar l(eft), r(ight), d(isparity), c(ombined)')
parser.add_argument('--debug-view', choices=['left', 'right', 'both'],
                    help='Debug-Anzeige: linkes/rechtes oder beide Bilder anzeigen')
parser.add_argument('--debug-disp', action='store_true',
                    help='Disparitätskarte anzeigen')
parser.add_argument('--debug-size', action='store_true',
                    help='Frame-Größen, Typen und Kalibrierungsparameter debuggen')
parser.add_argument('--debug-fps', action='store_true',
                    help='FPS-Messung ausgeben')
parser.add_argument('--debug-depth',
                    action='store_true',
                    help="Debug-Ausgaben für die Tiefenschätzung anzeigen.")
args = parser.parse_args()

# --- Init YOLOv8 TensorRT-Modell ---
model = YOLOv8TensorRT(
    engine_path="./models/trained_yolov8n.engine",
    input_width=480,
    input_height=416,
    conf_thresh=0.4,
    iou_thresh=0.5,
)
CLASS_NAMES = ["bottle", "can"]

# --- SORT Tracker + Track-Zustände ---
tracker = Sort(max_age=5, min_hits=1)   # max_age: wie viele Frames ein Track überleben kann ohne YOLO-Detection
track_states = {}  # track_id -> { 'bbox':(x1,y1,x2,y2), 'smoothed_u', 'smoothed_z', 'z_mm', 'last_yolo_bbox', 'last_update_ts', 'is_yolo' }
# Parameter
MIN_CONF_FOR_TRACK = 0.35
last_yolo_ts = None  # Zeitstempel der letzten echten YOLO-Detection
STALE_TIMEOUT = 7.0  # Sekunden, wie in motion.py

# --- Kamera-Parameter (aus Kalibrierung!) ---
# Kalibrierungsdatei laden
fs = cv2.FileStorage("camCalibParams.yaml", cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise SystemExit("[FATAL] Konnte camCalibParams.yaml nicht öffnen. Pfad prüfen.")

# linke kfe-Matrix und Translationsvektor auslesen
left_kfe = fs.getNode("LeftKFE").mat()
translation = fs.getNode("LeftTranslation").mat()

fs.release()

# Einzelwerte extrahieren
fx = float(left_kfe[0, 0])      # Fokalweite in Pixeln
cx = float(left_kfe[0, 2])
if args.debug_size:
    print("Left camera intrinsics:")
    print(f"fx = {fx}, cx = {cx}")

baseline_mm = float(abs(translation[0, 0]))    # Basislinie in mm
if args.debug_size:
    print(f"Baseline: {baseline_mm:.3f} mm")

# Update-Intervall der Tiefenmessung
DISPARITY_UPDATE_RATE = 6
disparity = None
frame_idx = -1      # Frame-Index (für Disparität + Inferenz scheduling)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# --- StereoSGBM-Kofiguration ---
stereo = cv2.StereoSGBM_create(
    minDisparity = 0,
    numDisparities = 96,        # Muss durch 16 teilbar sein
    blockSize = 7,              # 5–15 empfohlen, ungerade
    P1 = 8 * 1 * 7 ** 2,        # Graustufenbild
    P2 = 32 * 1 * 7 ** 2,       # Graustufenbild
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=8,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# --- Hilfsfunktion: Nur den neuesten Frame verarbeiten ---
def get_latest_message(sock, poller, timeout_ms=5):
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
def get_depth_from_bbox(bbox, disparity_map, fx, baseline_mm,
                        min_valid_pixels=5,
                        iqr_multiplier=1.5):
    # --- Disparitätsbereich für 10–90 cm berechnen ---
    z_min_mm = 100.0   # 10 cm
    z_max_mm = 900.0   # 90 cm
    d_max = (fx * baseline_mm) / z_min_mm   # Disparität für 10 cm
    d_min = (fx * baseline_mm) / z_max_mm   # Disparität für 90 cm
    margin = 0.05  # 5 % Puffer
    d_max *= (1 + margin)
    d_min *= (1 - margin)
#    print(f"Valid disparity range: {d_min:.2f} - {d_max:.2f} px")

    # Bounding Box & ROI
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = disparity_map.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w-1, x2); y2 = min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    # Nur mittleren Bereich der BB verwenden
    y_start_middle = int(y1 + (y2 - y1) * 0.30)
    y_end_middle = int(y1 + (y2 - y1) * 0.70)
    if y_end_middle <= y_start_middle:
        return None
    disparity_roi = disparity_map[y_start_middle:y_end_middle, x1:x2]
    disparity_roi = disparity_roi[::2, ::2]  # nur jedes 2. Pixel
    if disparity_roi.size < min_valid_pixels:
        return None

    # Disparitäten im erlaubten Bereich filtern
    valid_disparities = disparity_roi[
        (disparity_roi >= d_min) & (disparity_roi <= d_max)
    ]
    if valid_disparities.size == 0:
        return None
    
    # IQR-Ausreißer entfernen (robuster)
    q1, q3 = np.percentile(valid_disparities, [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    vals_iqr = valid_disparities[(valid_disparities >= lower) & (valid_disparities <= upper)]
    if vals_iqr.size >= min_valid_pixels:
        chosen_disp = float(np.median(vals_iqr))
    else:
        chosen_disp = float(np.median(valid_disparities))
    
    if not np.isfinite(chosen_disp) or chosen_disp <= 0:
        return None

    # Tiefe in mm berechnen
    z_mm = (fx * baseline_mm) / chosen_disp    # Tiefenformel

    # Debug: Disparität zu BB
    if args.debug_depth:
        print(f"BBox: {bbox} | disp={chosen_disp:.2f} -> z={z_mm:.1f} mm")

    return z_mm

# --- Hilfsfunktion: Bounding Boxes inkl. Abstand (z_mm falls vorhanden) zeichnen ---
def draw_processed_detections(left_img, detections):
    if not detections:  # None oder leere Liste
        return left_img
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

# --- Hilfsfunktion: IoU zwischen zwei Bounding Boxes (a und b) berechnen ---
def iou_box(a, b):
    # a,b = (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(a_area + b_area - inter_area + 1e-6)

# --- Hilfsfunktion: Zentrum einer Bounding Box berechnen ---
def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

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
def update_sort_and_send(processed, disparity, fx, baseline_mm, img_w, cx_val, robot_socket,
                         class_names=CLASS_NAMES, alpha_u=ALPHA_U_TRACK, alpha_z=ALPHA_Z_TRACK):
    """
    Aktualisiert SORT mit aktuellen YOLO-Detections (processed) und sendet
    das Payload für den nächsten Track (nächstes Objekt).
    - z_mm wird NUR bei echten YOLO-Detections gemessen.
    - Zwischenframes (nur SORT-Predictions) behalten den letzten z_mm-Wert.
    """
    global tracker, track_states, last_committed, last_yolo_ts

    # 1) Detections für SORT vorbereiten (x1,y1,x2,y2,conf)
    dets_for_sort = []
    proc_map = []  # um später die Zuordnung zur YOLO-Detection zu machen
    for d in processed:
        conf = d.get('conf', 0.0)
        if conf < MIN_CONF_FOR_TRACK:
            continue
        x1, y1, x2, y2 = map(float, d['bbox'])
        dets_for_sort.append([x1, y1, x2, y2, float(conf)])
        proc_map.append(d)
    dets_np = np.array(dets_for_sort) if len(dets_for_sort) > 0 else np.empty((0,5))

    # 2) SORT aktualisieren (leeres Array liefert nur Prediction)
    tracks = tracker.update(dets_np)

    # 3) Track-Zustände aktualisieren
    now = time.time()
    active_track_ids = set()
    for t in tracks:
        tx1, ty1, tx2, ty2, tid = t
        tid = int(tid)
        tbbox = (int(tx1), int(ty1), int(tx2), int(ty2))
        active_track_ids.add(tid)

        # beste zugeordnete YOLO-Detection finden (IoU >= 0.4)
        best_idx, best_iou = None, 0.0
        for i, d in enumerate(proc_map):
            pb = tuple(d['bbox'])
            iou_v = iou_box(tbbox, pb)
            if iou_v > best_iou:
                best_iou = iou_v
                best_idx = i

        matched_processed = None
        is_yolo = False
        if best_idx is not None and best_iou >= 0.4:
            matched_processed = proc_map[best_idx]
            is_yolo = True
        else:
            # Fallback: Track-Zentrum vs YOLO-Zentrum
            best_dist = None
            best_idx = None
            tcx, tcy = bbox_center(tbbox)
            for i, d in enumerate(proc_map):
                pcx, pcy = bbox_center(tuple(d['bbox']))
                dist = (tcx - pcx)**2 + (tcy - pcy)**2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None and best_dist < ( (img_w * 0.25) ** 2 ):
                matched_processed = proc_map[best_idx]
                is_yolo = True

        # Track-State initialisieren
        ts = track_states.get(tid)
        if ts is None:
            ts = {
                'bbox': tbbox,
                'smoothed_u': None,
                'smoothed_z': None,
                'z_mm': None,
                'last_yolo_bbox': None,
                'last_update_ts': now,
                'is_yolo': False,
                'class_id': None,
                'conf': 0.0
            }

        # Bounding Box updaten
        ts['bbox'] = tbbox
        ts['last_update_ts'] = now
        ts['is_yolo'] = bool(is_yolo)

        if matched_processed is not None:
            # echte YOLO-Detection: u_px glätten und ggf. Disparität messen
            u_px = matched_processed['u_px']
            ts['last_yolo_bbox'] = tuple(matched_processed['bbox'])
            ts['class_id'] = matched_processed.get('class_id')
            ts['conf'] = matched_processed.get('conf', 0.0)

            ts['smoothed_u'] = float(u_px) if ts['smoothed_u'] is None else alpha_u * float(u_px) + (1-alpha_u) * ts['smoothed_u']

            # --- Disparität & Tracking-Update ---
            if ts['last_yolo_bbox'] is not None and disparity is not None:
                # Nur bei echten YOLO-Detections UND wenn neue Disparität verfügbar
                if frame_idx % DISPARITY_UPDATE_RATE == 0:
                    z_new = get_depth_from_bbox(ts['last_yolo_bbox'], disparity, fx, baseline_mm)
                    if z_new is not None:
                        ts['z_mm'] = float(z_new)
                        ts['smoothed_z'] = (
                            float(z_new) if ts.get('smoothed_z') is None 
                            else alpha_z * float(z_new) + (1.0 - alpha_z) * float(ts['smoothed_z'])
                        )
            last_yolo_ts = now    
        else:
            # Nur SORT-Prediction: Tiefe bleibt unverändert, aber u_px vom BoundingBox-Zentrum weiter glätten
            ucx, _ = bbox_center(tbbox)
            ts['smoothed_u'] = (
                float(ucx) if ts['smoothed_u'] is None 
                else alpha_u * float(ucx) + (1.0 - alpha_u) * ts['smoothed_u']
            )
        track_states[tid] = ts

    # alte Tracks entfernen
    for tid in list(track_states.keys()):
        if tid not in active_track_ids:
            age = time.time() - track_states[tid]['last_update_ts']
            if age > 1.0:
                del track_states[tid]

    # 4) Kandidat wählen: 1) geringstes smoothed_z, 2) nearest-to-center, 3) last_committed
    chosen_tid = None
    chosen_ts = None

    # 4.a Tracks mit Depth bevorzugen
    with_depth = [(tid, ts) for tid, ts in track_states.items() if ts.get('smoothed_z') is not None]
    if with_depth:
        chosen_tid, chosen_ts = min(with_depth, key=lambda x: x[1]['smoothed_z'])
    else:
        # 4.b fallback: Track am Bildzentrum (smoothed_u)
        best = None; best_dist = None
        for tid, ts in track_states.items():
            if ts.get('smoothed_u') is None:
                continue
            dist = abs(ts['smoothed_u'] - cx_val)
            if best_dist is None or dist < best_dist:
                best_dist = dist; best = (tid, ts)
        if best is not None:
            chosen_tid, chosen_ts = best
        else:
            # 4.c letzter Notfall: last_committed verwenden (falls vorhanden)
            if last_committed is not None:
                chosen_tid = -1
                # last_committed normalisieren in chosen_ts-Form
                chosen_ts = {
                    'smoothed_u': last_committed.get('u_px'),
                    'z_mm': last_committed.get('z_mm'),
                    'smoothed_z': last_committed.get('z_mm'),
                    'class_id': last_committed.get('class_id'),
                    'conf': last_committed.get('conf', 0.0),
                    'is_yolo': False
                }

    # Wenn kein Kandidat vorhanden -> nichts senden
    if chosen_ts is None:
        return None
    
    # 5) Globale EWMA-Glättung (committed state updaten)
    u_raw = chosen_ts.get('smoothed_u')
    # z_raw priorisiert smoothed_z, sonst z_mm
    z_raw = chosen_ts.get('smoothed_z') if chosen_ts.get('smoothed_z') is not None else chosen_ts.get('z_mm')
    cls = chosen_ts.get('class_id')
    conf = chosen_ts.get('conf', 0.0)

    # Stale-Check
    if last_yolo_ts is not None and (time.time() - last_yolo_ts) > STALE_TIMEOUT:
    # keine echte YOLO-Detection für > STALE_TIMEOUT
        return None  # nichts senden → Roboter stoppt

    # defensive: u_raw muss vorhanden sein
    if u_raw is None:
        return None
    
    if last_committed is None:
        u_s = u_raw
        z_s = z_raw
    else:
        # u_px immer glätten
        u_s = alpha_u * u_raw + (1.0 - alpha_u) * last_committed['u_px']

        # z_mm nur glätten, wenn valide
        if z_raw is None:
            z_s = last_committed['z_mm']
        elif last_committed['z_mm'] is None:
            z_s = z_raw
        elif abs(z_raw - last_committed['z_mm']) > RESET_Z_THRESHOLD_MM:
            z_s = z_raw  # Sprung → sofort übernehmen
        else:
            z_s = alpha_z * z_raw + (1.0 - alpha_z) * last_committed['z_mm']

    # --- Update globalen committed State ---
    last_committed = {
        'u_px': float(u_s),
        'z_mm': float(z_s) if z_s is not None else None,
        'class_id': cls,
        'conf': float(conf)
    }

    # --- Payload bauen ---
    # Ableitungen für Steuerung !
    u_offset_px = float(u_s - cx_val)

    # Winkel (radiant und degree)
    angle_rad = math.atan2(u_offset_px, fx)
    angle_deg = math.degrees(angle_rad)

    # Normieren auf halbe Bildbreite
    u_norm = u_offset_px / (img_w / 2.0)

    cls_id = chosen_ts.get('class_id')
    cls_name = class_names[cls_id] if isinstance(cls_id, int) and 0 <= cls_id < len(class_names) else ''

    if chosen_tid == -1:
        source = 'sort'
    else:
        # chosen_ts['is_yolo'] == True bedeutet: dieser Track wurde in DIESEM Frame mit einer YOLO-Detection matched
        source = 'yolo' if chosen_ts.get('is_yolo') else 'sort'

    payload = {
        'detection': {
            'track_id': int(chosen_tid),
            'class_id': chosen_ts.get('class_id'),
            'class_name': cls_name,
            'conf': float(chosen_ts.get('conf', 0.0)),
            'u_px': float(u_s),
            'u_offset_px': u_offset_px,
            'angle_rad': angle_rad,
            'angle_deg': angle_deg,
            'u_norm': u_norm,
            'source': source
        }
    }
    if z_s is not None:
        payload['detection']['z_mm'] = float(z_s)

    # 6) Senden (non-blocking) + Rückgabe
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

    # --- YOLO-Inferenz mit YOLOv8 TensorRT ---
    detections_for_processing = []  # Liste zum Zeichnen
    payload = None             # fürs Robot-Command

    try:
        raw_dets = model.infer(left_img)  # tatsächliche Inferenz
    except Exception as e:
        raw_dets = []
        print(f"[WARN] model.infer() failed: {e}")

    if raw_dets:
        # YOLO-Detections aufbereiten
        processed = build_processed_detections(raw_dets, disparity, fx, baseline_mm, CLASS_NAMES)
        if processed:
            # 1) SORT aktualisieren / YOLO-Daten übergeben
            payload = update_sort_and_send(processed, disparity, fx, baseline_mm, img_w=left_width,
                                                cx_val=cx, robot_socket=robot_socket, class_names=CLASS_NAMES)
            # 2) Für Anzeige: YOLO-Detections
            detections_for_processing = processed
        else:
            # Model lieferte nix
            detections_for_processing = []
            payload = None

    else:
        # Keine YOLO-Detections: SORT-Prediction aufrufen
        payload = update_sort_and_send([], disparity, fx, baseline_mm, img_w=left_width,
                                       cx_val=cx, robot_socket=robot_socket, class_names=CLASS_NAMES)
        # Für Zeichnung
        detections_for_processing = []

    # --- Ergebniszeichnung ---
    # 1) YOLO-Detections
    left_img = draw_processed_detections(left_img, detections_for_processing)

    # 2) Predicted SORT-Tracks: orange zeichnen (pro Track ohne YOLO-Match in diesem Frame)
    for tid, ts in track_states.items():
        if not ts.get('is_yolo'):  # nur reine SORT-Predictions
            x1, y1, x2, y2 = ts['bbox']
            cv2.rectangle(left_img, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange BGR
            cv2.putText(left_img, f"trk{tid}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)

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