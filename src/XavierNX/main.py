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
from collections import deque
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

# ZeroMQ (PULL vom Nano)
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.RCVHWM, 4)
socket.setsockopt(zmq.LINGER, 0)
socket.bind("tcp://*:5555")  # auf Verbindung warten

# Poller erzeugen
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

print("Empfänger bereit...")

# ZeroMQ (PUSH zum Raspberry)
robot_socket = context.socket(zmq.PUSH)
robot_socket.setsockopt(zmq.SNDHWM, 4)   # max queued messages before dropping
robot_socket.setsockopt(zmq.LINGER, 0)   # kein blockierendes Schließen
robot_socket.connect(f"tcp://192.168.123.161:5560")

# Glättungsfaktoren / EWMA Faktor 0..1 -> höher = reaktiver, niedriger = ruhiger
ALPHA_U_TRACK = 0.6             # Glättung für u_px (horizontale Position)
ALPHA_Z_TRACK = 0.4             # Glättung für z_mm (Distanz)
last_committed = None           # Globaler Glättungsstatus anhand letzter Werte
RESET_Z_THRESHOLD_MM = 250.0    # wenn neuer z mehr ist, dann Reset

# --- CLI Parser-Argumente definieren ---
parser = argparse.ArgumentParser(description="Empfängt Bilder, führt Objekterkennung durch und führt optional Debugfunktionen aus.")
parser.add_argument('--debug-view', choices=['left','right','both'],
                    help='YOLO-Ergebnisse, Track-Ids und Distanz im linken, rechten oder beiden Bildern anzeigen')
parser.add_argument('--debug-payload', choices=['left','right','both'],
                    help='Den gesendeten Track (Payload) im linken Bild markieren')
parser.add_argument('--debug-size', action='store_true',
                    help='Frame-Größen, Typen und Kalibrierungsparameter debuggen')
parser.add_argument('--debug-fps', action='store_true',
                    help='FPS-Messung ausgeben')
parser.add_argument('--debug-img', metavar='imgname', 
                    help='Dateiname für das zu speichernde Bild, dann wählbar l(eft), r(ight), c(ombined)')
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

# --- Kamera-Parameter (aus Kalibrierungsdatei) laden ---
fs = cv2.FileStorage("camCalibParams.yaml", cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise SystemExit("[FATAL] Konnte camCalibParams.yaml nicht öffnen. Pfad prüfen.")

# linke kfe-Matrix und Translationsvektor auslesen
left_kfe = fs.getNode("LeftKFE").mat()
translation = fs.getNode("LeftTranslation").mat()
fs.release()

# Einzelwerte extrahieren
fx = float(left_kfe[0, 0])                    # Fokalweite in Pixeln
cx = float(left_kfe[0, 2])
baseline_mm = float(abs(translation[0, 0]))   # Basislinie in mm

# --- Tracking-Variablen ---
track_states = {}  # Track-Status (id -> state dict)
next_track_id = 0  # Nächste verfügbare Track-ID

# Farbpalette für die Track-ID Disparitäten (RGB tuples)
track_palette = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255),
                         (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128),
                         (64,64,64), (192,192,192), (255,165,0), (75,0,130), (255,20,147)]

# --- Timeout für das Senden an den Raspberry Pi ---
STALE_TIMEOUT = 7.0  # Sekunden ohne echte YOLO-Detection → nichts senden

# ----------------- Receive-Phase -----------------

# --- Nur den neuesten Frame verarbeiten ---
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

# ----------------- Detection-Phase -----------------

# --- Bounding Boxes aufbauen  ---
def build_processed_detections(raw_dets, class_names):
    processed = []

    for det in raw_dets:
        x1, y1, x2, y2, conf, cls_id = det
        # Bounding Box Mitte
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        processed.append({
            'x1': int(x1), 'y1': int(y1),'x2': int(x2), 'y2': int(y2),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'conf': float(conf),
            'class_id': int(cls_id),
            'class_name': class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else f"class_{int(cls_id)}",
            'u_px': float(u),
            'v_px': float(v)
        })
    return processed

# --- Bounding Boxes paaren (Stereo-Matching links/rechts) ---
def match_bboxes(processed_left, processed_right, fx, baseline_mm,
                 max_v_diff=3.5, height_ratio_thresh=(0.75, 1.35), z_range_mm=(100, 900)):
    matches = []
    used = set()

    # Erwartete Disparitätsspanne aus den bekannten Parametern
    if fx is not None and baseline_mm is not None:
        d_max = (fx * baseline_mm) / z_range_mm[0]  # nahes Objekt
        d_min = (fx * baseline_mm) / z_range_mm[1]  # fernes Objekt
        margin = 0.1  # 10 % Puffer
        d_max *= (1 + margin)
        d_min *= (1 - margin)
    else:
        d_min, d_max = None, None

    # Für jede linke Box die beste rechte Box suchen
    for i, det_left in enumerate(processed_left):
        best = None
        best_score = 1e9
        best_j = -1

        for j, det_right in enumerate(processed_right):
            if j in used:
                continue
            # gleiche Klasse?
            if det_left['class_id'] != det_right['class_id']:
                continue
            
            vL, vR = det_left['v_px'], det_right['v_px']
            hL = det_left['y2'] - det_left['y1']
            hR = det_right['y2'] - det_right['y1']
            v_diff = abs(vL - vR)
            h_ratio = hL / max(hR, 1e-6)
#            print(f"[DEBUG] Checking match y-Abweichung = {v_diff:.2f} px | Ratio: {h_ratio:.2f}")

            # Filter: maximale v-Abweichung und Höhenverhältnis
            if v_diff > max_v_diff:
                continue
            if not (height_ratio_thresh[0] <= h_ratio <= height_ratio_thresh[1]):
                continue
            # Disparität und Tiefenbereich prüfen
            uL, uR = det_left['u_px'], det_right['u_px']
            disp = abs(uL - uR)
            if d_min is not None and (disp < d_min or disp > d_max):
                continue
            
            # Beste Übereinstimmung anhand der Disparität zum erwarteten Wert
            score = disp + 0.5 * v_diff + 0.5 * abs(1 - h_ratio) * hL
            if score < best_score:
                best_score = score
                best = (det_left, det_right, disp)
                best_j = j

        if best is not None:
            used.add(best_j)
            # rechte Box im linken Detection speichern
            det_left['right_bbox'] = best[1]['bbox']
            matches.append(best)
    return matches

# ----------------- Tracking-Phase -----------------

# --- Track-States aktualisieren (Zuordnung der Detections über Zeit) ---
def update_tracks(matches, track_states, next_track_id, max_u_diff=20, max_v_diff=10, max_disp_history=10):
    now = time.time()
    used_tracks = set()
    
    for det_left, det_right, disp in matches:
        uL, vL = det_left['u_px'], det_left['v_px']
        best_tid, best_dist = None, None
        
        # Existierende Tracks prüfen
        for tid, ts in track_states.items():
            if tid in used_tracks:
                continue
            u_prev, v_prev = ts['left_center']
            du, dv = abs(uL - u_prev), abs(vL - v_prev)
            if du <= max_u_diff and dv <= max_v_diff:
                dist = du + dv
                if best_dist is None or dist < best_dist:
                    best_dist, best_tid = dist, tid
        
        if best_tid is not None:
            # Existierender Track
            ts = track_states[best_tid]
            ts['left_center'] = (uL, vL)
            ts['disparities'].append(disp)
            ts['last_update'] = now
            det_left['track_id'] = best_tid
            det_right['track_id'] = best_tid
            used_tracks.add(best_tid)
        else:
            # Neuer Track
            track_states[next_track_id] = {
                'left_center': (uL, vL),
                'disparities': deque([disp], maxlen=max_disp_history),
                'last_update': now
            }
            det_left['track_id'] = next_track_id
            det_right['track_id'] = next_track_id
            used_tracks.add(next_track_id)
            next_track_id += 1
    
    # Alte Tracks entfernen
    to_delete = [tid for tid, ts in track_states.items() if now - ts['last_update'] > 1.0]
    for tid in to_delete:
        del track_states[tid]

    # --- Debug-Ausgabe aller aktiven Tracks ---
    print("=== TRACK [DEBUG] ===")
    for tid, ts in track_states.items():
        print(f"Track {tid}: disparities={list(ts['disparities'])}")
  
    return track_states, next_track_id

# --- Distanz aus Disparität berechnen ---
def depth_from_disparity(track_states, fx, baseline_mm):
    for tid, ts in track_states.items():
        if 'disparities' in ts and ts['disparities']:
            # Median für stabile Tiefe
            disp_median = np.median(list(ts['disparities']))
            z_mm = (fx * baseline_mm) / disp_median
            ts['z_mm'] = z_mm
        else:
            ts['z_mm'] = None
    return track_states
'''
# ----------------- Target-Auswahl-Phase -----------------

# Track mit der geringsten Distanz auswählen und rig-zentieren
def select_target(track_states):

# Track mit der geringsten Distanz auswählen

# evtl. Median der Disparitäten für stabile Tiefe

# Rig-zentrierter Offset in Pixeln
cx_bar = 0.5 * (cxL + cxR)
u_rig = uL - 0.5 * disp
u_offset_px = u_rig - cx_bar

# Winkel berechnen (Radiant und Grad)
angle_rad = math.atan(u_offset_px / fx)
angle_deg = math.degrees(angle_rad)

# ----------------- Steuerungs-/Sende-Phase -----------------

# Zielparameter mit EWMA glätten
def smooth_target(u_px, z_mm, last_committed):

# Payload dict mit geglätteten Werten erstellen
def build_payload(det, smoothed_u, smoothed_z):

# --- Payload an den Raspberry Pi senden ---
def send_payload(det, track_states, robot_socket):
    global last_committed

    if det is None or 'track_id' not in det:
        return None

    tid = det['track_id']
    ts = track_states.get(tid)
    if ts is None or not ts['disparities']:
        return None

    # Durchschnittliche Disparität
    avg_disp = sum(ts['disparities']) / len(ts['disparities'])
    z_mm = depth_from_disparity(avg_disp, fx, baseline_mm)
    if z_mm is None:
        return None

    u_px = det['u_px']

    # Glättung
    if last_committed is None:
        smoothed_u = u_px
        smoothed_z = z_mm
    else:
        u_prev, z_prev = last_committed
        if abs(z_mm - z_prev) > RESET_Z_THRESHOLD_MM:
            smoothed_u = u_px
            smoothed_z = z_mm
        else:
            smoothed_u = ALPHA_U_TRACK * u_px + (1 - ALPHA_U_TRACK) * u_prev
            smoothed_z = ALPHA_Z_TRACK * z_mm + (1 - ALPHA_Z_TRACK) * z_prev

    last_committed = (smoothed_u, smoothed_z)

    payload = {
        'track_id': tid,
        'class_id': det['class_id'],
        'class_name': det['class_name'],
        'u_px': smoothed_u,
        'z_mm': smoothed_z,
        'timestamp': time.time()
    }

    try:
        robot_socket.send_json(payload, flags=zmq.NOBLOCK)
        print(f"[SENT] {json.dumps(payload)}")
    except zmq.Again:
        print("[WARN] Robot socket busy, message dropped.")
    return payload
'''
# ----------------- Debugfunktionen -----------------

# --- Bounding Boxes inkl. Abstand (z_mm falls vorhanden) zeichnen ---
def draw_processed_detections(img, detections, side='left', color=(0,255,0), track_states=None):
    # --- Beide Bilder nebeneinander zeichnen ---
    if side == "both":
        left_img, right_img = img  # Tuple entpacken
        left_dets = detections.get("left", [])
        right_dets = detections.get("right", [])

        # Rekursive Aufrufe für beide Seiten
        left_img = draw_processed_detections(left_img, detections["left"], side="left", color=(0,255,0), track_states=track_states)
        right_img = draw_processed_detections(right_img, detections["right"], side="right", color=(255,200,0), track_states=track_states)
        # Beide Bilder nebeneinander kombinieren
        combined = np.hstack((left_img, right_img))
        
        # Disparity-Linien zwischen korrespondierenden Detections zeichnen
        for det_left in left_dets:
            if 'track_id' in det_left and 'right_bbox' in det_left:
                tid = det_left['track_id']
                # Index der Farbe im Palette-Rad
                col = track_palette[tid % len(track_palette)]
                # Mittelpunkt links/rechts
                uL, vL = int(det_left['u_px']), int(det_left['v_px'])
                uR = int(det_left['right_bbox'][0] + (det_left['right_bbox'][2]-det_left['right_bbox'][0])/2)
                vR = vL # rektifiziert → gleiche v-Koordinate
                # Linie zeichnen (linkes Bild → rechtes Bild)
                cv2.line(combined, (uL, vL), (uR + left_img.shape[1], vR), col, 2)
        return combined

    # --- Einzelbild Fall (left oder right) ---
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det.get('class_name', "obj")
        conf = det.get('conf', 0.0)
        u, v = int(det['u_px']), int(det['v_px'])

        # Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Mittelpunkt
        cv2.circle(img, (u, v), 4, (0, 0, 255), -1)
        # Label
        label = f"{cls_name} {conf:.2f}"
        # Track-ID anzeigen
        if 'track_id' in det:
            tid = det['track_id']
            label += f" | ID:{tid}"
            # Distanz aus Track-States
            if track_states is not None:
                ts = track_states.get(tid)
                if ts is not None and 'z_mm' in ts and ts['z_mm'] is not None:
                    label += f" | z={ts['z_mm']:.0f}mm"

        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

'''
# gesendeten Track zeichnen
def draw_payload_on_left(img, payload, track_states, color_yolo=(0,255,0), color_sort=(0,165,255)):
    if img is None or payload is None:
        return img
    det = payload.get('detection', {})
    tid = det.get('track_id', None)
    if tid is None:
        return img
    ts = track_states.get(tid)
    if ts is None or 'bbox' not in ts:
        return img
    color = color_yolo if det.get('source') == 'yolo' else color_sort
    x1,y1,x2,y2 = ts['bbox']
    label = f"send trk{tid} ({det.get('source','')})"
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img
'''
# Frame-Größe & Typen debuggen
def print_debug_frame_info(left_width, left_height, right_width, right_height, left_type, right_type):
    print(f"Left Frame: {left_width}x{left_height} | Right Frame: {right_width}x{right_height}")
    print(f"Empfangener left_type: {left_type}, expected: {cv2.CV_8UC3}")
    print(f"Empfangener right_type: {right_type}, expected: {cv2.CV_8UC3}") 
    print(f"Kamerakalibrierung: fx={fx}, cx={cx}, baseline={baseline_mm:.3f} mm")

# FPS-Anzeige und Messung
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

# PNG-Aufnahme
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

# ----------------- Hauptloop -----------------

try:
    while True:
        # Nachricht erhalten
        message = get_latest_message(socket, poller, timeout_ms=15)
        if message is None:
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
        else:
            print(f"[WARN] Unbekannter OpenCV-Typ: left_type={left_type}, right_type={right_type}")
            continue

        # --- YOLO-Inferenz auf beiden Bildern mit YOLOv8 TensorRT ---
        detections_for_processing = []  # Liste zum Zeichnen
        payload = None                  # fürs Robot-Command

        try:
            raw_dets_left = model.infer(left_img)   # tatsächliche Inferenz auf linkem Bild
        except Exception as e:
            raw_dets_left = []
            print(f"[WARN] model.infer(left) failed: {e}")
        try:
            raw_dets_right = model.infer(right_img) # tatsächliche Inferenz auf rechtem Bild
        except Exception as e:
            raw_dets_right = []
            print(f"[WARN] model.infer(right) failed: {e}")

        # Aufbereitung der Detections
        processed_left = build_processed_detections(raw_dets_left, CLASS_NAMES)
        processed_right = build_processed_detections(raw_dets_right, CLASS_NAMES)

        # --- Stereo-Matching der Bounding Boxes ---
        matches = match_bboxes(processed_left, processed_right, fx=fx, baseline_mm=baseline_mm)

        # Track-Update
        track_states, next_track_id = update_tracks(matches, track_states, next_track_id)

        # --- Tiefenberechnung für alle aktiven Tracks ---
        track_states = depth_from_disparity(track_states, fx, baseline_mm)

#        print(f"[DEBUG] Tracks: {len(track_states)} | Matches: {len(matches)} | Detections L:{len(processed_left)} R:{len(processed_right)}")


        #  --- Senden des Tracks mit der geringsten Distanz ---



        # --- Ergebniszeichnung ---
        # 1) YOLO-Detections
        left_img = draw_processed_detections(left_img, detections_for_processing)
        right_img = draw_processed_detections(right_img, detections_for_processing)

# ----------------- CLI-Argumente für Debug-Ausgaben nutzen -----------------

        # YOLO-Detections
        if args.debug_view:
            if args.debug_view == 'left':
                left_img = draw_processed_detections(
                    left_img, processed_left, side='left', color=(0,255,0), track_states=track_states)
                cv2.imshow("YOLO Detections Left", left_img)

            elif args.debug_view == 'right':
                right_img = draw_processed_detections(
                    right_img, processed_right, side='right', color=(255,200,0), track_states=track_states)
                cv2.imshow("YOLO Detections Right", right_img)

            elif args.debug_view == 'both':
                combined = draw_processed_detections(
                    (left_img, right_img),
                    {"left": processed_left, "right": processed_right},
                    side='both', track_states=track_states)
                cv2.imshow("YOLO Detections Both", combined)
        '''
        # Payload zeichnen
        if args.debug_payload and payload is not None:
            left_img_vis = draw_payload_on_left(left_img_vis, payload, track_states)
            cv2.imshow("Payload", left_img_vis)
        '''
        # Frame-Size Debug (einmalig)
        if args.debug_size and size_printed == 0:
            print_debug_frame_info(left_width, left_height, right_width, right_height, left_type, right_type)
            size_printed = 1

        # FPS-Zähler
        if args.debug_fps:
            frame_count += 1
            frame_count, last_fps_time, seconds_elapsed, fps_outputs = update_fps_counter(
                frame_count, last_fps_time, seconds_elapsed, fps_outputs, fps_list
            )

        # Tastendruck für Debug-Aufnahmen
        key = cv2.waitKey(1) & 0xFF
        if args.debug_img and key in [ord('l'), ord('r'), ord('c')]:
            if key == ord('l'):
                capture_frame(left_img, args.debug_img)
            elif key == ord('r'):
                capture_frame(right_img, args.debug_img)
            elif key == ord('c'):
                combined = np.hstack((left_img, right_img))
                capture_frame(combined, args.debug_img)
        
        # --- Escape-Taste zum Beenden ---
        if key == 27:  # ESC
            break
except KeyboardInterrupt:
    print("\nBeendet durch Benutzer (KeyboardInterrupt).")
except Exception as e:
    print(f"[FATAL] Unerwarteter Fehler: {e}")
finally:
    # --- Aufräumen ---
    socket.close()
    robot_socket.close()
    context.term()
    cv2.destroyAllWindows()