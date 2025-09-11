"""
ZeroMQ PULL Receiver + PID-basierte Yaw-Nachführung für Unitree (High-Level).
- Empfängt JSON per ZeroMQ (vom Xavier NX)
- Führt PID primär auf angle_rad aus (Fallback: u_norm) -> cmd.yawSpeed
- Führt eine einfache Hysterese-basierte Distanzregelung auf den Distanzwert z_mm aus-> cmd.velocity
- Sendet HighCmd (High-Level Befehle) per robot_interface (UDP)
"""

import zmq
import json
import time
import sys
import argparse

# Pfad zum Roboter-Interface
sys.path.append('./lib/python/arm64')   # ggf. anpassen
try:
    import robot_interface as sdk
except Exception as e:
    print("[FATAL] Konnte robot_interface nicht importieren. Prüfe sys.path und SDK-Install.")
    raise

# --- ZeroMQ (PULL vom Xavier NX) ---
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.RCVHWM, 15)
socket.setsockopt(zmq.LINGER, 0)
socket.bind("tcp://*:5560")  # auf Verbindung warten

print("Empfänger bereit...")

# --- Parser-Argumente ---
parser = argparse.ArgumentParser(description="Empfängt JSON vom Xavier (PUSH -> PULL) und steuert Unitree per UDP.")
parser.add_argument("--verbose", action="store_true", help="Mehr Ausgabe.")
args = parser.parse_args()

# --- Unitree UDP / SDK init ---
HIGHLEVEL = 0xee
udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
cmd = sdk.HighCmd()
state = sdk.HighState()
udp.InitCmdData(cmd)

cmd.gaitType = 1       # Walking
cmd.mode = 2           # Geschwindigkeitsmode
# Init Motion
cmd.velocity = [0, 0]
cmd.yawSpeed = 0.0

# ---------- PID-Zustand ----------
Kp = 0.6
Ki = 0.2
Kd = 0.03
MAX_YAW = 0.6               # Max yawSpeed (rad/s)

#  Hysterese-Schwellen 
ANGLE_INNER = 0.07    # rad (~4°) -> "gut genug zentriert"
ANGLE_OUTER = 0.1     # rad (~5.7°) -> erst dann wieder aktiv werden
UNORM_INNER = 0.03    # norm (~3%) 
UNORM_OUTER = 0.04    # norm (~4%)
# Hysterese-Zustände merken
inside_dead_angle = False
inside_dead_unorm = False

integrator = 0.0
prev_error = 0.0
prev_time = time.time()
integrator_limit = MAX_YAW * 2.0  # Anti-Windup
deriv_filtered = 0.0
deriv_tau = 0.02

# ---------- Distanz-Parameter ----------
DIST_TARGET = 600.0     # mm
DIST_TOL_INNER = 40.0   # mm
DIST_TOL_OUTER = 70.0   # mm
MAX_FWD_SPEED = 0.4     # m/s (vorwärts/rückwärts)
# Hysterese-Zustände für Distanz
inside_dead_dist = False

# Glättung für von Fallback in z_mm vorhanden
last_forward_cmd = 0.0
ALPHA_FWD = 0.3  # Glättung (0..1)

# ----------------- Gieren -----------------

# PID-relevante Werte extrahieren
def extract_angle_or_unorm(data):
    if data is None:
        return None, None

    # Primär: geglätteter Winkel
    angle_rad = data.get("smoothed_angle_rad")
    if angle_rad is not None:
        return float(angle_rad), "angle_rad"

    # Fallback: geglättete Normierung der u-Position
    u_norm = data.get("smoothed_u_norm")
    if u_norm is not None:
        return float(u_norm), "u_norm"
    return None, None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# PID-Regelung
def compute_pid(signal, dt, source):
    """
    PID-Regler mit Hysterese.
    signal = angle_rad oder u_norm (als Fallback).
    Fehler = -signal (Ziel: 0).
    """
    global integrator, prev_error, deriv_filtered, inside_dead_angle, inside_dead_unorm

    error = -signal

    # Hysterese je nach Quelle
    if source == "angle_rad":
        if abs(signal) < ANGLE_INNER:
            inside_dead_angle = True
        elif abs(signal) > ANGLE_OUTER:
            inside_dead_angle = False

        if inside_dead_angle:
            # Ruhezone: nichts regeln, Integrator NICHT löschen (freeze)
            prev_error = 0.0
            deriv_filtered = 0.0
            return 0.0

    else:  # Fallback u_norm
        if abs(signal) < UNORM_INNER:
            inside_dead_unorm = True
        elif abs(signal) > UNORM_OUTER:
            inside_dead_unorm = False

        if inside_dead_unorm:
            integrator = 0.0
            prev_error = 0.0
            deriv_filtered = 0.0
            return 0.0

    # P
    P = Kp * error

    # I (Trapezregel)
    integrator += 0.5 * (error + prev_error) * Ki * dt
    integrator = clamp(integrator, -integrator_limit, integrator_limit)

    # D (gefiltert)
    raw_deriv = (error - prev_error) / dt if dt > 0 else 0.0
    alpha = dt / (deriv_tau + dt) if (deriv_tau + dt) > 0 else 1.0
    deriv_filtered = (1 - alpha) * deriv_filtered + alpha * raw_deriv
    D = Kd * deriv_filtered

    prev_error = error

    yaw = P + integrator + D
    yaw = clamp(yaw, -MAX_YAW, MAX_YAW)
    return yaw

# ----------------- Distanzregelung -----------------

def smooth_speed(new_speed):
    global last_forward_cmd
    last_forward_cmd = (1 - ALPHA_FWD) * last_forward_cmd + ALPHA_FWD * new_speed
    return last_forward_cmd


def compute_forward_speed(data):
    """
    Einfache Hysterese-basierte Distanzregelung.
    """
    global inside_dead_dist
    z_mm = data.get("smoothed_z_mm")
    angle_rad = data.get("smoothed_angle_rad")

    if z_mm is None:
            # Kein Stereo-Match verfügbar. Fallback: nur dann langsam vorwärts gehen, wenn wir bereits gut genug ausgerichtet sind.
        if angle_rad is not None and inside_dead_angle or inside_dead_unorm:
            return 0.05   # langsam vorwärts, um Stereo-Match zu provozieren
        else:
            return 0.0   # warten bis ausgerichtet

    # --- Sicherheitsfenster ---
    if z_mm < 400:
        # zu nah → langsam zurück
        return clamp(-0.05, -MAX_FWD_SPEED, MAX_FWD_SPEED)

    if z_mm > 800:
        # zu weit weg → langsam vor
        return clamp(0.05, -MAX_FWD_SPEED, MAX_FWD_SPEED)
    
    # --- Einregelung auf Ziel ---
    error = float(z_mm) - DIST_TARGET  # >0: zu weit weg, <0: zu nah

    # Hysterese
    if abs(error) <= DIST_TOL_INNER:
        inside_dead_dist = True
    elif abs(error) > DIST_TOL_OUTER:
        inside_dead_dist = False

    if inside_dead_dist:
        return 0.0  # Ruhezone
    else:
        # proportional steuern, Geschwindigkeit begrenzen
        if error < 0:
            k = 0.001  # sanfter rückwärts
        else:
            k = 0.002  # schneller vorwärts

        speed = k * error

        # --- Zusatz: weich abbremsen, wenn < 500 mm ---
        if z_mm < 500:
            speed = min(speed, 0.1)  # maximal 0.1 m/s vorwärts

        return clamp(speed, -MAX_FWD_SPEED, MAX_FWD_SPEED)

# ----------------- Hauptloop -----------------

try:
    while True:
        # Nachricht erhalten
        try:
            msg = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.005)
            continue

        # JSON decodieren
        try:
            data = json.loads(msg.decode("utf-8"))
        except Exception as e:
            print(f"[WARN] JSON-Decodierung fehlgeschlagen: {e}")
            continue

        # Ausgabe
        if data is not None and args.verbose:
            dets = [data] if isinstance(data, dict) else data
            for i, d in enumerate(dets):
                track_id    = d.get("track_id")
                u           = d.get("smoothed_u")
                u_norm      = d.get("smoothed_u_norm")
                angle_rad   = d.get("smoothed_angle_rad")
                angle_deg   = d.get("smoothed_angle_deg")
                z_mm        = d.get("smoothed_z_mm")

                print(f"[INFO] Det {i}: ID={track_id} || Angle_rad={angle_rad:.2f} || U_norm={u_norm:.2f} || Z_mm={z_mm:.0f}mm")
        else:
            print("[INFO] Keine Detektionen im Payload.")

        # PID auf Winkel
        angle, source = extract_angle_or_unorm(data)
        if angle is None:
            print("[WARN] Kein angle_rad oder u_norm im Payload")
            continue

        now = time.time()
        dt = max(now - prev_time, 1e-3)
        yaw_cmd = compute_pid(angle, dt, source)
        prev_time = now
        last_detection_time = now

        # Forward Geschwindigkeit auf Distanz
        forward_cmd = smooth_speed(compute_forward_speed(data))

        # HighCmd setzen & senden
        cmd.mode = 2
        cmd.velocity = [forward_cmd, 0]     # [vorwärts, seitwärts]
        cmd.yawSpeed = float(yaw_cmd)       # Gieren

        try:
            udp.SetSend(cmd)
            udp.Send()
            if args.verbose:
                print(f"[DEBUG] forward_cmd={forward_cmd:.2f} m/s, yaw_cmd={yaw_cmd:.3f} rad/s")
        except Exception as e:
            print(f"[WARN] udp.Send gescheitert: {e}")
        
except KeyboardInterrupt:
    print("\n[INFO] Durch Benutzer beendet. Sende Stopp-Befehl an Roboter.")
    try:
        cmd.velocity[0] = cmd.velocity[1] = cmd.yawSpeed = 0.0
        udp.SetSend(cmd)
        udp.Send()
    except Exception:
        pass
finally:
    try:
        socket.close()
        context.term()
    except Exception:
        pass
    print("[INFO] Beende.")