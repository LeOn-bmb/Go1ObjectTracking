"""
ZeroMQ PULL Receiver + PID-basierte Yaw-Nachführung für Unitree (High-Level).
- Empfängt JSON per ZeroMQ (vom Xavier NX)
- Führt PID primär auf angle_rad aus (Fallback: u_norm) -> cmd.yawSpeed
- Sendet HighCmd per robot_interface (UDP)
- Stale-Timeout: nach keiner Detection > STALE_TIMEOUT wird Drehung gestoppt und Integrator zurückgesetzt.
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

cmd.mode = 2           # Geschwindigkeitsmode
cmd.velocity[0] = 0.0
cmd.velocity[1] = 0.0
cmd.yawSpeed = 0.0

# ---------- PID-Zustand ----------
Kp = 1.0
Ki = 0.3
Kd = 0.02

MAX_YAW = 1.0               # Max yawSpeed (rad/s)
DEADBAND_ANGLE = 0.035      # rad (~2°)
DEADBAND_UNORM = 0.03       # 3 % halbe Bildbreite

integrator = 0.0
prev_error = 0.0
prev_time = time.time()
integrator_limit = MAX_YAW * 2.0  # Anti-Windup
deriv_filtered = 0.0
deriv_tau = 0.02

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# --- PID-Regelung zum gieren ---
def compute_pid(signal, dt, source):
    """
    PID-Regler. Fehler = -signal (Ziel: 0).
    signal = angle_rad (oder u_norm als Fallback).
    """
    global integrator, prev_error, deriv_filtered

    error = -signal

    # Deadband je nach Quelle
    deadband = DEADBAND_ANGLE if source == "angle_rad" else DEADBAND_UNORM
    if abs(signal) < deadband:
        integrator *= 0.0
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

# --- PID-relevante Werte extrahieren ---
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

try:
    while True:
        # Nachricht erhalten
        try:
            msg = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            continue

        # JSON decodieren
        try:
            data = json.loads(msg.decode("utf-8"))
        except Exception as e:
            print(f"[WARN] JSON-Decodierung fehlgeschlagen: {e}")
            continue

        # Ausgabe
        if data is not None:
            dets = [data] if isinstance(data, dict) else data
            for i, d in enumerate(dets):
                track_id    = d.get("track_id")
                u           = d.get("smoothed_u")
                u_norm      = d.get("smoothed_u_norm")
                angle_rad   = d.get("smoothed_angle_rad")
                angle_deg   = d.get("smoothed_angle_deg")
                z_mm        = d.get("smoothed_z_mm")

                if args.verbose:
                    print(f"[INFO] Det {i}: ID={track_id} || Angle_rad={angle_rad:.2f} || U_norm={u_norm:.2f} || Z_mm={z_mm:.0f}mm")
        else:
            print("[INFO] Keine Detektionen im Payload.")
        '''
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

        # HighCmd setzen & senden
        cmd.mode = 2
        cmd.velocity[0] = 0.0
        cmd.velocity[1] = 0.0
        cmd.yawSpeed = float(yaw_cmd)

        try:
            udp.SetSend(cmd)
            udp.Send()
            if args.verbose:
                print(f"[DEBUG] {source}={angle:.3f} -> yaw_cmd={yaw_cmd:.3f} rad/s")
            else:
                print(f"[INFO] yaw_cmd={yaw_cmd:.3f}")
        except Exception as e:
            print(f"[WARN] udp.Send gescheitert: {e}")
        '''
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