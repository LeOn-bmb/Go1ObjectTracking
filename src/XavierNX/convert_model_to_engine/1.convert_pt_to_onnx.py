# Shell-cmd: yolo export model=best.pt format=onnx device=0 dynamic=False simplify=True imgsz=416,480 opset=12

import os
from ultralytics import YOLO

# Verzeichnis mit .pt-Dateien
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
print(f"📁 Suche .pt-Modelle in: {model_dir}\n")

# Alle .pt-Dateien durchgehen
for filename in os.listdir(model_dir):
    if filename.endswith(".pt"):
        pt_path = os.path.join(model_dir, filename)
        onnx_path = os.path.join(model_dir, filename.replace(".pt", ".onnx"))

        if os.path.exists(onnx_path):
            print(f"⚠️  {filename} wurde bereits konvertiert, überspringe.")
            continue

        print(f"🔄 Konvertiere: {filename}")
        try:
            model = YOLO(pt_path)
            export_result = model.export(format="onnx", dynamic=False, simplify=True, imgsz=(416, 480), opset=12)
            if isinstance(export_result, (list, tuple)):
                exported_path = export_result[0]
            else:
                exported_path = export_result

            # Falls nötig: verschiebe .onnx in Zielverzeichnis
            if os.path.exists(exported_path) and exported_path != onnx_path:
                os.rename(exported_path, onnx_path)

            print(f"✅ Exportiert nach: {onnx_path}\n")

        except Exception as e:
            print(f"❌ Fehler bei {filename}: {e}\n")